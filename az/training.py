import json
import os
from pathlib import Path
import random
import time
from collections import deque
import numpy as np
import torch
from .network import Network, Evaluator
from .selfplay import collect

DEFAULTS = dict(rule="freestyle", channels=64, blocks=6, simulations=200, cpuct=2.0,
                temperature_moves=20, workers=16, games_per_round=32, train_steps=200,
                replay_capacity=100000, batch_size=256, learning_rate=0.001,
                weight_decay=0.0001, seed=20260910, eval_every=10, eval_pairs=10)


def augment(x, pi, rotation, mirror):
    x = np.rot90(x, rotation, axes=(-2, -1))
    pi = np.rot90(pi.reshape(15,15), rotation)
    if mirror:
        x, pi = x[..., ::-1], pi[:, ::-1]
    return x.copy(), pi.reshape(225).copy()


def update(model, optimizer, replay, batch_size, device, rng):
    indices = rng.integers(len(replay), size=batch_size)
    xs, ps, zs = [], [], []
    for i in indices:
        x, pi, z = replay[int(i)]
        x, pi = augment(x, pi, int(rng.integers(4)), bool(rng.integers(2)))
        xs.append(x)
        ps.append(pi)
        zs.append(z)
    x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    pi = torch.tensor(np.stack(ps), dtype=torch.float32, device=device)
    pi = pi / pi.sum(dim=1, keepdim=True)
    z = torch.tensor(zs, dtype=torch.float32, device=device)
    model.train()
    logits, values = model(x)
    policy_loss = -(pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = (z-values).square().mean()
    loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    probs = torch.softmax(logits.detach(), dim=1)
    entropy = -(probs * torch.log_softmax(logits.detach(), dim=1)).sum(1).mean()
    return {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()), "policy_entropy": float(entropy)}


def atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def checkpoint_path(root, name):
    root = Path(root)
    if name == "latest":
        paths = sorted(root.glob("checkpoint-*.pt"))
        if not paths:
            raise FileNotFoundError(f"No checkpoint in {root}")
        return paths[-1]
    if name == "best":
        return root / "best.pt"
    return Path(name)


def load_checkpoint(path, rule):
    # Local trusted checkpoints only: optimizer/RNG/replay require pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != 1 or state["config"]["rule"] != rule:
        raise ValueError("Checkpoint format or rule mismatch")
    return state


def save(root, cfg, model, optimizer, replay, rng, round_id, step, total_games, pending_steps=0):
    state = {"format": 1, "config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "replay": list(replay), "round": round_id, "step": step, "total_games": total_games,
             "pending_steps": pending_steps,
             "rng": rng.bit_generator.state, "python_rng": random.getstate(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None}
    path = Path(root) / f"checkpoint-{round_id:08d}-{step:010d}.pt"
    atomic_save(state, path)
    for old in sorted(Path(root).glob("checkpoint-*.pt"))[:-3]:
        old.unlink()
    return path


def append_json(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(cfg, root, device, seconds, resume=None, stop=lambda: False, max_rounds=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not resume and list(root.glob("checkpoint-*.pt")):
        raise ValueError("Existing training found: use --resume latest or a different --output.")
    state = load_checkpoint(checkpoint_path(root, resume), cfg["rule"]) if resume else None
    if state:
        if "optimizer" not in state:
            raise ValueError("Inference-only best checkpoint cannot resume training; use latest.")
        changed = {key for key in set(cfg) | set(state["config"])
                   if cfg.get(key) != state["config"].get(key)}
        incompatible = changed - {"workers"}
        if incompatible:
            raise ValueError(f"Resume configuration differs in incompatible fields: {sorted(incompatible)}")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = Network(cfg["channels"], cfg["blocks"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    replay = deque(maxlen=cfg["replay_capacity"])
    round_id = step = total_games = pending_steps = 0
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        replay.extend(state["replay"])
        round_id, step, total_games = state["round"], state["step"], state["total_games"]
        pending_steps = state.get("pending_steps",0)
        rng.bit_generator.state = state["rng"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    start = time.monotonic()
    deadline = start + seconds
    evaluator = Evaluator(model, device)
    rounds_this_run = 0
    last_path = None
    while time.monotonic() < deadline and not stop():
        if max_rounds is not None and rounds_this_run >= max_rounds:
            break
        rounds_this_run += 1
        games = []
        perf = {"inference_positions":0,"batches":0,"average_inference_batch_size":0,
                "largest_inference_batch_size":0,"seconds":0}
        if pending_steps == 0:
            round_id += 1
            seeds = rng.integers(0, 2**32, cfg["games_per_round"])
            data, games, perf = collect(cfg, evaluator, seeds, deadline, stop)
            replay.extend(data)
            total_games += len(games)
            for game in games:
                append_json(root / "games.jsonl", {"round": round_id, **game})
            pending_steps = cfg["train_steps"] if games else 0
        metrics = {}
        updates = 0
        if replay:
            for _ in range(pending_steps):
                if stop() or time.monotonic() >= deadline:
                    break
                metrics = update(model, optimizer, replay, cfg["batch_size"], device, rng)
                step += 1
                updates += 1
                pending_steps -= 1
        last_path = save(root, cfg, model, optimizer, replay, rng, round_id, step, total_games,pending_steps)
        record = {"round": round_id, "step": step, "updates": updates, "games": len(games),
                  "total_games": total_games, "replay_size": len(replay), "pending_steps":pending_steps,
                  "elapsed_seconds": time.monotonic()-start, "remaining_seconds": max(0,deadline-time.monotonic()),
                  "black_wins": sum(g["winner"] == 1 for g in games),
                  "white_wins": sum(g["winner"] == -1 for g in games),
                  "draws": sum(g["winner"] == 0 for g in games),
                  "black_win_rate": sum(g["winner"] == 1 for g in games)/len(games) if games else None,
                  "white_win_rate": sum(g["winner"] == -1 for g in games)/len(games) if games else None,
                  "draw_rate": sum(g["winner"] == 0 for g in games)/len(games) if games else None,
                  "completed_game_simulations_per_second": sum(g["simulations"] for g in games)/max(perf["seconds"], 1e-6),
                  **perf, **metrics}
        append_json(root / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        if not (root / "best.pt").exists() and updates:
            atomic_save({"format": 1, "config": cfg, "model": model.state_dict(), "step": step}, root / "best.pt")
        if updates and pending_steps == 0 and round_id % cfg["eval_every"] == 0 and not stop() and time.monotonic() < deadline:
            from .evaluation import evaluate_suite
            result = evaluate_suite(model, cfg, device, root, cfg["eval_pairs"], deadline, stop)
            append_json(root / "evaluations.jsonl", {"round": round_id, **result})
            comparison = result.get("historical", {})
            if comparison.get("complete") and comparison.get("score", 0) > 0.55:
                atomic_save({"format": 1, "config": cfg, "model": model.state_dict(), "step": step}, root / "best.pt")
    if last_path is None:
        last_path = save(root, cfg, model, optimizer, replay, rng, round_id, step, total_games,pending_steps)
    return {"checkpoint": str(last_path), "step": step, "total_games": total_games,
            "elapsed_seconds": time.monotonic()-start}
