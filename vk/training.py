import json
import os
from pathlib import Path
import random
import time
import copy
from collections import deque
import numpy as np
import torch
from .network import DEFAULT_ARCH, Network, Evaluator, architecture_of
from .selfplay import collect

DEFAULTS = dict(rule="freestyle", arch=DEFAULT_ARCH, channels=128, blocks=10,
                simulations=200, cpuct=2.0,
                temperature_moves=20, workers=16, games_per_round=32, train_steps=200,
                replay_capacity=100000, batch_size=256, learning_rate=0.001,
                weight_decay=0.0001, seed=20260910, eval_every=10, eval_pairs=10,
                min_replay_size=1, promotion_every=10, promotion_pairs=10,
                opening_plies=8)

# Fields a resume may legitimately change: both describe how data is gathered
# in this process, not what the stored model and optimizer mean.
RESUME_FREE_FIELDS = {"workers", "opening_plies"}


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


def save(root, cfg, model, optimizer, replay, rng, round_id, step, total_games, pending_steps=0,
         champion_model=None, champion_optimizer=None, champion_step=0, teacher=None):
    state = {"format": 1, "config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "replay": list(replay), "round": round_id, "step": step, "total_games": total_games,
             "pending_steps": pending_steps,
             "rng": rng.bit_generator.state, "python_rng": random.getstate(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None}
    state.update({"candidate_mode": "tactical_square3_line4",
                  "champion_model": champion_model, "champion_optimizer": champion_optimizer,
                  "champion_step": champion_step, "teacher": teacher})
    path = Path(root) / f"checkpoint-{round_id:08d}-{step:010d}.pt"
    atomic_save(state, path)
    for old in sorted(Path(root).glob("checkpoint-*.pt"))[:-3]:
        old.unlink()
    return path


def append_json(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reconcile_jsonl(path, committed_round):
    """Remove diagnostics written after the checkpoint selected for resume."""
    path = Path(path)
    if not path.exists():
        return 0
    kept, removed = [], 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            removed += 1
            continue
        if record.get("round", -1) <= committed_round:
            kept.append(line)
        else:
            removed += 1
    if removed:
        temporary = path.with_suffix(path.suffix + ".reconcile.tmp")
        temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        os.replace(temporary, path)
    return removed


def train(cfg, root, device, seconds, resume=None, stop=lambda: False, max_rounds=None,
          init_checkpoint=None):
    root = Path(root)
    if resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if not resume and root.exists() and any(root.iterdir()):
        raise ValueError("Existing output found: use --resume latest or a different --output.")
    root.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(checkpoint_path(root, resume), cfg["rule"]) if resume else None
    if state:
        if "optimizer" not in state:
            raise ValueError("Inference-only best checkpoint cannot resume training; use latest.")
        # Older checkpoints predate fields added to DEFAULTS since; compare them
        # at their current default rather than reporting a spurious mismatch.
        stored = {key: state["config"].get(key, value) for key, value in DEFAULTS.items()}
        stored.update({key: value for key, value in state["config"].items()
                       if key not in DEFAULTS})
        stored["arch"] = architecture_of(state["config"])
        changed = {key for key in set(cfg) | set(stored) if cfg.get(key) != stored.get(key)}
        incompatible = changed - RESUME_FREE_FIELDS
        if incompatible:
            raise ValueError(f"Resume configuration differs in incompatible fields: {sorted(incompatible)}")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = Network(cfg["arch"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    replay = deque(maxlen=cfg["replay_capacity"])
    round_id = step = total_games = pending_steps = 0
    teacher = None
    if init_checkpoint:
        initial = load_checkpoint(checkpoint_path(root, init_checkpoint), cfg["rule"])
        if architecture_of(initial["config"]) != cfg["arch"]:
            raise ValueError("Initial checkpoint architecture mismatch")
        if initial.get("pretrain_report") and not initial["pretrain_report"].get("accepted"):
            raise ValueError("Initial pretraining checkpoint did not pass the formal acceptance gates")
        model.load_state_dict(initial["model"])
        teacher = initial.get("teacher")
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
        champion_state = copy.deepcopy(state.get("champion_model") or state["model"])
        champion_optimizer = copy.deepcopy(state.get("champion_optimizer") or optimizer.state_dict())
        champion_step = state.get("champion_step", step)
        teacher = state.get("teacher")
        removed = {name: reconcile_jsonl(root / name, round_id)
                   for name in ("games.jsonl", "metrics.jsonl", "evaluations.jsonl")}
        if any(removed.values()):
            print(json.dumps({"resume_reconciled": removed,
                              "checkpoint_round": round_id}), flush=True)
        del state
    else:
        champion_state = copy.deepcopy(model.state_dict())
        champion_optimizer = copy.deepcopy(optimizer.state_dict())
        champion_step = step
    champion = Network(cfg["arch"]).to(device)
    champion.load_state_dict(champion_state)
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    start = time.monotonic()
    deadline = start + seconds
    evaluator = Evaluator(champion, device)
    minimum = cfg.get("min_replay_size", 1)
    if not (root / "best.pt").exists():
        atomic_save({"format": 1, "config": cfg, "model": champion_state,
                     "step": champion_step, "teacher": teacher,
                     "candidate_mode": "tactical_square3_line4"}, root / "best.pt")
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
            pending_steps = cfg["train_steps"] if games and len(replay) >= minimum else 0
        trainable = len(replay) >= minimum
        metrics = {}
        updates = 0
        if trainable:
            for _ in range(pending_steps):
                if stop() or time.monotonic() >= deadline:
                    break
                metrics = update(model, optimizer, replay, cfg["batch_size"], device, rng)
                step += 1
                updates += 1
                pending_steps -= 1
        champion_result = None
        if (updates and pending_steps == 0 and
                round_id % cfg.get("promotion_every", cfg.get("eval_every", 10)) == 0 and
                not stop() and time.monotonic() < deadline):
            from .evaluation import match, tactical_gate
            tactics = tactical_gate(model, cfg, device)
            arena = match(model, cfg, device, champion, cfg.get("promotion_pairs", 100),
                          deadline, stop, sequential=True) if tactics["passed"] else {}
            promoted = tactics["passed"] and arena.get("wilson_lower", 0) > 0.5
            champion_result = {"promoted": promoted, "tactical": tactics, "arena": arena}
            if promoted:
                champion_state = copy.deepcopy(model.state_dict())
                champion_optimizer = copy.deepcopy(optimizer.state_dict())
                champion_step = step
                champion.load_state_dict(champion_state)
                atomic_save({"format": 1, "config": cfg, "model": champion_state,
                             "step": champion_step, "teacher": teacher,
                             "candidate_mode": "tactical_square3_line4"}, root / "best.pt")
            else:
                model.load_state_dict(champion_state)
                optimizer.load_state_dict(champion_optimizer)
            append_json(root / "evaluations.jsonl", {"round": round_id, **champion_result})
        last_path = save(root, cfg, model, optimizer, replay, rng, round_id, step, total_games,
                         pending_steps, champion_state, champion_optimizer, champion_step, teacher)
        def average(name):
            values = [game.get(name) for game in games if game.get(name) is not None]
            return float(np.mean(values)) if values else None
        # Opening length is the number of moves that preceded search, i.e. what
        # balanced_opening supplied. Samples cover only the searched moves.
        openings = [len(game["opening"]) for game in games if game.get("opening") is not None]
        mean_opening = float(np.mean(openings)) if openings else None
        record = {"round": round_id, "step": step, "updates": updates, "games": len(games),
                  "total_games": total_games, "replay_size": len(replay), "pending_steps":pending_steps,
                  "trainable": trainable, "min_replay_size": minimum,
                  "opening_distinct": len({tuple(g["opening"]) for g in games if g.get("opening")}),
                  "opening_sample_size": mean_opening,
                  "elapsed_seconds": time.monotonic()-start, "remaining_seconds": max(0,deadline-time.monotonic()),
                  "black_wins": sum(g["winner"] == 1 for g in games),
                  "white_wins": sum(g["winner"] == -1 for g in games),
                  "draws": sum(g["winner"] == 0 for g in games),
                  "black_win_rate": sum(g["winner"] == 1 for g in games)/len(games) if games else None,
                  "white_win_rate": sum(g["winner"] == -1 for g in games)/len(games) if games else None,
                  "draw_rate": sum(g["winner"] == 0 for g in games)/len(games) if games else None,
                  "candidate_count_mean": average("candidate_count_mean"),
                  "forced_win_count": sum(g.get("forced_win_count", 0) for g in games),
                  "forced_defense_count": sum(g.get("forced_defense_count", 0) for g in games),
                  "strategic_count": sum(g.get("strategic_count", 0) for g in games),
                  "search_max_depth": max((g.get("search_max_depth", 0) for g in games), default=0),
                  "search_prior_kl_mean": average("search_prior_kl_mean"),
                  "value_abs_mean": average("value_abs_mean"),
                  "champion": champion_result,
                  "completed_game_simulations_per_second": sum(g["simulations"] for g in games)/max(perf["seconds"], 1e-6),
                  **perf, **metrics}
        append_json(root / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
    if last_path is None:
        last_path = save(root, cfg, model, optimizer, replay, rng, round_id, step, total_games,
                         pending_steps, champion_state, champion_optimizer, champion_step, teacher)
    return {"checkpoint": str(last_path), "step": step, "total_games": total_games,
            "elapsed_seconds": time.monotonic()-start}
