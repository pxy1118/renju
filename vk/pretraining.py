"""Supervised pretraining from versioned teacher NPZ shards."""
from pathlib import Path
import json
import math
import time
import numpy as np
import torch

from .network import Network
from .training import atomic_save, augment


def load_split(root, split):
    paths = sorted((Path(root) / split).glob("shard-*.npz"))
    if not paths:
        raise ValueError(f"Teacher dataset has no {split} shards")
    fields = {key: [] for key in ("state", "policy", "value", "game_id", "ply", "teacher_best", "teacher_nodes")}
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != set(fields):
                raise ValueError(f"Invalid teacher shard fields: {path}")
            if len(shard["state"]) > 4096:
                raise ValueError(f"Teacher shard exceeds 4096 rows: {path}")
            for key in fields:
                fields[key].append(shard[key])
    return {key: np.concatenate(parts) for key, parts in fields.items()}


def _metrics(model, data, device, batch_size=1024):
    total, ce, squared, absolute, top1, top5 = 0, 0.0, 0.0, 0.0, 0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["state"]), batch_size):
            stop = min(len(data["state"]), start + batch_size)
            x = torch.from_numpy(data["state"][start:stop].astype(np.float32)).to(device)
            target = torch.from_numpy(data["policy"][start:stop].astype(np.float32)).to(device)
            values = torch.from_numpy(data["value"][start:stop].astype(np.float32)).to(device)
            best = torch.from_numpy(data["teacher_best"][start:stop].astype(np.int64)).to(device)
            logits, predicted = model(x)
            logp = torch.log_softmax(logits, 1)
            count = stop - start
            ce += float((-(target * logp).sum(1)).sum())
            delta = predicted - values
            squared += float(delta.square().sum())
            absolute += float(delta.abs().sum())
            ranked = logits.topk(5, 1).indices
            top1 += int((ranked[:, 0] == best).sum())
            top5 += int((ranked == best[:, None]).any(1).sum())
            total += count
    return {"policy_ce": ce / total, "value_mse": squared / total,
            "value_mae": absolute / total, "top1": top1 / total,
            "top5": top5 / total, "combined": (ce + squared) / total,
            "positions": total}


def pretrain(dataset, output, rule="freestyle", steps=20_000, batch_size=256,
             channels=64, blocks=6, device="cuda", seed=20260910,
             warmup_steps=500, learning_rate=3e-4, final_learning_rate=3e-5,
             weight_decay=1e-4):
    if rule != "freestyle":
        raise ValueError("The first teacher phase supports freestyle only")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Pretraining output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2 or manifest.get("rule") != rule:
        raise ValueError("Teacher manifest format or rule mismatch")
    train, validation, test = (load_split(dataset, name) for name in ("train", "validation", "test"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Network(channels, blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    cfg = {"rule": rule, "channels": channels, "blocks": blocks, "simulations": 400,
           "cpuct": 2.0, "temperature_moves": 20, "workers": 16, "games_per_round": 64,
           "train_steps": 100, "replay_capacity": 200000, "batch_size": 256,
           "learning_rate": 0.001, "weight_decay": 0.0001, "seed": seed,
           "eval_every": 5, "eval_pairs": 100, "min_replay_size": 20000,
           "promotion_every": 5, "promotion_pairs": 100}
    best_score, started = math.inf, time.monotonic()
    log_path = output / "metrics.jsonl"
    for step in range(1, steps + 1):
        if step <= warmup_steps:
            lr = learning_rate * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            lr = final_learning_rate + 0.5 * (learning_rate - final_learning_rate) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = rng.integers(len(train["state"]), size=batch_size)
        xs, ps = [], []
        for index in indices:
            x, pi = augment(train["state"][index], train["policy"][index],
                            int(rng.integers(4)), bool(rng.integers(2)))
            xs.append(x)
            ps.append(pi)
        x = torch.from_numpy(np.stack(xs).astype(np.float32)).to(device)
        policy = torch.from_numpy(np.stack(ps).astype(np.float32)).to(device)
        value = torch.from_numpy(train["value"][indices].astype(np.float32)).to(device)
        model.train()
        logits, predicted = model(x)
        policy_loss = -(policy * torch.log_softmax(logits, 1)).sum(1).mean()
        value_loss = (value - predicted).square().mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % min(250, steps) == 0 or step == steps:
            metrics = _metrics(model, validation, device)
            record = {"step": step, "learning_rate": lr, "train_loss": float(loss.detach()),
                      "elapsed_seconds": time.monotonic() - started, **metrics}
            with log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if metrics["combined"] < best_score:
                best_score = metrics["combined"]
                atomic_save({"format": 1, "config": cfg, "model": model.state_dict(), "step": step,
                             "teacher": {"manifest": manifest, "dataset": str(Path(dataset).resolve())},
                             "candidate_mode": "tactical_square3_line4"}, output / "best.pt")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = _metrics(model, test, device)
    from .evaluation import tactical_gate
    tactical = tactical_gate(model, cfg, device)
    acceptance = {"top1": test_metrics["top1"] >= 0.45,
                  "top5": test_metrics["top5"] >= 0.80,
                  "value_mae": test_metrics["value_mae"] <= 0.20,
                  "tactical": tactical["passed"]}
    report = {"best_step": best["step"], "validation_combined": best_score,
              "test": test_metrics, "tactical": tactical, "acceptance": acceptance,
              "accepted": all(acceptance.values()), "elapsed_seconds": time.monotonic() - started}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best["pretrain_report"] = report
    atomic_save(best, output / "best.pt")
    return report
