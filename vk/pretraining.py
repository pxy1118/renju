"""Supervised pretraining from versioned teacher NPZ shards."""
from pathlib import Path
import json
import math
import time
import numpy as np
import torch

from .datasets import (DEFAULT_GAP_WEIGHTS, LEGACY_ACTION, TOPK_FIELDS, gap_weights,
                       load_split, topk_valid, weighted_indices)
from .network import DEFAULT_ARCH, Network, architecture
from .training import atomic_save, augment

__all__ = ["load_split", "pretrain"]


def selection_score(policy_ce, value_mse, policy_weight=1.0, value_weight=1.0):
    """The number the best checkpoint is chosen by.

    It must use the same weights as the training loss: with ``value_weight=0``
    a selection score that still counted MSE would keep picking checkpoints on
    the noise of an untrained value head.
    """
    return policy_weight * policy_ce + value_weight * value_mse


def _topk_slice(data, start, stop):
    """Teacher top-k winrate rows for optional diagnostics, or ``None``."""
    actions = data.get("teacher_topk_actions")
    winrates = data.get("teacher_topk_winrates")
    if actions is None or winrates is None:
        return None
    return actions[start:stop], winrates[start:stop]


def _metrics(model, data, device, batch_size=1024):
    total, ce, squared, absolute, top1, top5 = 0, 0.0, 0.0, 0.0, 0, 0
    topk_rows, topk_agreement, topk_outside, topk_regret = 0, 0, 0, 0.0
    topk_valid_rows = 0
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
            topk = _topk_slice(data, start, stop)
            if topk is not None:
                actions, winrates = topk
                valid = topk_valid(actions)
                if valid.any():
                    # Cost of the predicted move in raw winrate points, using
                    # Rapfi's own top-k from the same root search. Rows inherited
                    # from a format-2 source have no per-action winrate and are
                    # skipped instead of being scored as if they had.
                    top1_winrate = winrates[:, 0].astype(np.float64)
                    predicted_action = ranked[:, 0].cpu().numpy()
                    matched = valid & (actions == predicted_action[:, None])
                    inside = matched.any(1)
                    scored = np.where(matched, winrates.astype(np.float64), -np.inf).max(1)
                    scored = np.where(inside, scored, 0.0)
                    cost = top1_winrate - scored
                    chosen = inside & (top1_winrate < 0.95)
                    topk_rows += int(chosen.sum())
                    topk_regret += float(cost[chosen].sum())
                    topk_agreement += int((actions[:, 0] == predicted_action)[valid[:, 0]].sum())
                    topk_outside += int((~inside & valid[:, 0]).sum())
                    topk_valid_rows += int(valid[:, 0].sum())
            total += count
    metrics = {"policy_ce": ce / total, "value_mse": squared / total,
               "value_mae": absolute / total, "top1": top1 / total,
               "top5": top5 / total, "combined": (ce + squared) / total,
               "positions": total}
    if topk_valid_rows:
        metrics["topk_valid_rows"] = topk_valid_rows
        metrics["topk_scored_rows"] = topk_rows
        metrics["topk_predicted_is_best_share"] = topk_agreement / topk_valid_rows
        metrics["topk_predicted_outside_share"] = topk_outside / topk_valid_rows
        if topk_rows:
            metrics["topk_predicted_winrate_cost_mean"] = topk_regret / topk_rows
    return metrics


def pretrain(dataset, output, rule="freestyle", steps=20_000, batch_size=256,
             arch=DEFAULT_ARCH, device="cuda", seed=20260910,
             warmup_steps=500, learning_rate=3e-4, final_learning_rate=3e-5,
             weight_decay=1e-4, policy_weight=1.0, value_weight=1.0,
             mix_dataset=None, mix_share=0.5, critical_weighting=False,
             gap_buckets=DEFAULT_GAP_WEIGHTS):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Pretraining output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") not in (2, 3) or manifest.get("rule") != rule:
        raise ValueError("Teacher manifest format or rule mismatch")
    if policy_weight <= 0 and value_weight <= 0:
        raise ValueError("At least one of policy_weight and value_weight must be positive")
    if not 0.0 <= mix_share <= 1.0:
        raise ValueError(f"mix_share must be in [0, 1], got {mix_share}")
    train, validation, test = (load_split(dataset, name) for name in ("train", "validation", "test"))
    mixed = None
    if mix_dataset:
        mixed = load_split(mix_dataset, "train")
        if len(mixed["state"]) == 0:
            raise ValueError("Mixed dataset has no training rows")
    sampling = {}
    if critical_weighting:
        # Spend the budget where the choice matters. Weights enter the batch
        # *sampler*, not the loss, so the objective is unchanged and only the
        # per-position attention shifts -- the cheapest way to test whether the
        # decision-critical positions deserve more of the training budget.
        for name, split in (("teacher", train), ("mix", mixed)):
            if split is None:
                continue
            weights = gap_weights(split["teacher_topk_winrates"],
                                  gap_buckets, split["teacher_topk_actions"])
            usable = int(np.isfinite(split["teacher_topk_winrates"][:, 0].astype(np.float64)).sum())
            if not usable:
                raise ValueError(f"critical_weighting needs top-k winrates in the {name} dataset")
            sampling[name] = {"weights": weights, "mean": float(weights.mean()), "usable": usable}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Network(arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    width, _, _ = architecture(arch)
    cfg = {"rule": rule, "arch": arch, "channels": width, "blocks": model.blocks,
           "simulations": 400, "candidates": "tactical", "search": "mcts",
           "opening_mode": "sampled",
           "cpuct": 2.0, "temperature_moves": 20, "workers": 16, "games_per_round": 64,
           "train_steps": 100, "replay_capacity": 200000, "batch_size": 256,
           "learning_rate": 0.001, "weight_decay": 0.0001, "seed": seed,
           "eval_every": 5, "eval_pairs": 100,
           # Self-play produces roughly one round of samples before the first
           # update; a round cut short by the deadline must still unlock it, so
           # the threshold is set well below a full round's output.
           "min_replay_size": 256,
           "promotion_every": 5, "promotion_pairs": 100, "opening_plies": 8}
    # A policy-only run must not pick its best checkpoint on the noise of an
    # untrained value head, so the selection score mirrors the training loss.
    losses = {"policy_weight": float(policy_weight), "value_weight": float(value_weight)}
    weights = {**losses, "critical_weighting": bool(critical_weighting),
               "gap_weights": [list(bucket) for bucket in gap_buckets]
               if critical_weighting else None}
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
        source_name = "teacher"
        source = train
        if mixed is not None and rng.random() < mix_share:
            source, source_name = mixed, "mix"
        if source_name in sampling:
            indices = weighted_indices(sampling[source_name]["weights"], batch_size, rng)
        else:
            indices = rng.integers(len(source["state"]), size=batch_size)
        xs, ps = [], []
        for index in indices:
            x, pi = augment(source["state"][index], source["policy"][index],
                            int(rng.integers(4)), bool(rng.integers(2)))
            xs.append(x)
            ps.append(pi)
        x = torch.from_numpy(np.stack(xs).astype(np.float32)).to(device)
        policy = torch.from_numpy(np.stack(ps).astype(np.float32)).to(device)
        value = torch.from_numpy(source["value"][indices].astype(np.float32)).to(device)
        model.train()
        logits, predicted = model(x)
        policy_loss = -(policy * torch.log_softmax(logits, 1)).sum(1).mean()
        value_loss = (value - predicted).square().mean()
        loss = policy_weight * policy_loss + value_weight * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % min(250, steps) == 0 or step == steps:
            metrics = _metrics(model, validation, device)
            record = {"step": step, "learning_rate": lr, "train_loss": float(loss.detach()),
                      "elapsed_seconds": time.monotonic() - started, **weights, **metrics}
            if sampling:
                record["sampling"] = {name: {"mean_weight": item["mean"],
                                             "usable_rows": item["usable"]}
                                      for name, item in sampling.items()}
            with log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            score = selection_score(metrics["policy_ce"], metrics["value_mse"], **losses)
            if score < best_score:
                best_score = score
                atomic_save({"format": 1, "config": cfg, "model": model.state_dict(), "step": step,
                             "teacher": {"manifest": manifest, "dataset": str(Path(dataset).resolve()),
                                         "mix_dataset": str(Path(mix_dataset).resolve()) if mixed is not None else None,
                                         "mix_share": mix_share if mixed is not None else None,
                                         **weights},
                             "candidate_mode": cfg.get("candidates", "tactical")}, output / "best.pt")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = _metrics(model, test, device)
    from .evaluation import tactical_gate
    # With value_weight=0 the value head is untrained, so a gate that routes
    # through MCTS would test the thing that was deliberately not trained.
    gate_mode = "mcts" if value_weight > 0 else "policy"
    tactical = tactical_gate(model, cfg, device, mode=gate_mode)
    acceptance = {"top1": test_metrics["top1"] >= 0.45,
                  "top5": test_metrics["top5"] >= 0.80,
                  "tactical": tactical["passed"]}
    if value_weight > 0:
        acceptance["value_mae"] = test_metrics["value_mae"] <= 0.20
    report = {"best_step": best["step"], "validation_score": best_score, **weights,
              "sampling": {name: {"mean_weight": item["mean"], "usable_rows": item["usable"]}
                           for name, item in sampling.items()} or None,
              "gate_mode": gate_mode, "test": test_metrics, "tactical": tactical,
              "acceptance": acceptance,
              "acceptance_note": ("value_mae is not gated because value_weight=0"
                                  if value_weight == 0 else None),
              "accepted": all(acceptance.values()), "elapsed_seconds": time.monotonic() - started}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best["pretrain_report"] = report
    atomic_save(best, output / "best.pt")
    return report
