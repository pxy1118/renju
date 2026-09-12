"""Supervised pretraining from versioned teacher shards.

This is the Rapfi cold start. It shares the objective with self-play training,
so the same weights apply to the same terms; the only differences are the data
source and the shape of the curriculum (cosine learning rate, optional gap
weighting, a selection score that mirrors the loss).
"""
from pathlib import Path
import json
import math
import time
import numpy as np
import torch

from .config import DEFAULTS, defaults
from .datasets import (DEFAULT_GAP_WEIGHTS, FORMAT_VERSION, SUPPORTED_VERSIONS, gap_weights,
                       load_split, topk_row_valid, weighted_indices)
from .network import DEFAULT_ARCH, Network, final_head_index
from .objective import Objective, compute_loss, require_multi_head, tensor_batch
from .records import (ACTION_NONE, SCHEMA_VERSION, VALUE_HEADS, augment_batch, head_mask,
                       topk_valid)
from .storage import atomic_save

__all__ = ["load_split", "pretrain", "selection_score"]

# Horizon heads keep this ratio to the final head when a caller scales the
# value term with a single number, which is what the CLI exposes.
HORIZON_WEIGHT_RATIOS = {"mid": DEFAULTS["value_weight_mid"] / DEFAULTS["value_weight_final"],
                         "short": DEFAULTS["value_weight_short"] / DEFAULTS["value_weight_final"]}


def selection_score(policy_ce, value_mse, policy_weight=1.0, value_weights=None):
    """The number the best checkpoint is chosen by.

    It must use the same weights as the training loss: with a weight of zero a
    selection score that still counted that term would keep picking checkpoints
    on the noise of an untrained head.
    """
    total = policy_weight * policy_ce
    for head, weight in (value_weights or {}).items():
        mse = value_mse.get(head)
        if mse is not None:
            total += weight * mse
    return total


def _teacher_metrics(model, data, device, batch_size=1024):
    """Teacher-set metrics, every head masked by its own valid rows."""
    total, policy_ce = 0, 0.0
    top1 = top5 = 0
    squared = {head: 0.0 for head in VALUE_HEADS}
    absolute = {head: 0.0 for head in VALUE_HEADS}
    counts = {head: 0 for head in VALUE_HEADS}
    topk_rows = topk_agreement = topk_outside = 0
    topk_regret = 0.0
    topk_valid_rows = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            stop = min(len(data), start + batch_size)
            rows = data[start:stop]
            x = torch.from_numpy(rows["state"].astype(np.float32)).to(device)
            target = torch.from_numpy(rows["policy"].astype(np.float32)).to(device)
            valid_policy = torch.from_numpy(rows["policy_valid"] != 0).to(device)
            values = torch.from_numpy(rows["value"].astype(np.float32)).to(device)
            logits, predicted = model(x)
            count = stop - start
            if valid_policy.any():
                logp = torch.log_softmax(logits, 1)
                policy_ce += float((-(target * logp).sum(1))[valid_policy].sum())
            for index, head in enumerate(VALUE_HEADS):
                mask = torch.from_numpy(head_mask(rows, head)).to(device)
                if not mask.any():
                    continue
                delta = (predicted[:, index] - values[:, index])[mask]
                squared[head] += float(delta.square().sum())
                absolute[head] += float(delta.abs().sum())
                counts[head] += int(mask.sum())
            named = rows["teacher_best"] != ACTION_NONE
            if named.any():
                selected = np.flatnonzero(named)
                best = torch.from_numpy(rows["teacher_best"][selected].astype(np.int64)).to(device)
                ranked = logits[torch.from_numpy(selected).to(device)].topk(5, 1).indices
                top1 += int((ranked[:, 0] == best).sum())
                top5 += int((ranked == best[:, None]).any(1).sum())
                # Cost of the predicted move in raw winrate points, using
                # Rapfi own top-k from the same root search. Rows inherited
                # from a format-2 source have no per-action winrate and are
                # skipped instead of being scored as if they had.
                actions = rows["teacher_topk_actions"][selected]
                winrates = rows["teacher_topk_winrates"][selected].astype(np.float64)
                slots = topk_valid(actions)
                rows_valid = slots[:, 0]
                if rows_valid.any():
                    top1_winrate = winrates[:, 0]
                    predicted_action = ranked[:, 0].cpu().numpy()
                    matched = slots & (actions == predicted_action[:, None])
                    inside = matched.any(1)
                    scored = np.where(matched, winrates, -np.inf).max(1)
                    scored = np.where(inside, scored, 0.0)
                    cost = top1_winrate - scored
                    chosen = inside & (top1_winrate < 0.95)
                    topk_rows += int(chosen.sum())
                    topk_regret += float(cost[chosen].sum())
                    topk_agreement += int((actions[:, 0] == predicted_action)[rows_valid].sum())
                    topk_outside += int((~inside & rows_valid).sum())
                    topk_valid_rows += int(rows_valid.sum())
            total += count
    metrics = {"positions": total,
               "policy_ce": policy_ce / total if total else 0.0,
               "top1": top1 / total if total else 0.0,
               "top5": top5 / total if total else 0.0}
    value_mse = {}
    for head in VALUE_HEADS:
        rows = counts[head]
        metrics[f"value_mse_{head}"] = float(squared[head] / rows) if rows else None
        metrics[f"value_mae_{head}"] = float(absolute[head] / rows) if rows else None
        metrics[f"value_rows_{head}"] = int(rows)
        value_mse[head] = metrics[f"value_mse_{head}"]
    # The historical single-head names stay the final head, which is the one the
    # acceptance gates are written against.
    metrics["value_mse"] = metrics["value_mse_final"]
    metrics["value_mae"] = metrics["value_mae_final"]
    metrics["combined"] = metrics["policy_ce"] + (metrics["value_mse"] or 0.0)
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
    if manifest.get("format_version") not in list(SUPPORTED_VERSIONS) + [FORMAT_VERSION] \
            or manifest.get("rule") != rule:
        raise ValueError("Teacher manifest format or rule mismatch")
    if policy_weight <= 0 and value_weight <= 0:
        raise ValueError("At least one of policy_weight and value_weight must be positive")
    if not 0.0 <= mix_share <= 1.0:
        raise ValueError(f"mix_share must be in [0, 1], got {mix_share}")
    train, validation, test = (load_split(dataset, name)
                               for name in ("train", "validation", "test"))
    mixed = None
    if mix_dataset:
        mixed = load_split(mix_dataset, "train")
        if len(mixed) == 0:
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
            weights = gap_weights(split["teacher_topk_winrates"], gap_buckets,
                                  split["teacher_topk_actions"])
            usable = int(topk_row_valid(split["teacher_topk_actions"]).sum())
            if not usable:
                raise ValueError(f"critical_weighting needs top-k winrates in the {name} dataset")
            sampling[name] = {"weights": weights, "mean": float(weights.mean()), "usable": usable}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = require_multi_head(Network(arch)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # The configuration stored with the distilled weights is the same table the
    # training loop reads, with the values a distillation run is known to use.
    # Self-play produces roughly one round of samples before the first update;
    # a round cut short by the deadline must still unlock it, so the threshold
    # is set well below a full round of output.
    cfg = defaults(rule, arch=arch, simulations=400, workers=16, games_per_round=64,
                   train_steps=100, replay_capacity=200000, batch_size=256,
                   learning_rate=0.001, weight_decay=0.0001, seed=seed, eval_every=5,
                   eval_pairs=100, min_replay_size=256, promotion_every=5,
                   promotion_pairs=100, opening_plies=8,
                   policy_weight=float(policy_weight), value_weight_final=float(value_weight),
                   value_weight_mid=float(value_weight) * HORIZON_WEIGHT_RATIOS["mid"],
                   value_weight_short=float(value_weight) * HORIZON_WEIGHT_RATIOS["short"])
    objective = Objective.from_config(cfg)
    value_weights = {head: objective.value_weights[index]
                     for index, head in enumerate(VALUE_HEADS)}
    losses = {"policy_weight": objective.policy_weight,
              "value_weights": {head: weight for head, weight in value_weights.items() if weight}}
    weights = {**losses, "critical_weighting": bool(critical_weighting),
               "gap_weights": [list(bucket) for bucket in gap_buckets]
               if critical_weighting else None}
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
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
        source_name, source = "teacher", train
        if mixed is not None and rng.random() < mix_share:
            source, source_name = mixed, "mix"
        if source_name in sampling:
            indices = weighted_indices(sampling[source_name]["weights"], batch_size, rng)
        else:
            indices = rng.integers(len(source), size=batch_size)
        rows = source[indices]
        states, policies = augment_batch(rows["state"], rows["policy"].astype(np.float32), rng)
        tensors = tensor_batch(rows, device, states, policies)
        model.train()
        logits, predictions = model(tensors["state"])
        report = compute_loss(logits, predictions, tensors, cfg)
        optimizer.zero_grad(set_to_none=True)
        report["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % min(250, steps) == 0 or step == steps:
            metrics = _teacher_metrics(model, validation, device)
            record = {"step": step, "learning_rate": lr,
                      "train_loss": float(report["loss"].detach()),
                      "train_policy_loss": float(report["policy_loss"].detach()),
                      "train_value_loss": float(report["value_loss"].detach()),
                      "elapsed_seconds": time.monotonic() - started, **weights, **metrics}
            if sampling:
                record["sampling"] = {name: {"mean_weight": item["mean"],
                                             "usable_rows": item["usable"]}
                                      for name, item in sampling.items()}
            with log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            score = selection_score(metrics["policy_ce"],
                                    {head: metrics[f"value_mse_{head}"] for head in VALUE_HEADS},
                                    objective.policy_weight, value_weights)
            if score < best_score:
                best_score = score
                atomic_save({"format": 2, "schema": SCHEMA_VERSION, "config": cfg,
                             "value_heads": list(VALUE_HEADS),
                             "model": model.state_dict(), "step": step,
                             "teacher": {"manifest": manifest, "dataset": str(Path(dataset).resolve()),
                                         "mix_dataset": str(Path(mix_dataset).resolve()) if mixed is not None else None,
                                         "mix_share": mix_share if mixed is not None else None,
                                         **weights},
                             "hard_rules": cfg["hard_rules"]}, output / "best.pt")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = _teacher_metrics(model, test, device)
    from .evaluation import tactical_gate
    # With every value weight at zero the value heads are untrained, so a gate
    # that routes through MCTS would test the thing that was deliberately not
    # trained.
    gate_mode = "mcts" if value_weights["final"] > 0 else "policy"
    tactical = tactical_gate(model, cfg, device, mode=gate_mode)
    acceptance = {"top1": test_metrics["top1"] >= 0.45,
                  "top5": test_metrics["top5"] >= 0.80,
                  "tactical": tactical["passed"]}
    if value_weights["final"] > 0:
        acceptance["value_mae"] = test_metrics["value_mae"] <= 0.20
    report = {"best_step": best["step"], "validation_score": best_score, **weights,
              "sampling": {name: {"mean_weight": item["mean"], "usable_rows": item["usable"]}
                           for name, item in sampling.items()} or None,
              "gate_mode": gate_mode, "test": test_metrics, "tactical": tactical,
              "acceptance": acceptance,
              "acceptance_note": ("value_mae is not gated because the value weight is zero"
                                  if value_weights["final"] == 0 else None),
              "accepted": all(acceptance.values()), "elapsed_seconds": time.monotonic() - started}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    best["pretrain_report"] = report
    atomic_save(best, output / "best.pt")
    return report
