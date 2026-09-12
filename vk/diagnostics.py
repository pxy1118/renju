"""Pure diagnostics: saturation, entropy, divergence, Q spread.

These are the numbers that say whether the training signal is alive. They are
plain functions over arrays so they can be unit tested without a network, a
game or a device, and the round report that combines them lives at the bottom.
"""
import numpy as np

from .records import VALUE_HEADS, head_mask, head_targets
from .targets import normalized_weights


def saturation(values, valid=None):
    """How concentrated a set of value labels or predictions is."""
    values = np.asarray(values, np.float64)
    if valid is not None:
        values = values[np.asarray(valid, bool)]
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"rows": 0, "mean_abs": None, "std": None,
                "abs_gt_0.9_share": None, "abs_lt_0.5_share": None}
    absolute = np.abs(finite)
    return {"rows": int(len(finite)), "mean_abs": float(absolute.mean()),
            "std": float(finite.std()),
            "abs_gt_0.9_share": float((absolute > 0.9).mean()),
            "abs_lt_0.5_share": float((absolute < 0.5).mean())}


def entropy(policy):
    """Mean Shannon entropy of a batch of distributions, in nats."""
    policy = np.asarray(policy, np.float64)
    if policy.ndim == 1:
        policy = policy[None, :]
    totals = policy.sum(axis=1, keepdims=True)
    safe = np.divide(policy, totals, out=np.zeros_like(policy), where=totals > 0)
    nonzero = safe > 0
    terms = np.where(nonzero, safe * np.log(np.where(nonzero, safe, 1.0)), 0.0)
    return float(-terms.sum(axis=1).mean()) if len(safe) else 0.0


def kl_divergence(p, q, mask=None):
    """Mean KL(p || q) over rows where p has mass, in nats."""
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    if p.ndim == 1:
        p, q = p[None, :], q[None, :]
    totals = p.sum(axis=1, keepdims=True)
    p = np.divide(p, totals, out=np.zeros_like(p), where=totals > 0)
    support = (p > 0) & (q > 0)
    if mask is not None:
        support = support & np.asarray(mask, bool)[:, None]
    rows = support.any(axis=1)
    if not rows.any():
        return 0.0
    terms = np.where(support, p * np.log(np.where(support, p, 1.0)
                                         / np.where(support, q, 1.0)), 0.0)
    return float(terms.sum(axis=1)[rows].mean())


def target_report(records):
    """What the produced positions say about the targets themselves."""
    if not len(records):
        return {}
    report = {"rows": int(len(records))}
    for head in VALUE_HEADS:
        report[f"target_saturation_{head}"] = saturation(head_targets(records, head))
    report["policy_valid_share"] = float((records["policy_valid"] != 0).mean())
    report["full_search_share"] = float((records["full_search"] != 0).mean())
    report["policy_target_entropy"] = entropy(
        records["policy"][records["policy_valid"] != 0].astype(np.float64)) \
        if (records["policy_valid"] != 0).any() else 0.0
    report["q_spread_mean"] = float(np.nanmean(records["q_spread"])) \
        if np.isfinite(records["q_spread"].astype(np.float64)).any() else 0.0
    report["policy_surprise_mean"] = float(np.nanmean(records["policy_surprise"].astype(np.float64))) \
        if np.isfinite(records["policy_surprise"].astype(np.float64)).any() else 0.0
    report["value_surprise_mean"] = float(np.nanmean(records["value_surprise"].astype(np.float64))) \
        if np.isfinite(records["value_surprise"].astype(np.float64)).any() else 0.0
    # Report the weights the sampler actually uses: normalised to mean one, so a
    # round whose surprises are all small does not look like a broken sampler.
    weights = normalized_weights(records["weight"].astype(np.float64))
    report["sample_weight_mean"] = float(weights.mean())
    report["sample_weight_max"] = float(weights.max())
    return report


def record_round(model, records, cfg, device, limit=1024, rng=None):
    """Network side of the round report: predictions against stored targets."""
    import torch

    if not len(records):
        return {}
    rng = rng if rng is not None else np.random.default_rng(0)
    count = min(int(limit), len(records))
    chosen = rng.choice(len(records), size=count, replace=False) if count < len(records) \
        else np.arange(len(records))
    batch = records[chosen]
    states = torch.tensor(batch["state"].astype(np.float32), dtype=torch.float32,
                          device=device)
    model.eval()
    with torch.inference_mode():
        logits, values = model(states)
        probability = torch.softmax(logits, dim=1).cpu().numpy()
        predicted = values.cpu().numpy()
    report = {"sampled_rows": int(count),
              "policy_entropy_network": entropy(probability),
              "kl_target_network": kl_divergence(
                  batch["policy"].astype(np.float64), probability,
                  mask=batch["policy_valid"] != 0)}
    for index, head in enumerate(VALUE_HEADS):
        mask = head_mask(batch, head)
        report[f"prediction_saturation_{head}"] = saturation(predicted[:, index], mask)
    return report
