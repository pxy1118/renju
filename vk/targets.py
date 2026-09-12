"""Pure numpy construction of every training target.

Target construction is deliberately separate from the search that produces the
visit counts and from the loss that consumes the targets: each piece is a small
function with an obvious contract, and each one can be tested without a network,
a game or a torch device.
"""
import numpy as np

from .records import VALUE_HEADS, WINNER_UNKNOWN, head_bit, head_index


def policy_target(visits, prior, noise=None, noise_weight=0.0,
                  prune_prop=0.02, prune_min_count=2, correct_noise=True):
    """The supervised policy target: noise-corrected, pruned, normalised.

    Dirichlet exploration is what makes the search explore, but it is not what
    we want to teach. Removing its expected visit contribution first, then
    dropping children that the search never really visited, keeps the genuinely
    contested moves and removes the tail that only exists because of noise.
    """
    visits = np.asarray(visits, np.float64)
    total = visits.sum()
    if total <= 0:
        return np.zeros(len(visits), np.float32)
    adjusted = visits
    if correct_noise and noise is not None and noise_weight > 0:
        noise = np.asarray(noise, np.float64)
        corrected = np.maximum(0.0, visits - noise_weight * noise * total)
        if corrected.sum() > 0:
            adjusted = corrected
    threshold = max(float(prune_min_count), float(prune_prop) * adjusted.max())
    kept = np.where(adjusted >= threshold, adjusted, 0.0)
    if kept.sum() <= 0:
        kept = np.zeros_like(adjusted)
        kept[int(np.argmax(visits))] = 1.0
    return (kept / kept.sum()).astype(np.float32)


def soft_target(policy, temperature=2.0):
    """The target raised to 1/T and renormalised: a softer supervision signal.

    With T=1 this is the target itself; larger T pulls probability mass towards
    the moves the search also considered, which is the restoring force that
    keeps a policy head from collapsing onto one move per position.
    """
    policy = np.maximum(np.asarray(policy, np.float64), 0.0)
    if policy.sum() <= 0:
        return policy.astype(np.float32)
    policy = policy / policy.sum()
    temperature = max(float(temperature), 1e-6)
    softened = np.power(policy, 1.0 / temperature)
    softened /= softened.sum()
    return softened.astype(np.float32)


def value_targets(search_values, winner, short_plies, mid_plies):
    """Multi-scale value targets for one finished game.

    search_values[t] is the root Q of the search run at ply t, from that ply's
    own side-to-move perspective; NaN means that ply has no search value. The
    final head always targets the true outcome. A horizon head targets the
    outcome when the game ends inside the horizon and otherwise bootstraps the
    search value at the horizon, sign-flipped once per ply because the side to
    move alternates.

    Returns (values[plies, heads], valid bitmask[plies], stats).
    """
    search_values = np.asarray(search_values, np.float64)
    plies = len(search_values)
    values = np.full((plies, len(VALUE_HEADS)), np.nan, np.float32)
    valid = np.zeros(plies, np.uint8)
    stats = {"rows": plies, "short_bootstrap": 0, "mid_bootstrap": 0, "exact": 0}
    if plies == 0:
        return values, valid, stats
    outcome_known = int(winner) != WINNER_UNKNOWN
    signs = np.where(np.arange(plies) % 2 == 0, 1.0, -1.0)
    if outcome_known:
        outcome = signs * float(winner)
        values[:, head_index("final")] = outcome
        valid |= head_bit("final")
        stats["exact"] = plies
    for head, horizon in (("short", int(short_plies)), ("mid", int(mid_plies))):
        index, bit = head_index(head), head_bit(head)
        for ply in range(plies):
            if ply + horizon >= plies:
                if not outcome_known:
                    continue
                value = signs[ply] * float(winner)
            else:
                bootstrap = search_values[ply + horizon]
                if not np.isfinite(bootstrap):
                    continue
                value = ((-1.0) ** horizon) * bootstrap
                stats[f"{head}_bootstrap"] += 1
            values[ply, index] = value
            valid[ply] |= bit
    return values, valid, stats


def surprise_weights(policy_surprise, value_surprise, cfg):
    """Per-row sampling weight from how surprising the search result was.

    The weight is a convex mix of a constant and a capped, scaled surprise, so
    a single anomalous position can never take over the batch while positions
    the network mispredicted still get more attention.
    """
    policy_surprise = np.nan_to_num(np.asarray(policy_surprise, np.float64), nan=0.0)
    value_surprise = np.nan_to_num(np.asarray(value_surprise, np.float64), nan=0.0)
    raw = (float(cfg["surprise_policy_weight"]) * policy_surprise
           + float(cfg["surprise_value_weight"]) * value_surprise)
    raw = np.maximum(raw, 0.0)
    reference = float(cfg["surprise_ref"])
    if reference <= 0:
        scaled = np.zeros_like(raw)
    else:
        scaled = np.clip(raw / reference, 0.0, float(cfg["surprise_cap"]))
    share = float(cfg["surprise_uniform_share"])
    return ((1.0 - share) + share * scaled).astype(np.float32)


def normalized_weights(weights):
    """Sampling weights rescaled to mean 1, so the softmax probe stays sane."""
    weights = np.asarray(weights, np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Sampling weights must be finite, non-negative, not all zero")
    return weights / weights.mean()
