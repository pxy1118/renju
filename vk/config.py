"""Every training, self-play, objective and data knob, in one place.

One declarative table, one validator, and one place that knows how a stored
configuration may legitimately differ from the live one. The training loop,
pretraining, the CLI and every evaluation tool read this module, so a loss
weight or a search budget can never be hardcoded in two places and drift.
"""
from pathlib import Path
import json
import numpy as np

from .network import ARCHITECTURES, architecture, architecture_of
from .records import VALUE_HEADS

SCHEMA_VERSION = 1

# Keys that are not numbers: free text or paths, a choice from a fixed set, or
# a boolean flag.
TEXT = ("rule", "arch", "opening_book")
CHOICES = {"hard_rules": ("forced", "none"),
           "search_bias": ("none", "tactical"),
           "search": ("mcts", "policy"),
           "opening_mode": ("sampled", "teacher", "book", "none")}
FLAGS = ("policy_noise_correction", "selfplay_dump")

# key -> (minimum, maximum, must be an integer). None means unbounded.
NUMBERS = {
    "channels": (1, None, True),
    "blocks": (1, None, True),
    "simulations": (1, None, True),
    "cpuct": (0.0, None, False),
    "search_bias_four": (0.0, None, False),
    "search_bias_neighbour": (0.0, None, False),
    "cheap_search_prob": (0.0, 1.0, False),
    "cheap_search_simulations": (1, None, True),
    "cheap_search_target_weight": (0.0, None, False),
    "root_noise_weight": (0.0, 1.0, False),
    "root_noise_total_concentration": (0.0, None, False),
    "policy_target_prune_prop": (0.0, 1.0, False),
    "policy_target_prune_min_count": (1, None, True),
    "temperature_moves": (0, None, True),
    "policy_weight": (0.0, None, False),
    "policy_soft_weight": (0.0, None, False),
    "policy_soft_temperature": (1.0, None, False),
    "value_weight_final": (0.0, None, False),
    "value_weight_mid": (0.0, None, False),
    "value_weight_short": (0.0, None, False),
    "value_horizon_short": (1, None, True),
    "value_horizon_mid": (1, None, True),
    "search_value_mix_final": (0.0, None, False),
    "search_value_mix_mid": (0.0, None, False),
    "search_value_mix_short": (0.0, None, False),
    "workers": (1, None, True),
    "games_per_round": (1, None, True),
    "train_steps": (1, None, True),
    "replay_capacity": (1, None, True),
    "batch_size": (1, None, True),
    "learning_rate": (0.0, None, False),
    "weight_decay": (0.0, None, False),
    "seed": (0, None, True),
    "eval_every": (1, None, True),
    "eval_pairs": (1, None, True),
    "min_replay_size": (1, None, True),
    "promotion_every": (1, None, True),
    "promotion_pairs": (1, None, True),
    "opening_plies": (0, None, True),
    "surprise_uniform_share": (0.0, 1.0, False),
    "surprise_cap": (1.0, None, False),
    "surprise_ref": (0.0, None, False),
    "surprise_policy_weight": (0.0, None, False),
    "surprise_value_weight": (0.0, None, False),
    "shard_size": (1, None, True),
}

DEFAULTS = dict(
    rule="freestyle", arch="hybrid-128-10", channels=128, blocks=10,
    # search
    simulations=200, cpuct=2.0, hard_rules="forced", search_bias="tactical",
    search_bias_four=2.0, search_bias_neighbour=0.5,
    cheap_search_prob=0.75, cheap_search_simulations=64, cheap_search_target_weight=0.0,
    root_noise_weight=0.25, root_noise_total_concentration=10.83,
    policy_target_prune_prop=0.02, policy_target_prune_min_count=2,
    temperature_moves=20, policy_noise_correction=True,
    # multi-scale objective
    policy_weight=1.0, policy_soft_weight=0.25, policy_soft_temperature=2.0,
    value_weight_final=1.0, value_weight_mid=0.5, value_weight_short=0.25,
    value_horizon_short=4, value_horizon_mid=10,
    search_value_mix_final=0.5, search_value_mix_mid=0.5, search_value_mix_short=0.0,
    # self-play loop
    search="mcts", opening_mode="sampled", opening_book=None,
    workers=16, games_per_round=32, train_steps=200,
    # replay and data
    replay_capacity=100000, batch_size=256, learning_rate=0.001, weight_decay=0.0001,
    seed=20260910, eval_every=10, eval_pairs=10, min_replay_size=1,
    promotion_every=10, promotion_pairs=10, opening_plies=8,
    surprise_uniform_share=0.5, surprise_cap=5.0, surprise_ref=1.0,
    surprise_policy_weight=1.0, surprise_value_weight=0.5,
    selfplay_dump=False, shard_size=4096,
)

# Fields a resume may legitimately change: they describe how data is gathered
# in this process, not what the stored model and optimizer mean.
RESUME_FREE = frozenset({"workers", "opening_plies", "selfplay_dump"})

# The pre-refactor candidate modes, mapped onto the decoupled pair that
# replaced them. This is the only place the old spelling is understood.
LEGACY_CANDIDATE_MODES = {"tactical": ("forced", "tactical"),
                          "forced": ("forced", "none"),
                          "legal": ("none", "none")}


def _number(key, value, minimum, maximum, integer):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid configuration: {key} must be a number")
    if integer and not isinstance(value, int):
        raise ValueError(f"Invalid configuration: {key} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid configuration: {key}={value!r} is outside "
                         f"[{minimum}, {maximum}]")


def translate_legacy(values):
    """Rewrite a stored configuration written before the candidate split."""
    values = dict(values)
    if "candidates" in values:
        mode = values.pop("candidates")
        try:
            hard, bias = LEGACY_CANDIDATE_MODES[mode]
        except KeyError:
            raise ValueError(f"Invalid configuration: candidates={mode!r} "
                             f"(known: {sorted(LEGACY_CANDIDATE_MODES)})") from None
        values.setdefault("hard_rules", hard)
        values.setdefault("search_bias", bias)
    return values


def mix_vector(cfg):
    """The leaf-value mix the search applies, normalised over the value heads."""
    values = np.array([float(cfg[f"search_value_mix_{head}"]) for head in VALUE_HEADS])
    if (values < 0).any() or values.sum() <= 0:
        raise ValueError("search_value_mix_* must be non-negative and not all zero")
    return values / values.sum()


def value_weights(cfg):
    return np.array([float(cfg[f"value_weight_{head}"]) for head in VALUE_HEADS])


def value_horizons(cfg):
    """Horizon in plies per head; the final head runs to the end of the game."""
    return {head: (None if head == "final" else int(cfg[f"value_horizon_{head}"]))
            for head in VALUE_HEADS}


def validate(cfg):
    """Check every key against the table above; returns the same dict."""
    for key in cfg:
        if key not in DEFAULTS:
            raise ValueError(f"Unknown configuration key: {key}")
    for key, value in cfg.items():
        if key in TEXT:
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Invalid configuration: {key} must be text or None")
            continue
        if key in CHOICES:
            if value not in CHOICES[key]:
                raise ValueError(f"Invalid configuration: {key}={value!r} "
                                 f"(known: {list(CHOICES[key])})")
            continue
        if key in FLAGS:
            if not isinstance(value, bool):
                raise ValueError(f"Invalid configuration: {key} must be a boolean")
            continue
        _number(key, value, *NUMBERS[key])
    if cfg["arch"] not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {cfg['arch']!r} "
                         f"(known: {sorted(ARCHITECTURES)})")
    width, pattern, _ = architecture(cfg["arch"])
    if (cfg["channels"], cfg["blocks"]) != (width, len(pattern)):
        raise ValueError(f"channels/blocks disagree with {cfg['arch']}: "
                         f"expected {width}/{len(pattern)}, "
                         f"got {cfg['channels']}/{cfg['blocks']}")
    if cfg["value_horizon_short"] >= cfg["value_horizon_mid"]:
        raise ValueError("value_horizon_short must be smaller than value_horizon_mid")
    mix_vector(cfg)
    return cfg


def from_file(path, rule):
    """The command line configuration: defaults, overrides, then validation."""
    cfg = dict(DEFAULTS)
    if path:
        values = translate_legacy(json.loads(Path(path).read_text(encoding="utf-8-sig")))
        unknown = set(values) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        cfg.update(values)
    cfg["rule"] = rule
    return validate(cfg)


def defaults(rule="freestyle", arch=None, **overrides):
    """A validated configuration for direct callers (pretraining, tests)."""
    cfg = dict(DEFAULTS)
    cfg["rule"] = rule
    if arch:
        cfg["arch"] = arch
        width, pattern, _ = architecture(arch)
        cfg["channels"], cfg["blocks"] = width, len(pattern)
    cfg.update(overrides)
    return validate(cfg)


def resume_incompatible(cfg, stored):
    """Configuration keys a resume may not change, given the stored config.

    A checkpoint written before a key existed is compared at that key current
    default rather than reported as a spurious mismatch.
    """
    baseline = {key: stored.get(key, DEFAULTS[key]) for key in DEFAULTS}
    changed = {key for key in set(cfg) | set(baseline) if cfg.get(key) != baseline.get(key)}
    return changed - RESUME_FREE


def upgrade_format1(stored):
    """A pre-refactor configuration brought onto the current key set."""
    stored = translate_legacy(stored or {})
    upgraded = {key: stored.get(key, DEFAULTS[key]) for key in DEFAULTS}
    upgraded.update({key: value for key, value in stored.items() if key in DEFAULTS})
    if "arch" not in stored:
        # A checkpoint written before arch existed still names its network
        # through channels/blocks; resolving it here keeps the legacy family
        # from being mistaken for the current default.
        try:
            upgraded["arch"] = architecture_of(stored)
        except ValueError:
            pass
    return upgraded


assert set(DEFAULTS) == set(TEXT) | set(CHOICES) | set(FLAGS) | set(NUMBERS)
