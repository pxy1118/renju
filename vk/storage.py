"""Checkpoint and log persistence.

Everything that reads or writes a run directory lives here. The training loop
owns the optimisation; this module owns the bytes. It is also the single place
that understands a checkpoint written before the multi-head schema existed.
"""
import json
import os
from pathlib import Path
import random
import torch

from .config import upgrade_format1 as upgrade_config
from .records import SCHEMA_VERSION, VALUE_HEADS

CHECKPOINT_FORMAT = 2
LEGACY_CHECKPOINT_FORMAT = 1


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


def load_checkpoint(path, rule, formats=(CHECKPOINT_FORMAT,)):
    # Local trusted checkpoints only: optimizer/RNG/replay require pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") not in formats:
        raise ValueError(f"Checkpoint format {state.get('format')!r} is not usable for "
                         f"this operation (accepted: {list(formats)}); pre-refactor "
                         f"checkpoints can be read for inference or used as an "
                         f"--init-checkpoint, but not resumed")
    if state["config"]["rule"] != rule:
        raise ValueError("Checkpoint format or rule mismatch")
    return state


def upgrade_format1(state, heads=None):
    """A pre-refactor checkpoint, made loadable by the current network.

    The old single value head becomes the final head verbatim; the horizon
    heads start as copies of it, so the network initially predicts the same
    value at every horizon instead of starting from noise. A pre-refactor
    replay buffer is a list of tuples with no schema and is dropped.
    """
    if state.get("format") != LEGACY_CHECKPOINT_FORMAT:
        return state
    heads = tuple(heads) if heads else VALUE_HEADS
    model = dict(state["model"])
    for head in heads[1:]:
        for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
            source = f"value.{suffix}"
            if source in model:
                model[f"value_{head}.{suffix}"] = model[source]
    upgraded = dict(state)
    upgraded.update({"format": CHECKPOINT_FORMAT, "schema": SCHEMA_VERSION,
                     "model": model, "config": upgrade_config(state.get("config"))})
    upgraded.pop("replay", None)
    return upgraded


def load_model_state(path, rule):
    """Weights for inference, upgrading a pre-refactor checkpoint if needed."""
    from .network import value_heads
    state = load_checkpoint(path, rule,
                            formats=(CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT))
    state = upgrade_format1(state, value_heads(state["config"].get("arch")))
    return state


def save(root, cfg, model, optimizer, replay_state, rng, round_id, step, total_games,
         pending_steps=0, champion_model=None, champion_optimizer=None, champion_step=0,
         teacher=None):
    state = {"format": CHECKPOINT_FORMAT, "schema": SCHEMA_VERSION, "config": cfg,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "replay": replay_state, "round": round_id, "step": step,
             "total_games": total_games, "pending_steps": pending_steps,
             "value_heads": list(VALUE_HEADS),
             "rng": rng.bit_generator.state, "python_rng": random.getstate(),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
             "champion_model": champion_model, "champion_optimizer": champion_optimizer,
             "champion_step": champion_step, "teacher": teacher}
    path = Path(root) / f"checkpoint-{round_id:08d}-{step:010d}.pt"
    atomic_save(state, path)
    for old in sorted(Path(root).glob("checkpoint-*.pt"))[:-3]:
        old.unlink()
    return path


def save_best(root, cfg, model_state, step, teacher=None):
    """The promoted champion, inference only: no optimizer, no replay."""
    path = Path(root) / "best.pt"
    atomic_save({"format": CHECKPOINT_FORMAT, "schema": SCHEMA_VERSION, "config": cfg,
                 "model": model_state, "step": step, "teacher": teacher,
                 "value_heads": list(VALUE_HEADS)}, path)
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
