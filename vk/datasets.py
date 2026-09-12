"""Versioned shards over the shared position schema.

One reader and one writer serve both data sources: self-play positions and
Rapfi teacher positions are the same record with a different source tag and a
different set of valid targets. A shard written before this schema existed is
recognised by its field set and normalised on the way in, so old files stay
readable without any of the old field names leaking into the rest of the code.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

from .records import (ACTION_NONE, DTYPE, FIELDS, LEGACY_FIELDS, SCHEMA_VERSION,
                      VALUE_HEADS, blank, normalize_legacy, sentinel_topk, topk_row_valid,
                      topk_valid)

FORMAT_VERSION = SCHEMA_VERSION
LEGACY_VERSION = 3
LEGACY_ACTION = ACTION_NONE
FORMAT = "renju-position-npz"
SUPPORTED_VERSIONS = tuple(sorted(LEGACY_FIELDS))
SHARD_ROW_LIMIT = 4096

# Weight given to a position by the gap between its best and second-best
# analysed move, as (upper bound, weight) with the last entry open-ended.
# Positions where every move is equivalent get a quarter of the attention;
# positions where the choice actually costs winrate get four times it.
DEFAULT_GAP_WEIGHTS = ((0.01, 0.25), (0.03, 0.5), (0.05, 1.0), (0.10, 2.0),
                       (float("inf"), 4.0))


def symmetry(array, index):
    """One of the eight D4 transforms over the final two dimensions."""
    result = np.rot90(array, index % 4, axes=(-2, -1))
    if index >= 4:
        result = result[..., ::-1]
    return result.copy()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_fields(version):
    """The field set a shard of this version carries."""
    version = int(version)
    if version == FORMAT_VERSION:
        return FIELDS
    try:
        return LEGACY_FIELDS[version]
    except KeyError:
        raise ValueError(f"Unsupported shard version: {version!r} "
                         f"(known: {sorted(SUPPORTED_VERSIONS) + [FORMAT_VERSION]})") from None


def split_for_game(game_id):
    """Game-level train/validation/test split: 0-7 / 8 / 9 by game id."""
    bucket = int(game_id) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def topk_gap(winrates, actions=None):
    """W_1 - W_2 per row: the best analysed move edge over the next one.

    Rows without two real analysed moves get NaN, so they can be excluded from
    weighting rather than silently treated as a zero gap.
    """
    winrates = np.asarray(winrates, np.float64)
    gap = winrates[:, 0] - winrates[:, 1]
    valid = ~np.isnan(gap)
    if actions is not None:
        valid = valid & topk_row_valid(actions)
    return np.where(valid, gap, np.nan)


def gap_weights(winrates, buckets=DEFAULT_GAP_WEIGHTS, actions=None):
    """Per-row sampling weight derived from the top-1/top-2 gap.

    A position where every candidate move is equivalent teaches nothing about
    which move to pick, so most of the budget belongs on positions where the
    choice actually costs winrate. Rows with no usable gap keep the lowest
    weight instead of being dropped: their policy target is still valid.
    """
    gap = topk_gap(winrates, actions)
    weights = np.ones(len(gap), np.float64)
    for upper, weight in reversed(buckets):
        weights = np.where(np.nan_to_num(gap, nan=np.inf) < upper, weight, weights)
    return weights


def weighted_indices(weights, size, rng):
    """size indices drawn without replacement, proportional to weights."""
    weights = np.asarray(weights, np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Sampling weights must be finite, non-negative and not all zero")
    return rng.choice(len(weights), size=size, replace=False, p=weights / weights.sum())


def stack_arrays(arrays):
    """Build one record batch from per-field arrays (the inverse of savez)."""
    rows = len(arrays["state"])
    out = blank(rows)
    for name in FIELDS:
        out[name] = arrays[name]
    return out


class ShardWriter:
    """Buffered writer for one dataset, always in the current schema."""

    def __init__(self, root, shard_size=4096, split_of=None):
        self.root, self.shard_size = Path(root), int(shard_size)
        self.split_of = split_of or split_for_game
        self.buffers = {split: [] for split in ("train", "validation", "test")}
        self.counts = {split: 0 for split in self.buffers}
        self.shards = {split: 0 for split in self.buffers}
        for split in self.buffers:
            (self.root / split).mkdir(parents=True, exist_ok=True)

    def add(self, split, row):
        """Append one row given as a mapping of field name to value."""
        missing = set(FIELDS) - set(row.keys() if hasattr(row, "keys") else FIELDS)
        if missing:
            raise ValueError(f"Shard rows require fields {sorted(missing)}")
        self.buffers[split].append(row)
        if len(self.buffers[split]) >= self.shard_size:
            self.flush(split)

    def add_batch(self, records, split=None):
        """Append a whole batch, split by game id unless a split is given."""
        if not len(records):
            return
        if records.dtype != DTYPE:
            raise ValueError("Shards only accept records of the current schema")
        if split is not None:
            for row in records:
                self.buffers[split].append(row)
                if len(self.buffers[split]) >= self.shard_size:
                    self.flush(split)
            return
        splits = np.array([self.split_of(int(game_id)) for game_id in records["game_id"]])
        for name in ("train", "validation", "test"):
            for row in records[splits == name]:
                self.buffers[name].append(row)
                if len(self.buffers[name]) >= self.shard_size:
                    self.flush(name)

    def flush(self, split):
        records = self.buffers[split]
        if not records:
            return
        path = self.root / split / f"shard-{self.shards[split]:05d}.npz"
        np.savez_compressed(path, **{name: np.stack([row[name] for row in records])
                                     for name in FIELDS})
        self.counts[split] += len(records)
        self.shards[split] += 1
        records.clear()

    def close(self):
        for split in self.buffers:
            self.flush(split)


def read_shard(path):
    """One shard, normalised onto the current schema."""
    with np.load(path, allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    if len(np.asarray(arrays["state"])) > SHARD_ROW_LIMIT:
        raise ValueError(f"Shard exceeds {SHARD_ROW_LIMIT} rows: {path}")
    names = set(arrays)
    if names == set(FIELDS):
        return stack_arrays(arrays)
    version = next((item for item in SUPPORTED_VERSIONS
                    if names == set(LEGACY_FIELDS[item])), None)
    if version is None:
        raise ValueError(f"Invalid shard fields: {path}")
    return normalize_legacy(arrays, version)


def load_split(root, split):
    """One split as a structured array in the current schema."""
    paths = sorted((Path(root) / split).glob("shard-*.npz"))
    if not paths:
        raise ValueError(f"Dataset has no {split} shards")
    parts = [read_shard(path) for path in paths]
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def load_all(root):
    """Every split concatenated, for tools that only need the whole dataset."""
    parts = [load_split(root, split)
             for split in ("train", "validation", "test")
             if list((Path(root) / split).glob("shard-*.npz"))]
    if not parts:
        raise ValueError(f"Dataset has no shards: {root}")
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def combine_datasets(sources, output, seed=20260910):
    """Merge datasets into one, re-tagging every game so its id stays unique.

    A source keeps its internal game_id ordering; only the high digits change
    (source_index * 10**7), so every position of one game still maps to the same
    split and merging can never leak a position across splits.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Combined output is not empty: {output}")
    sources = [Path(source) for source in sources]
    if not sources:
        raise ValueError("combine_datasets needs at least one source")
    described, rules = [], set()
    for index, source in enumerate(sources):
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
            if manifest_path.exists() else {}
        if manifest.get("rule"):
            rules.add(manifest["rule"])
        described.append({"path": str(source.resolve()),
                          "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
                          "manifest_version": manifest.get("format_version"),
                          "game_id_offset": index * 10 ** 7})
    if len(rules) > 1:
        raise ValueError(f"Sources disagree about the rule: {sorted(rules)}")
    writer = ShardWriter(output)
    for source, info in zip(sources, described):
        offset, written = info["game_id_offset"], 0
        for split in ("train", "validation", "test"):
            for path in sorted((source / split).glob("shard-*.npz")):
                records = read_shard(path).copy()
                shifted = records["game_id"].astype(np.uint64) + np.uint64(offset)
                records["game_id"] = shifted.astype(np.uint32)
                # Recompute the split from the shifted id: the offset is a
                # multiple of ten, so a game can never move between splits.
                writer.add_batch(records)
                written += len(records)
        info["rows"] = written
    writer.close()
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "value_heads": list(VALUE_HEADS), "schema": SCHEMA_VERSION,
        "rule": rules.pop() if rules else None,
        "seed": seed, "positions_written": sum(writer.counts.values()),
        "split": "game_id modulo 10: 0-7 train / 8 validation / 9 test",
        "counts": writer.counts, "shards": writer.shards, "shard_size": writer.shard_size,
        "augmentation": "D4 at training time only",
        "sources": described,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
