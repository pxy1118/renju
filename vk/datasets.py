"""Versioned teacher shard storage shared by Rapfi and DAgger generation.

Two formats exist:

``2``
    The original seven-field record set. ``policy`` is the softmax-normalised
    odds transform of the Rapfi MultiPV winrates, which is a *distribution over
    moves*, not a winrate -- see ``vk.teacher.teacher_policy`` for the formula.
    No per-action winrate was stored, so the true cost of a move cannot be
    recovered from the file alone.

``3``
    The same seven fields plus ``teacher_topk_actions`` and
    ``teacher_topk_winrates``: the raw Rapfi winrates of the analysed MultiPV
    moves, in analysis (descending winrate) order.

Reading is version aware and returns one dict of concatenated arrays for every
format. Rows carried over from a format-2 source are represented *at row level*
by sentinels -- ``teacher_topk_actions == LEGACY_ACTION`` and NaN winrates --
so a merged dataset needs no side channel to say which rows have real top-k
data. ``topk_valid`` is the single predicate for that question.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

FORMAT_VERSION = 3
LEGACY_VERSION = 2
SUPPORTED_VERSIONS = (LEGACY_VERSION, FORMAT_VERSION)

BASE_FIELDS = ("state", "policy", "value", "game_id", "ply", "teacher_best", "teacher_nodes")
TOPK_FIELDS = ("teacher_topk_actions", "teacher_topk_winrates")
SHARD_FIELDS = {LEGACY_VERSION: BASE_FIELDS, FORMAT_VERSION: BASE_FIELDS + TOPK_FIELDS}
TOPK_SLOTS = 5
# 255 can never be a board action (0-224), so it is an unambiguous "no teacher
# move in this slot" marker rather than a plausible coordinate.
LEGACY_ACTION = np.uint8(255)
FORMAT = "renju-rapfi-teacher-npz"


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
    try:
        return SHARD_FIELDS[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Unsupported teacher format version: {version!r} "
                         f"(known: {sorted(SHARD_FIELDS)})") from None


def topk_valid(actions):
    """Boolean mask of rows whose ``teacher_topk_actions`` are real moves.

    Works on a single row or a stacked array; ``LEGACY_ACTION`` slots (format-2
    rows merged into a format-3 shard) are False.
    """
    return np.asarray(actions) != LEGACY_ACTION


# Weight given to a position by the gap between its best and second-best
# analysed move, as (upper bound, weight) with the last entry open-ended.
# Positions where every move is equivalent get a quarter of the attention;
# positions where the choice actually costs winrate get four times it.
DEFAULT_GAP_WEIGHTS = ((0.01, 0.25), (0.03, 0.5), (0.05, 1.0), (0.10, 2.0),
                       (float("inf"), 4.0))


def topk_gap(winrates, actions=None):
    """``W_1 - W_2`` per row: the best analysed move's edge over the next one.

    Rows without two real analysed moves get NaN, so they can be excluded from
    weighting rather than silently treated as a zero gap.
    """
    winrates = np.asarray(winrates, np.float64)
    gap = winrates[:, 0] - winrates[:, 1]
    valid = ~np.isnan(gap)
    if actions is not None:
        valid = valid & topk_valid(actions)[:, 0] & topk_valid(actions)[:, 1]
    return np.where(valid, gap, np.nan)


def gap_weights(winrates, buckets=DEFAULT_GAP_WEIGHTS, actions=None):
    """Per-row sampling weight derived from the top-1/top-2 gap.

    A position where every candidate move is equivalent teaches nothing about
    *which* move to pick, so most of the budget belongs on positions where the
    choice actually costs winrate. Rows with no usable gap keep the lowest
    weight instead of being dropped: their policy target is still valid.
    """
    gap = topk_gap(winrates, actions)
    weights = np.ones(len(gap), np.float64)
    for upper, weight in reversed(buckets):
        weights = np.where(np.nan_to_num(gap, nan=np.inf) < upper, weight, weights)
    return weights


def weighted_indices(weights, size, rng):
    """``size`` indices drawn without replacement, proportional to ``weights``."""
    weights = np.asarray(weights, np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Sampling weights must be finite, non-negative and not all zero")
    return rng.choice(len(weights), size=size, replace=False, p=weights / weights.sum())


def sentinel_topk(rows):
    """Format-3 top-k arrays for ``rows`` rows that carry no teacher winrates."""
    return (np.full((rows, TOPK_SLOTS), LEGACY_ACTION, np.uint8),
            np.full((rows, TOPK_SLOTS), np.nan, np.float16))


def _split(game_id):
    bucket = int(game_id) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


class ShardWriter:
    def __init__(self, root, shard_size=4096, version=FORMAT_VERSION, fields=None):
        self.root, self.shard_size = Path(root), int(shard_size)
        self.version = int(version)
        # ``fields`` lets a specialised dataset (child-value pairs) reuse the
        # sharding without pretending to be a teacher set.
        self.fields = tuple(fields) if fields else shard_fields(self.version)
        self.buffers = {split: [] for split in ("train", "validation", "test")}
        self.counts = {split: 0 for split in self.buffers}
        self.shards = {split: 0 for split in self.buffers}
        for split in self.buffers:
            (self.root / split).mkdir(parents=True, exist_ok=True)

    def add(self, split, record):
        self.buffers[split].append(record)
        if len(self.buffers[split]) >= self.shard_size:
            self.flush(split)

    def flush(self, split):
        records = self.buffers[split]
        if not records:
            return
        missing = [name for name in self.fields if name not in records[0]]
        if missing:
            raise ValueError(f"Writer version {self.version} requires fields {missing}")
        arrays = {name: np.stack([record[name] for record in records]) for name in self.fields}
        path = self.root / split / f"shard-{self.shards[split]:05d}.npz"
        np.savez_compressed(path, **arrays)
        self.counts[split] += len(records)
        self.shards[split] += 1
        records.clear()

    def close(self):
        for split in self.buffers:
            self.flush(split)


def _pad_topk(arrays, rows):
    """Fill absent top-k fields with sentinels so every row is explicit."""
    if "teacher_topk_actions" not in arrays or "teacher_topk_winrates" not in arrays:
        arrays["teacher_topk_actions"], arrays["teacher_topk_winrates"] = sentinel_topk(rows)
    return arrays


def load_split(root, split, fields=None):
    """Load one split as concatenated arrays.

    ``fields`` pins an exact field set (legacy callers); otherwise the shard's
    own field set decides which version it is, and format-3 output is always
    returned with the top-k fields present.
    """
    paths = sorted((Path(root) / split).glob("shard-*.npz"))
    if not paths:
        raise ValueError(f"Teacher dataset has no {split} shards")
    collected, version = {key: [] for key in BASE_FIELDS}, None
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            names = tuple(shard.files)
            if fields is not None:
                if set(names) != set(fields):
                    raise ValueError(f"Invalid teacher shard fields: {path}")
                current = dict(shard)
            else:
                match = next((item for item in SUPPORTED_VERSIONS
                              if set(names) == set(shard_fields(item))), None)
                if match is None:
                    raise ValueError(f"Invalid teacher shard fields: {path}")
                current = dict(shard)
                version = match if version is None else version
            if len(current["state"]) > 4096:
                raise ValueError(f"Teacher shard exceeds 4096 rows: {path}")
            for key in current:
                collected.setdefault(key, []).append(current[key])
    out = {key: np.concatenate(parts) for key, parts in collected.items() if parts}
    if fields is None:
        if "teacher_topk_actions" not in out or "teacher_topk_winrates" not in out:
            out["teacher_topk_actions"], out["teacher_topk_winrates"] = sentinel_topk(len(out["state"]))
        return out
    return {key: out[key] for key in fields}


def load_all(root, fields=None):
    """Every split concatenated, for tools that only need the whole dataset."""
    parts = [load_split(root, split, fields)
             for split in ("train", "validation", "test")
             if list((Path(root) / split).glob("shard-*.npz"))]
    if not parts:
        raise ValueError(f"Teacher dataset has no shards: {root}")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def combine_datasets(sources, output, seed=20260910):
    """Merge datasets into one, re-tagging every game so its id stays unique.

    A source keeps its internal ``game_id`` ordering; only the high digits
    change (``source_index * 10**7``), so every position of one game still maps
    to the same split and merging can never leak a position across splits.
    Format-2 rows are carried across with sentinel top-k fields.
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
        version = int(manifest.get("format_version", FORMAT_VERSION))
        shard_fields(version)
        if manifest.get("rule"):
            rules.add(manifest["rule"])
        described.append({"path": str(source.resolve()),
                          "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
                          "format_version": version, "game_id_offset": index * 10 ** 7})
    if len(rules) > 1:
        raise ValueError(f"Sources disagree about the rule: {sorted(rules)}")
    writer = ShardWriter(output)
    for index, (source, source_info) in enumerate(zip(sources, described)):
        version, offset, written = source_info["format_version"], source_info["game_id_offset"], 0
        for split in ("train", "validation", "test"):
            for path in sorted((source / split).glob("shard-*.npz")):
                with np.load(path, allow_pickle=False) as shard:
                    names = tuple(shard.files)
                    if set(names) != set(shard_fields(version)):
                        raise ValueError(f"Invalid teacher shard fields: {path}")
                    rows = len(shard["state"])
                    game_id = shard["game_id"].astype(np.uint64) + np.uint64(offset)
                    for row in range(rows):
                        record = {key: shard[key][row] for key in BASE_FIELDS}
                        record["game_id"] = np.uint32(game_id[row])
                        if version >= FORMAT_VERSION:
                            for key in TOPK_FIELDS:
                                record[key] = shard[key][row]
                        else:
                            actions, winrates = sentinel_topk(1)
                            record["teacher_topk_actions"] = actions[0]
                            record["teacher_topk_winrates"] = winrates[0]
                        writer.add(_split(int(game_id[row])), record)
                        written += 1
        source_info["rows"] = written
    writer.close()
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "rule": rules.pop() if rules else None,
        "seed": seed, "positions_requested": sum(writer.counts.values()),
        "positions_written": sum(writer.counts.values()),
        "split": "game_id modulo 10: 0-7 train / 8 validation / 9 test",
        "counts": writer.counts, "shards": writer.shards, "shard_size": writer.shard_size,
        "augmentation": "D4 at training time only",
        "sources": described,
        "topk": TOPK_SLOTS,
        "topk_semantics": ("teacher_topk_actions/teacher_topk_winrates are raw Rapfi "
                           "MultiPV winrates from the side-to-move perspective; slots "
                           "filled with 255 / NaN mark rows inherited from a format-2 "
                           "source that stored no per-action winrate"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
