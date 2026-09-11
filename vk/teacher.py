"""Rapfi teacher-game generation and safe, sharded NPZ storage."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import math
import threading
import time
import numpy as np

from .candidates import tactical_candidates
from .game import Game
from .rapfi import RapfiClient, RapfiError

FORMAT_VERSION = 2


def symmetry(array, index):
    """One of the eight D4 transforms over the final two dimensions."""
    result = np.rot90(array, index % 4, axes=(-2, -1))
    if index >= 4:
        result = result[..., ::-1]
    return result.copy()


def canonical_key(game):
    board = game.board.reshape(15, 15)
    return min(symmetry(board, index).tobytes() for index in range(8)) + bytes((game.player + 1,))


def teacher_policy(game, moves):
    if not moves:
        raise ValueError("Teacher returned no moves")
    probabilities = np.array([min(1 - 1e-6, max(1e-6, move.winrate)) for move in moves], np.float64)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    target = np.zeros(225, np.float64)
    for weight, move in zip(weights, moves):
        target[move.action] += 0.98 * weight
    candidates = tactical_candidates(game).mask
    target[candidates] += 0.02 / candidates.sum()
    target /= target.sum()
    return target.astype(np.float16)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShardWriter:
    def __init__(self, root, shard_size=4096):
        self.root, self.shard_size = Path(root), int(shard_size)
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
        names = ("state", "policy", "value", "game_id", "ply", "teacher_best", "teacher_nodes")
        arrays = {name: np.stack([record[name] for record in records]) for name in names}
        path = self.root / split / f"shard-{self.shards[split]:05d}.npz"
        np.savez_compressed(path, **arrays)
        self.counts[split] += len(records)
        self.shards[split] += 1
        records.clear()

    def close(self):
        for split in self.buffers:
            self.flush(split)


def _split(game_id):
    bucket = int(game_id) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def _play_teacher_game(client, game_id, seed, sample_plies):
    rng, game, records = np.random.default_rng(seed), Game("freestyle"), []
    # Rapfi's empty-board shortcut returns only the forced center and no value.
    # Do not manufacture a value label; begin recording after that forced move.
    game.move(112)
    while game.adjudicate() is None:
        analysis = client.analyze(game, 5)
        target = teacher_policy(game, analysis.moves)
        best = analysis.moves[0]
        records.append({
            "state": game.encode().astype(np.uint8), "policy": target,
            "value": np.float16(2 * best.winrate - 1), "game_id": np.uint32(game_id),
            "ply": np.uint16(np.count_nonzero(game.board)), "teacher_best": np.uint16(best.action),
            "teacher_nodes": np.uint64(best.nodes),
        })
        if np.count_nonzero(game.board) < sample_plies:
            probabilities = np.zeros(225, np.float64)
            probabilities[[move.action for move in analysis.moves]] = target[
                [move.action for move in analysis.moves]].astype(np.float64)
            probabilities /= probabilities.sum()
            action = int(rng.choice(225, p=probabilities))
        else:
            action = best.action
        game.move(action)
    return records


def generate_teacher_dataset(engine, engine_dir, output, positions=50_000, workers=4,
                             threads=4, hash_mb=256, max_nodes=200_000, timeout=5.0,
                             retries=2, seed=20260910, shard_size=4096,
                             sample_plies=12):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Teacher output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    clients, lock, local = [], threading.Lock(), threading.local()

    def client():
        if not hasattr(local, "client"):
            local.client = RapfiClient(engine, engine_dir, threads, hash_mb, max_nodes, timeout, retries)
            with lock:
                clients.append(local.client)
        return local.client

    def play(gid):
        return _play_teacher_game(client(), gid, seed + gid * 1_000_003, sample_plies)

    writer, seen, game_id, failures, stalled_waves = ShardWriter(output, shard_size), set(), 0, 0, 0
    audit_pairs = 0
    audit_sum_same = 0.0
    audit_sum_negated = 0.0
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while sum(writer.counts.values()) + sum(map(len, writer.buffers.values())) < positions:
                before = sum(writer.counts.values()) + sum(map(len, writer.buffers.values()))
                ids = list(range(game_id, game_id + workers))
                futures = [pool.submit(play, gid) for gid in ids]
                game_id += workers
                for gid, future in zip(ids, futures):
                    try:
                        records = future.result()
                    except (RapfiError, TimeoutError, OSError) as exc:
                        failures += 1
                        print(f"discarded teacher game {gid}: {exc}", flush=True)
                        continue
                    for left, right in zip(records, records[1:]):
                        left_value, right_value = float(left["value"]), float(right["value"])
                        audit_pairs += 1
                        audit_sum_same += abs(left_value - right_value)
                        audit_sum_negated += abs(left_value + right_value)
                    if sum(writer.counts.values()) + sum(map(len, writer.buffers.values())) >= positions:
                        continue
                    split = _split(gid)
                    for record in records:
                        player = 1 if bool(record["state"][2, 0, 0]) else -1
                        board = np.where(record["state"][0], player,
                                         np.where(record["state"][1], -player, 0))
                        key = canonical_key(Game("freestyle", board, player))
                        if key not in seen:
                            seen.add(key)
                            writer.add(split, record)
                            current = sum(writer.counts.values()) + sum(map(len, writer.buffers.values()))
                            if current >= positions:
                                break
                current = sum(writer.counts.values()) + sum(map(len, writer.buffers.values()))
                stalled_waves = stalled_waves + 1 if current == before else 0
                if stalled_waves >= 3:
                    raise RapfiError("Teacher generation made no progress for three waves")
                print(f"teacher positions={current}/{positions} games={game_id} failures={failures}", flush=True)
    finally:
        writer.close()
        for item in clients:
            item.close()

    engine_path, engine_dir = Path(engine).resolve(), Path(engine_dir).resolve()
    perspective_audit = {
        "adjacent_pairs": audit_pairs,
        "mean_abs_v_plus_next": audit_sum_negated / audit_pairs if audit_pairs else None,
        "mean_abs_v_minus_next": audit_sum_same / audit_pairs if audit_pairs else None,
        "passed": positions < 100 or audit_pairs < 10 or audit_sum_negated < audit_sum_same,
    }
    if not perspective_audit["passed"]:
        raise RapfiError("Teacher values failed the side-to-move perspective audit")
    source_files = [path for path in engine_dir.rglob("*")
                    if path.is_file() and (path.name == "config.toml" or path.suffix.lower() in {".bin", ".nnue", ".lz4"})]
    manifest = {
        "format": "renju-rapfi-teacher-npz", "format_version": FORMAT_VERSION,
        "rule": "freestyle", "seed": seed, "positions_requested": positions,
        "positions_written": sum(writer.counts.values()), "games_attempted": game_id,
        "failed_games": failures, "split": "game_id modulo 10: 0-7/8/9",
        "counts": writer.counts, "shards": writer.shards, "shard_size": shard_size,
        "augmentation": "D4 at training time only",
        "rapfi_version": next((item.version for item in clients if item.version != "unknown"), "unknown"),
        "engine": {"path": str(engine_path), "sha256": sha256(engine_path)},
        "engine_files": [{"path": str(path.relative_to(engine_dir)), "sha256": sha256(path)}
                         for path in sorted(source_files)],
        "generation": {"workers": workers, "threads_per_engine": threads, "hash_mb": hash_mb,
                       "max_nodes": max_nodes, "timeout_seconds": timeout, "retries": retries,
                       "multipv": 5, "sample_plies": sample_plies,
                       "yxboard_roles": "1=current player, 2=opponent",
                       "policy_mix": {"teacher": 0.98, "local_candidates": 0.02}},
        "value_perspective_audit": perspective_audit,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
