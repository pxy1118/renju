"""Rapfi teacher-game generation and safe, sharded NPZ storage.

A teacher position is the same record self-play produces, with source marked as
teacher and the engine analysis kept alongside it. The value targets are built
once the game is over: the final head reads the result of the teacher game, and
the horizon heads read the chained Rapfi winrates, which is exactly the same
construction self-play uses, only with an engine search value instead of the
network own.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import threading
import time
import numpy as np

from .candidates import hard_candidates
from .config import DEFAULTS, value_horizons
from .datasets import (FORMAT, FORMAT_VERSION, SCHEMA_VERSION, ShardWriter, sha256,
                       split_for_game, symmetry)
from .game import Game
from .rapfi import RapfiClient, RapfiError
from .records import ACTION_NONE, SOURCE_TEACHER, VALUE_HEADS, blank, set_head, stack
from .targets import value_targets

__all__ = ["FORMAT_VERSION", "ShardWriter", "canonical_key", "generate_teacher_dataset",
           "sha256", "symmetry", "teacher_policy", "teacher_winrates", "topk_slots"]

TOPK_SLOTS = 5


def canonical_key(game):
    board = game.board.reshape(15, 15)
    return min(symmetry(board, index).tobytes() for index in range(8)) + bytes((game.player + 1,))


def teacher_winrates(moves):
    """Raw Rapfi winrate of every analysed move, in the order Rapfi ranked them.

    These are probabilities from the side to move perspective, exactly as the
    engine reported them. They are what a *cost* measurement must use: the
    stored policy vector is the softmax of their odds transform, which is a
    distribution over moves and only looks like a winrate by coincidence.
    """
    if not moves:
        raise ValueError("Teacher returned no moves")
    return np.array([min(1 - 1e-6, max(1e-6, move.winrate)) for move in moves], np.float64)


def teacher_policy(game, moves):
    """Distribution matching target: odds-softmax of the winrates, 98/2 mixed.

    Kept bit-compatible with the data generated before the schema change, so a
    change in this formula would silently make the two datasets incomparable.
    The 2% mix now spreads over the hard candidate set, which is every legal
    point unless a five or a block is forced.
    """
    probabilities = teacher_winrates(moves)
    logits = np.log(probabilities) - np.log1p(-probabilities)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    target = np.zeros(225, np.float64)
    for weight, move in zip(weights, moves):
        target[move.action] += 0.98 * weight
    candidates = hard_candidates(game, "forced").mask
    target[candidates] += 0.02 / candidates.sum()
    target /= target.sum()
    return target.astype(np.float16)


def topk_slots(moves):
    """(actions, winrates) arrays of TOPK_SLOTS entries, sentinel padded."""
    actions = np.full(TOPK_SLOTS, ACTION_NONE, np.uint8)
    winrates = np.full(TOPK_SLOTS, np.nan, np.float16)
    for index, move in enumerate(moves[:TOPK_SLOTS]):
        actions[index] = move.action
        winrates[index] = move.winrate
    return actions, winrates


def _play_teacher_game(client, game_id, seed, sample_plies, rule="freestyle"):
    rng, game, moves = np.random.default_rng(seed), Game(rule), []
    # Open the game without asking the engine: on a near-empty board it answers
    # a shortcut that carries no value records. Freestyle has no forced first
    # point, so one is drawn; Renju forces the centre and then has exactly one
    # legal reply, so drawing from the legal set covers both rules.
    game.move(int(rng.choice(np.flatnonzero(game.legal()))))
    reply = np.flatnonzero(game.legal())
    if len(reply) == 1:
        game.move(int(reply[0]))
    while game.adjudicate() is None:
        analysis = client.analyze(game, 5)
        target = teacher_policy(game, analysis.moves)
        best = analysis.moves[0]
        topk_actions, topk_winrates = topk_slots(analysis.moves)
        winrates = [move.winrate for move in analysis.moves]
        moves.append({
            "state": game.encode().astype(np.uint8), "policy": target,
            "search_value": np.float16(2 * best.winrate - 1),
            "q_spread": np.float16(max(winrates) - min(winrates)) if len(winrates) > 1 else np.nan,
            "ply": np.uint16(np.count_nonzero(game.board)),
            "teacher_best": np.uint16(best.action), "teacher_nodes": np.uint64(best.nodes),
            "teacher_topk_actions": topk_actions, "teacher_topk_winrates": topk_winrates,
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
    return _teacher_records(moves, game, game_id)


def _teacher_records(moves, game, game_id):
    """The finished teacher game as one record batch with multi-scale targets."""
    horizons = value_horizons(DEFAULTS)
    search_values = np.array([float(move["search_value"]) for move in moves], np.float64)
    values, valid, _ = value_targets(search_values, game.winner,
                                     horizons["short"], horizons["mid"])
    records = blank(len(moves))
    for index, move in enumerate(moves):
        records["state"][index] = move["state"]
        records["policy"][index] = move["policy"]
        records["policy_valid"][index] = 1
        records["policy_weight"][index] = 1.0
        records["search_value"][index] = move["search_value"]
        records["q_spread"][index] = move["q_spread"]
        records["simulations"][index] = move["teacher_nodes"]
        records["full_search"][index] = 1
        records["game_id"][index] = np.uint32(game_id)
        records["ply"][index] = move["ply"]
        records["winner"][index] = game.winner
        records["source"][index] = SOURCE_TEACHER
        records["teacher_best"][index] = move["teacher_best"]
        records["teacher_nodes"][index] = move["teacher_nodes"]
        records["teacher_topk_actions"][index] = move["teacher_topk_actions"]
        records["teacher_topk_winrates"][index] = move["teacher_topk_winrates"]
    records["value"] = values
    records["value_valid"] = valid
    return records


def generate_teacher_dataset(engine, engine_dir, output, positions=50_000, workers=4,
                             threads=4, hash_mb=256, max_nodes=200_000, timeout=5.0,
                             retries=2, seed=20260910, shard_size=4096,
                             sample_plies=12, rule="freestyle"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Teacher output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    clients, lock, local = [], threading.Lock(), threading.local()

    def client():
        if not hasattr(local, "client"):
            local.client = RapfiClient(engine, engine_dir, threads, hash_mb, max_nodes,
                                       timeout, retries, rule)
            with lock:
                clients.append(local.client)
        return local.client

    def play(gid):
        return _play_teacher_game(client(), gid, seed + gid * 1_000_003, sample_plies, rule)

    writer, seen, game_id, failures, stalled_waves = ShardWriter(output, shard_size), set(), 0, 0, 0
    audit_pairs = 0
    audit_sum_same = 0.0
    audit_sum_negated = 0.0
    audit_top1_pairs = 0
    audit_top1_error = 0.0
    written = 0
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while written < positions:
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
                        left_value, right_value = float(left["search_value"]), float(right["search_value"])
                        audit_pairs += 1
                        audit_sum_same += abs(left_value - right_value)
                        audit_sum_negated += abs(left_value + right_value)
                        # Only rows where the teacher actually followed its own top-1
                        # one ply deeper can be checked against 1 - W(parent top-1):
                        # inside the sampling window the played move may be another
                        # analysed move, and then the child describes a different line.
                        if int(right["teacher_topk_actions"][0]) != int(left["teacher_best"]):
                            continue
                        audit_top1_pairs += 1
                        audit_top1_error += abs(1.0 - float(right["teacher_topk_winrates"][0]))
                    split = split_for_game(gid)
                    for row in records:
                        player = 1 if bool(row["state"][2, 0, 0]) else -1
                        board = np.where(row["state"][0], player,
                                         np.where(row["state"][1], -player, 0))
                        key = canonical_key(Game(rule, board, player))
                        if key in seen:
                            continue
                        seen.add(key)
                        writer.add(split, row)
                        written += 1
                        if written >= positions:
                            break
                print(f"teacher positions={written}/{positions} games={game_id} "
                      f"failures={failures}", flush=True)
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
        "format": FORMAT, "format_version": FORMAT_VERSION, "schema": SCHEMA_VERSION,
        "value_heads": list(VALUE_HEADS),
        "source": "teacher",
        "rule": rule, "seed": seed, "positions_requested": positions,
        "positions_written": written, "games_attempted": game_id,
        "failed_games": failures, "split": "game_id modulo 10: 0-7 train / 8 validation / 9 test",
        "counts": writer.counts, "shards": writer.shards, "shard_size": shard_size,
        "augmentation": "D4 at training time only",
        "topk": TOPK_SLOTS,
        "policy_semantics": ("policy[i] is the softmax of the odds transform of the Rapfi "
                             "winrates, normalised over the analysed moves and 98/2 mixed "
                             "with the hard candidate set. It is a distribution over moves, "
                             "NOT a winrate; never read it as P(win) for move i."),
        "value_semantics": ("value[:, final] is the result of the teacher game from the mover "
                            "perspective; value[:, mid/short] truncate the game at "
                            f"{DEFAULTS['value_horizon_mid']}/{DEFAULTS['value_horizon_short']} "
                            "plies and bootstrap the chained Rapfi winrate when the game has "
                            "not ended by then."),
        "winrate_semantics": (f"teacher_topk_actions/teacher_topk_winrates hold the raw Rapfi "
                              "MultiPV winrates from the side-to-move perspective, in analysis "
                              f"order. Slots are {int(ACTION_NONE)}/NaN when fewer than "
                              f"{TOPK_SLOTS} moves were analysed."),
        "q_spread_semantics": "the MultiPV winrate spread of the analysed moves",
        "rapfi_version": next((item.version for item in clients if item.version != "unknown"), "unknown"),
        "engine": {"path": str(engine_path), "sha256": sha256(engine_path)},
        "engine_files": [{"path": str(path.relative_to(engine_dir)), "sha256": sha256(path)}
                         for path in sorted(source_files)],
        "generation": {"workers": workers, "threads_per_engine": threads, "hash_mb": hash_mb,
                       "max_nodes": max_nodes, "timeout_seconds": timeout, "retries": retries,
                       "multipv": 5, "sample_plies": sample_plies, "rule": rule,
                       "yxboard_roles": "1=current player, 2=opponent",
                       "policy_mix": {"teacher": 0.98, "hard_candidates": 0.02}},
        "value_perspective_audit": perspective_audit,
        "top1_consistency": {"pairs": audit_top1_pairs,
                             "mean_abs_one_minus_next_top1": audit_top1_error / audit_top1_pairs
                             if audit_top1_pairs else None,
                             "note": "Rapfi is not a fixed-depth solver: 1 - W(parent top1) need "
                                     "not equal W(child top1). A large value means the winrates "
                                     "are noisy across sibling positions, which weakens any "
                                     "single-position cost argument."},
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
