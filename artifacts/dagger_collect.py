"""Collect DAgger-style samples: the model plays, Rapfi labels every position.

The teacher dataset records "what Rapfi meets when Rapfi plays well". This
records what the model actually meets, and labels each of those positions with
a deeper Rapfi analysis, which is the only way to get labels on the positions
the model's own mistakes create.

Output is a standard shard set in the current schema, so it merges with the
historical teacher data through vk.datasets.combine_datasets and feeds the
normal pretraining path. The per-decision ``hard``/``regret`` flags stay out of the
training schema and go to a sidecar CSV instead: they are diagnostics for
deciding *whether* to do hard-example oversampling, not part of the data
contract.

Cost note: every position costs one MultiPV annotation, and each model move
outside the annotation's top-k costs one more query on the child. Run it with a
small ``--positions`` first and read the regret distribution before deciding to
spend real Rapfi time on a large set.
"""
import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vk.datasets import (FORMAT_VERSION, SCHEMA_VERSION, ShardWriter,  # noqa: E402
                         read_shard, sha256, split_for_game)
from vk.records import (SOURCE_TEACHER, VALUE_HEADS, WINNER_UNKNOWN,  # noqa: E402
                        blank, head_targets, set_head)
from vk.evaluation import bootstrap_interval                     # noqa: E402
from vk.network import Evaluator, Network, architecture_of       # noqa: E402
from vk.openings import balanced_opening                         # noqa: E402
from vk.rapfi import RapfiClient                                 # noqa: E402
from vk.search import MCTS                                       # noqa: E402
from vk.teacher import teacher_policy, topk_slots                # noqa: E402


def _dagger_row(game, game_id, moves, best):
    """One annotated position in the current schema.

    The annotation is the point: policy and the final-head value come from the
    deeper Rapfi analysis, and the top-k winrates stay alongside it. The result
    of this game is not known while it is still being annotated, so the final
    target is the engine estimate and the record says the winner is unknown.
    """
    actions, winrates = topk_slots(moves)
    record = blank(1)
    record["state"][0] = game.encode().astype(np.uint8)
    record["policy"][0] = teacher_policy(game, moves)
    record["policy_valid"][0] = 1
    record["policy_weight"][0] = 1.0
    record["search_value"][0] = np.float16(2 * best.winrate - 1)
    record["q_spread"][0] = np.float16(max(move.winrate for move in moves)
                                       - min(move.winrate for move in moves))
    record["weight"][0] = 1.0
    record["simulations"][0] = np.uint32(best.nodes)
    record["full_search"][0] = 1
    record["game_id"][0] = np.uint32(game_id)
    record["ply"][0] = np.uint16(np.count_nonzero(game.board))
    record["winner"][0] = WINNER_UNKNOWN
    record["source"][0] = SOURCE_TEACHER
    record["teacher_best"][0] = np.uint16(best.action)
    record["teacher_nodes"][0] = np.uint64(best.nodes)
    record["teacher_topk_actions"][0] = actions
    record["teacher_topk_winrates"][0] = winrates
    set_head(record, "final", np.array([2 * best.winrate - 1], np.float32))
    return record[0]


def model_move(game, evaluator, cfg, tree):
    if cfg["search"] == "policy":
        from vk.search import policy_move
        return policy_move(game, evaluator, cfg["hard_rules"], cfg["search_bias"])[0]
    return int(np.argmax(tree.policy(game)))


def collect(args):
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Network(architecture_of(state["config"])).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    evaluator = Evaluator(model, args.device)
    cfg = {"search": args.search, "hard_rules": args.hard_rules,
           "search_bias": args.search_bias,
           "simulations": args.simulations, "cpuct": args.cpuct}
    del state

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"DAgger output is not empty: {output}")
    writer = ShardWriter(output, args.shard_size)
    diagnostics, regrets, positions, games = [], [], 0, 0
    started = time.monotonic()
    with RapfiClient(args.engine, args.engine_dir, args.threads, args.hash_mb,
                     args.max_nodes, args.timeout, 2, rule=args.rule) as client:
        for game_id in range(args.games):
            if positions >= args.positions:
                break
            game = balanced_opening(args.rule, args.seed + game_id, args.opening_plies)
            tree = MCTS(evaluator, cfg["simulations"], cfg["cpuct"],
                        hard_rules=cfg["hard_rules"], bias=cfg["search_bias"]) \
                if cfg["search"] == "mcts" else None
            annotated = 0
            while game.adjudicate() is None and positions < args.positions:
                analysis = client.analyze(game, args.annotation_multipv)
                moves = analysis.moves
                best = moves[0]
                action = model_move(game, evaluator, cfg, tree)
                in_topk = any(move.action == action for move in moves)
                regret = None
                if not in_topk:
                    child = game.copy()
                    child.move(action, validate=False)
                    if child.adjudicate() is not None:
                        p_model = 1.0 if child.winner == game.player else 0.0
                    else:
                        p_model = 1.0 - float(client.analyze(child, 1).moves[0].winrate)
                    regret = float(best.winrate) - p_model
                    regrets.append(regret)
                writer.add(split_for_game(game_id), _dagger_row(game, game_id, moves, best))
                diagnostics.append({"sample_id": positions, "game_id": game_id,
                                    "ply": int(np.count_nonzero(game.board)),
                                    "model_action": action, "teacher_best": best.action,
                                    "in_topk": in_topk,
                                    "regret": "" if regret is None else f"{regret:.6f}",
                                    "hard": int(regret is not None and regret >= args.regret_threshold)})
                positions += 1
                annotated += 1
                game.move(action, validate=False)
                if tree is not None:
                    tree.advance(action, game)
            games += 1
            print(f"dagger games={games} positions={positions}/{args.positions} "
                  f"model_ply={annotated} elapsed={time.monotonic() - started:.0f}s", flush=True)
            if positions >= args.positions:
                break
    writer.close()
    if not positions:
        raise RuntimeError("DAgger collection produced no positions")

    values = np.concatenate([head_targets(read_shard(path), "final")
                             for path in sorted(output.rglob("shard-*.npz"))]).astype(np.float32)
    hard = sum(row["hard"] for row in diagnostics)
    manifest = {
        "format": "renju-position-npz", "format_version": FORMAT_VERSION,
        "schema": SCHEMA_VERSION, "value_heads": list(VALUE_HEADS), "source": "teacher",
        "rule": args.rule, "seed": args.seed, "player": "model",
        "positions_requested": args.positions, "positions_written": positions,
        "games": games, "counts": writer.counts, "shards": writer.shards,
        "shard_size": args.shard_size,
        "split": "game_id modulo 10: 0-7 train / 8 validation / 9 test",
        "augmentation": "D4 at training time only",
        "topk": args.annotation_multipv,
        "collection": {"checkpoint": str(Path(args.checkpoint).resolve()),
                       "search": args.search, "hard_rules": args.hard_rules,
                       "search_bias": args.search_bias,
                       "simulations": args.simulations if args.search == "mcts" else 0,
                       "cpuct": args.cpuct, "opening_plies": args.opening_plies,
                       "annotation_multipv": args.annotation_multipv,
                       "annotation_max_nodes": args.max_nodes},
        "engine": {"path": str(Path(args.engine).resolve()),
                   "sha256": sha256(args.engine) if Path(args.engine).is_file() else None,
                   "version": client.version},
        "diagnostics": {
            "outside_topk": int(sum(not row["in_topk"] for row in diagnostics)),
            "outside_topk_share": float(np.mean([not row["in_topk"] for row in diagnostics])),
            "hard_threshold": args.regret_threshold, "hard": hard,
            "hard_share": hard / positions,
            "regret_mean": float(np.mean(regrets)) if regrets else None,
            "regret_mean_ci95": bootstrap_interval(regrets) if regrets else None,
            "sidecar": "diagnostics.csv (not part of the training schema)",
        },
        "value_abs_mean": float(np.abs(values).mean()),
        "value_saturation": {"abs_lt_0.5_share": float(np.mean(np.abs(values) < 0.5)),
                             "abs_gt_0.9_share": float(np.mean(np.abs(values) > 0.9))},
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    csv = output / "diagnostics.csv"
    columns = ("sample_id", "game_id", "ply", "model_action", "teacher_best",
               "in_topk", "regret", "hard")
    with csv.open("w", encoding="utf-8") as destination:
        destination.write(",".join(columns) + "\n")
        for row in diagnostics:
            destination.write(",".join(str(row[column]) for column in columns) + "\n")
    print(json.dumps(manifest["diagnostics"], indent=2), flush=True)
    print(f"manifest {output / 'manifest.json'}\ncsv      {csv}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rule", default="freestyle", choices=["freestyle", "renju"])
    parser.add_argument("--positions", type=int, default=2000)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--search", default="mcts", choices=["mcts", "policy"])
    parser.add_argument("--hard-rules", default="forced", choices=["forced", "none"])
    parser.add_argument("--search-bias", default="tactical", choices=["none", "tactical"])
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--cpuct", type=float, default=2.0)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--annotation-multipv", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--regret-threshold", type=float, default=0.05)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
