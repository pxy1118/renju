"""Can the value labels tell sibling positions apart?

The measured problem: a value head trained as ``V(state) = Rapfi top-1 winrate``
saturates (mean |V| of 0.963 at MCTS leaves), so ``Q_A - Q_B`` carries no
information and PUCT collapses back onto the policy prior. Saturation is a
property of the *labels*, not of the network, and this tool measures it.

For every parent position with top-k winrates it emits one row per analysed
move: the child position together with the winrate of that same move, which is
what the network should predict for the child. Two modes:

``chained`` (default, no engine)
    ``target_i = 1 - W_i`` from the parent's own analysis. Free and consistent
    with how the teacher value is built (``2p - 1``); the cost is Rapfi's own
    noise across sibling branches, which the parent manifest records as
    ``top1_consistency``.

``fresh`` (``--engine``)
    Play the move and ask Rapfi about the child, so each target comes from that
    child's own search. More faithful, one engine query per child.

Either way the interesting output is the *spread*: how far apart the sibling
targets of one parent are. If that spread is tiny, no value network can learn
to discriminate the moves, and MCTS cannot get anything from Q.
"""
import argparse
from pathlib import Path
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vk.datasets import FORMAT_VERSION, ShardWriter, _split, load_split, topk_valid   # noqa: E402
from vk.game import Game                                        # noqa: E402


def sibling_spread(winrates):
    """Per-parent spread of analysed winrates; parents need at least two moves."""
    usable = ~np.isnan(winrates)
    counts = usable.sum(axis=1)
    rows = counts >= 2
    if not rows.any():
        return {}
    values = winrates[rows].astype(np.float64)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(np.where(usable[rows], values, np.nan), axis=1) - \
            np.nanmin(np.where(usable[rows], values, np.nan), axis=1)
    return {
        "parents": int(rows.sum()),
        "mean_sibling_spread": float(np.nanmean(spread)),
        "median_sibling_spread": float(np.nanmedian(spread)),
        "p90_sibling_spread": float(np.nanpercentile(spread, 90)),
        "share_spread_under_0.01": float(np.mean(spread < 0.01)),
        "share_spread_under_0.03": float(np.mean(spread < 0.03)),
        "note": ("a small spread means the labels themselves cannot separate the "
                 "sibling moves, which is what a Q term would need"),
    }


def saturation(values):
    values = np.asarray(values, np.float64)
    return {"rows": len(values), "abs_mean": float(np.abs(values).mean()),
            "std": float(values.std()),
            "abs_gt_0.9_share": float(np.mean(np.abs(values) > 0.9)),
            "abs_lt_0.5_share": float(np.mean(np.abs(values) < 0.5))}


def build_rows(data, limit, stride, engine=None, rule="freestyle", cache=None):
    """Yield one ``(child state, value)`` row per analysed move of each parent."""
    actions, winrates = data["teacher_topk_actions"], data["teacher_topk_winrates"]
    valid = topk_valid(actions)
    order = list(range(0, len(actions), max(1, stride)))
    if limit:
        order = order[:limit]
    for index in order:
        slots = np.flatnonzero(valid[index])
        if len(slots) < 2:
            continue
        state = data["state"][index]
        player = 1 if bool(state[2, 0, 0]) else -1
        board = np.where(state[0], player, np.where(state[1], -player, 0)).astype(np.int8)
        game = Game(rule, board, player)
        for slot in slots:
            action = int(actions[index][slot])
            winrate = float(winrates[index][slot])
            if not 0 <= action < 225 or not game.legal()[action]:
                continue
            child = game.copy()
            child.move(action, validate=False)
            if child.adjudicate() is not None:
                target = 1.0 if child.winner == child.player else -1.0
            elif engine is None:
                # Chained: the parent says this move is worth `winrate`, so the
                # child is worth that from the opponent's point of view.
                target = 1.0 - 2.0 * winrate
            else:
                key = (rule, child.board.tobytes(), child.player)
                cached = cache.get(key) if cache is not None else None
                if cached is None:
                    cached = float(engine.analyze(child, 1).moves[0].winrate)
                    if cache is not None:
                        cache[key] = cached
                target = 1.0 - 2.0 * cached
            yield {"state": child.encode().astype(np.uint8),
                   "value": np.float16(target),
                   "parent": np.uint32(index), "action": np.uint16(action),
                   "slot": np.uint8(slot),
                   "parent_winrate": np.float16(winrate),
                   "ply": np.uint16(np.count_nonzero(game.board) + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--output", required=True,
                        help="directory for the child-value dataset (must be empty)")
    parser.add_argument("--rule", default="freestyle", choices=["freestyle", "renju"])
    parser.add_argument("--limit", type=int, default=5000, help="parent positions to use (0 = all)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--engine")
    parser.add_argument("--engine-dir")
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--report", help="report path (default: <output>/diagnostic.json)")
    args = parser.parse_args()

    data = load_split(args.dataset, args.split)
    usable = topk_valid(data["teacher_topk_actions"])[:, :2].all(axis=1)
    winrates = data["teacher_topk_winrates"].astype(np.float64)
    diagnostic = {
        "dataset": str(Path(args.dataset).resolve()), "split": args.split,
        "parents_total": len(winrates), "parents_with_two_moves": int(usable.sum()),
        "parent_value": saturation(2.0 * winrates[:, 0] - 1.0),
        "sibling_spread": sibling_spread(winrates),
    }
    if not usable.any():
        # Format-2 data has no per-action winrate at all, so there is nothing to
        # measure; say so instead of producing an empty directory and a report
        # full of NaN.
        raise SystemExit(f"{args.dataset} has no top-k winrates (format-2 rows); "
                         f"regenerate it as format 3 before measuring value labels")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Child-value output is not empty: {output}")
    engine, cache = None, {}
    if args.engine:
        from vk.rapfi import RapfiClient
        engine = RapfiClient(args.engine, args.engine_dir, args.threads, args.hash_mb,
                             args.max_nodes, args.timeout, 2, rule=args.rule)
    started = time.monotonic()
    # A child-value row is not a teacher record: it carries the child state and
    # the target, plus enough provenance to trace a row back to its parent.
    fields = ("state", "value", "parent", "action", "slot", "parent_winrate", "ply")
    try:
        rows, writer, targets = 0, ShardWriter(output, args.shard_size, fields=fields), []
        for row in build_rows(data, args.limit, args.stride, engine=engine,
                              rule=args.rule, cache=cache):
            writer.add(_split(int(row["parent"])), row)
            targets.append(float(row["value"]))
            rows += 1
            if args.engine and rows % 2000 == 0:
                print(f"rows={rows} elapsed={time.monotonic() - started:.0f}s", flush=True)
        writer.close()
    finally:
        if engine is not None:
            engine.close()
    if not rows:
        raise RuntimeError("No child-value rows were produced; does the dataset have top-k data?")

    children = np.concatenate([np.load(path, allow_pickle=False)["value"]
                               for path in sorted(output.rglob("shard-*.npz"))])
    diagnostic.update({
        "mode": "fresh" if args.engine else "chained",
        "rows": rows, "counts": writer.counts, "shards": writer.shards,
        "child_value": saturation(children.astype(np.float64)),
        "elapsed_seconds": time.monotonic() - started,
        "note": ("rows are (child state, value) pairs; two rows of one parent are two "
                 "sibling positions whose targets must differ for Q to discriminate them"),
    })
    destination = Path(args.report) if args.report else output / "diagnostic.json"
    destination.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(json.dumps(diagnostic, indent=2), flush=True)
    print(f"written {destination}", flush=True)


if __name__ == "__main__":
    main()
