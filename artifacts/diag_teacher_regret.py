"""Measure what a model move actually costs, in raw winrate points.

Why this tool exists: ``policy[i]`` in a teacher shard is the softmax of the
odds transform of Rapfi's MultiPV winrates (``vk.teacher.teacher_policy``). It
is a distribution over moves and only resembles a winrate by accident, so any
"gap" read off it -- the historical "23% real blunder rate" -- is not a winrate
loss. The only honest cost measurement re-asks the engine.

Scale, fixed once and used everywhere below:

    p_*      = Rapfi's winrate for its own best move, side to move, raw [0, 1]
    p_child  = Rapfi's winrate for the child position after the model's move,
               from the *opponent's* point of view
    p_model  = 1 - p_child                      (the model's move, parent's view)
    regret   = p_* - p_model = p_* + p_child - 1

so ``regret = 0.05`` means the model's move cost five winrate points. When the
model's move is itself one of the analysed top-5 there is nothing to re-ask:
``p_model`` is already known from the same root search, which is both cheaper
and noise-free, and it is the majority case.

Two checkpoints can be scored in one sweep (``--compare-checkpoint``). Every
engine query is then shared, and the report is a *paired* delta over identical
positions -- the same discipline as ``vk.evaluation.paired_delta``, applied to
move cost rather than game results.

Caveat, recorded in the report: ``p_i`` values come from one root search, so
comparing moves across branches assumes the engine's evaluation is comparable
between them. Rapfi is not a fixed-depth solver, so that assumption is
approximate; the ``top1_consistency`` audit in a teacher manifest measures it.
"""
import argparse
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vk.candidates import candidate_mask, forced_candidates  # noqa: E402
from vk.datasets import load_split, topk_valid               # noqa: E402
from vk.evaluation import bootstrap_interval                 # noqa: E402
from vk.game import Game                                     # noqa: E402
from vk.network import Evaluator, Network, architecture_of   # noqa: E402
from vk.rapfi import RapfiClient                             # noqa: E402

BUCKETS = ((0.01, "equivalent"), (0.02, "minor"), (0.05, "moderate"),
           (0.10, "real_error"), (float("inf"), "severe"))


def bucket_of(regret):
    for limit, name in BUCKETS:
        if regret < limit:
            return name
    return "severe"


def restore(state):
    """Rebuild the position a stored state encodes.

    ``Game.encode`` packs ``[our stones, their stones, we-are-black]``, so the
    colour must come from plane 2. Assuming black would be harmless in
    freestyle and silently wrong in Renju, where legality depends on who is
    moving.
    """
    player = 1 if bool(state[2, 0, 0]) else -1
    board = np.where(state[0], player, np.where(state[1], -player, 0))
    return board.astype(np.int8)


def engine_digest(engine):
    """Fingerprint of the engine *and* its executable, for cache invalidation.

    A fake or rebuilt engine is part of the measurement, so answers it gave
    under a different implementation must not be reused.
    """
    digest = hashlib.sha256()
    path = Path(engine)
    if path.exists():
        digest.update(path.read_bytes())
    neighbour = path.parent / "config.toml"
    if neighbour.exists():
        digest.update(neighbour.read_bytes())
    return digest.hexdigest()


class AnalysisCache:
    """Engine answers keyed by position, engine binary and node budget."""

    def __init__(self, path, key):
        self.path, self.key = Path(path), key
        self.data = {}
        if self.path.exists():
            stored = json.loads(self.path.read_text(encoding="utf-8"))
            if stored.get("key") == key:
                self.data = stored.get("entries", {})
            else:
                print(f"cache key changed; starting fresh ({self.path})", flush=True)

    @staticmethod
    def position_key(rule, board, player):
        digest = hashlib.sha256()
        digest.update(rule.encode())
        digest.update(np.asarray(board, np.int8).tobytes())
        digest.update(bytes((player + 1,)))
        return digest.hexdigest()[:32]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value, flush=False):
        self.data[key] = value
        if flush:
            self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"key": self.key, "entries": self.data}),
                             encoding="utf-8")
        temporary.replace(self.path)


def engine_moves(client, game, multipv, cache, rule, commit=False):
    """``{action: winrate}`` for the analysed MultiPV records, cached."""
    key = cache.position_key(rule, game.board, game.player)
    cached = cache.get(key)
    if cached is None:
        cached = {move.action: float(move.winrate)
                  for move in client.analyze(game, multipv).moves}
        cache.put(key, cached, flush=commit)
    return cached


def model_move(game, evaluator, candidates):
    """The model's action under one candidate mode, with no search involved."""
    restricted = forced_candidates(game) if candidates in ("forced", "tactical") \
        else candidate_mask(game, candidates)
    logits, _ = evaluator(game.encode())
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (225,) or not np.isfinite(logits).all():
        raise RuntimeError("Invalid network policy output")
    allowed = restricted.mask.copy()
    if not allowed.any():
        raise ValueError("Cannot choose a move: no legal placement")
    return int(np.argmax(np.where(allowed, logits, -np.inf)))


def statistics(rows):
    """Aggregate one list of measured rows; returns ``{}`` when nothing was scored."""
    rows = [row for row in rows if row is not None]
    if not rows:
        return {}
    regret = np.array([row["regret"] for row in rows], np.float64)
    buckets = {}
    for limit, name in BUCKETS:
        buckets[name] = int((regret < limit).sum()) if limit != float("inf") else int(len(regret))
    return {
        "rows": len(rows),
        "policy_top1_hit": float(np.mean([row["policy_top1"] for row in rows])),
        "policy_top5_hit": float(np.mean([row["policy_top5"] for row in rows])),
        "outside_teacher_top5": float(np.mean([row["outside_top5"] for row in rows])),
        "mask_excludes_teacher_best": float(np.mean([row["mask_excludes_best"] for row in rows])),
        "regret_mean": float(regret.mean()),
        "regret_median": float(np.median(regret)),
        "regret_p90": float(np.percentile(regret, 90)),
        "regret_mean_ci95": bootstrap_interval(regret),
        "severe_share": float(np.mean(regret >= 0.10)),
        "bucket_counts": buckets,
        "bucket_share": {name: buckets[name] / len(rows) for name in buckets},
    }


def paired_statistics(rows_a, rows_b):
    """Paired comparison of arm A against arm B on identical positions."""
    pairs = [(a, b) for a, b in zip(rows_a, rows_b) if a is not None and b is not None]
    if not pairs:
        return None
    deltas = np.array([a["regret"] - b["regret"] for a, b in pairs], np.float64)
    better = int((deltas < -1e-9).sum())
    worse = int((deltas > 1e-9).sum())
    return {
        "rows": len(pairs),
        "mean_regret_delta": float(deltas.mean()),
        "mean_regret_delta_ci95": bootstrap_interval(deltas),
        "meaning": ("positive means the second checkpoint is cheaper (A - B); "
                    "negative means the second checkpoint lost more winrate"),
        "a_cheaper": better, "b_cheaper": worse, "ties": len(pairs) - better - worse,
        "severe_a": int(np.mean([a["regret"] >= 0.10 for a, _ in pairs]) * len(pairs)),
        "severe_b": int(np.mean([b["regret"] >= 0.10 for _, b in pairs]) * len(pairs)),
        "critical_mean_delta": float(np.mean(
            [a["regret"] - b["regret"] for a, b in pairs
             if a["decision_critical"] or b["decision_critical"]])) if any(
            a["decision_critical"] or b["decision_critical"] for a, b in pairs) else None,
    }


def sibling_diagnostic(dataset, split, limit):
    """How far apart are the value targets of two sibling positions?

    Without a spread here no value head can discriminate the moves, which is
    what a PUCT Q term needs. Purely offline: it reads the stored winrates.
    ``artifacts/value_dataset_diagnostic.py`` builds the child-value training
    rows from the same idea and reports this as part of its output.
    """
    data = load_split(dataset, split)
    winrates = data["teacher_topk_winrates"].astype(np.float64)
    valid = topk_valid(data["teacher_topk_actions"])
    usable = valid & ~np.isnan(winrates)
    if limit:
        usable = usable[:limit]
        winrates = winrates[:limit]
    counts = usable.sum(axis=1)
    rows = counts >= 2
    if not rows.any():
        return None
    values = np.where(usable[rows], winrates[rows], np.nan)
    spread = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    return {"parents": int(rows.sum()),
            "mean_sibling_spread": float(np.nanmean(spread)),
            "median_sibling_spread": float(np.nanmedian(spread)),
            "p90_sibling_spread": float(np.nanpercentile(spread, 90)),
            "share_spread_under_0.01": float(np.mean(spread < 0.01)),
            "note": ("sibling spread is the value signal MCTS needs; a small spread "
                     "means the labels cannot separate the moves")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare-checkpoint",
                        help="score a second checkpoint on the same rows and report the "
                             "paired regret difference (A = --checkpoint, B = this one)")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--rule", default="freestyle", choices=["freestyle", "renju"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--stride", type=int, default=1,
                        help="take every Nth record before applying --limit")
    parser.add_argument("--candidates", default="forced",
                        choices=["tactical", "forced", "legal"],
                        help="forced/tactical both use the deterministic rule set; "
                             "legal lets every point compete")
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--parent-multipv", type=int, default=5,
                        help="parent MultiPV; 5 gives the top1-top2 gap decision_critical needs")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", help="report path (default: artifacts/regret-<stamp>.json)")
    parser.add_argument("--cache", help="engine answer cache (default: next to the report)")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--traceback", action="store_true",
                        help="print a full traceback for every skipped row")
    args = parser.parse_args()

    data = load_split(args.dataset, args.split)
    loaded, described = {}, {}
    for label, path in (("a", args.checkpoint), ("b", args.compare_checkpoint)):
        if path is None:
            continue
        state = torch.load(path, map_location=args.device, weights_only=False)
        model = Network(architecture_of(state["config"])).to(args.device)
        model.load_state_dict(state["model"])
        model.eval()
        loaded[label] = Evaluator(model, args.device)
        described[label] = {"path": str(Path(path).resolve()), "run": Path(path).parent.name,
                            "step": state.get("step"),
                            "config": {key: state["config"].get(key) for key in
                                       ("arch", "channels", "blocks", "rule")}}
        del state

    actions_all = data["teacher_topk_actions"]
    winrates_all = data["teacher_topk_winrates"]
    valid_all = topk_valid(actions_all)
    if not valid_all.any():
        print("dataset has no stored top-k winrates; querying the engine for every parent",
              flush=True)
    order = list(range(0, len(actions_all), max(1, args.stride)))
    if args.limit:
        order = order[:args.limit]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = Path(args.output) if args.output else \
        Path(__file__).resolve().parents[1] / "artifacts" / f"regret-{args.split}-{stamp}.json"
    cache = AnalysisCache(args.cache or output.with_suffix(".cache.json"),
                          json.dumps({"engine": str(Path(args.engine).resolve()),
                                      "engine_digest": engine_digest(args.engine),
                                      "max_nodes": args.max_nodes,
                                      "multipv": args.parent_multipv}))
    rows = {label: [] for label in loaded}
    failures, started = 0, time.monotonic()
    disagreements = present = 0
    with RapfiClient(args.engine, args.engine_dir, args.threads, args.hash_mb,
                     args.max_nodes, args.timeout, 2, rule=args.rule) as client:
        for done, index in enumerate(order, 1):
            try:
                game = Game(args.rule, restore(data["state"][index]),
                            1 if bool(data["state"][index][2, 0, 0]) else -1)
                ranked = sorted(engine_moves(client, game, args.parent_multipv, cache,
                                             args.rule).items(), key=lambda item: -item[1])
                p_best = float(ranked[0][1])
                engine_best = int(ranked[0][0])
                recorded_best = int(data["teacher_best"][index]) if "teacher_best" in data else None
                if recorded_best is not None and recorded_best != engine_best:
                    disagreements += 1
                    present += int(any(action == recorded_best for action, _ in ranked))
                forced = forced_candidates(game).mask
                for label, evaluator in loaded.items():
                    allowed = game.legal() if args.candidates == "legal" else forced
                    logits, _ = evaluator(game.encode())
                    choice = int(np.argmax(np.where(allowed, logits, -np.inf)))
                    in_top5 = any(action == choice for action, _ in ranked)
                    if in_top5:
                        p_model = float(dict(ranked)[choice])
                    else:
                        child = game.copy()
                        child.move(choice, validate=False)
                        if child.adjudicate() is not None:
                            p_model = 1.0 if child.winner == game.player else 0.0
                        else:
                            best = min(engine_moves(client, child, 1, cache, args.rule).items(),
                                       key=lambda item: -item[1])
                            p_model = 1.0 - float(best[1])
                    regret = p_best - p_model
                    rows[label].append({
                        "index": int(index), "ply": int(data["ply"][index]),
                        "game_id": int(data["game_id"][index]),
                        "model_action": choice, "engine_best": engine_best,
                        "recorded_best": recorded_best,
                        "recorded_best_in_ranking": bool(
                            recorded_best is not None
                            and any(action == recorded_best for action, _ in ranked)),
                        "p_best": p_best, "p_model": p_model, "regret": regret,
                        "bucket": bucket_of(regret), "in_top5": in_top5,
                        "outside_top5": not in_top5,
                        "policy_top1": choice == engine_best, "policy_top5": in_top5,
                        "mask_excludes_best": not bool(allowed[engine_best]),
                        "decision_critical": len(ranked) > 1 and (p_best - ranked[1][1]) > 0.05,
                        "decided": p_best >= 0.95})
            except Exception as exc:                     # engine or dataset anomaly
                failures += 1
                print(f"skipped row {index}: {type(exc).__name__}: {exc}", flush=True)
                if args.traceback:
                    import traceback as _traceback
                    _traceback.print_exc()
                for label in loaded:
                    rows[label].append(None)
                continue
            if args.progress_every and done % args.progress_every == 0:
                print(f"analysed {done}/{len(order)} failures={failures} "
                      f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    cache.flush()

    def subsets(label):
        scored = [row for row in rows[label] if row is not None and not row["decided"]]
        return {"all": statistics(rows[label]),
                "decided_excluded": statistics(scored) if scored and len(scored) != len(rows[label]) else None,
                "decision_critical": statistics([row for row in scored if row["decision_critical"]]) or None,
                "by_ply": {f"{low}-{high - 1}": statistics(
                    [row for row in scored if low <= row["ply"] < high])
                    for low, high in ((0, 12), (12, 24), (24, 40), (40, 226))}}

    excluded = [row for row in rows["a"] if row is not None]
    report = {        "dataset": str(Path(args.dataset).resolve()), "split": args.split,
        "rows_requested": len(order), "rows_skipped": failures, "stride": args.stride,
        "candidates": args.candidates,
        "scale": ("regret is in raw winrate probability points: "
                  "regret = p_best - p_model = p_best + p_child - 1"),
        "caveat": ("p_i come from one Rapfi root search; comparing moves across "
                   "branches assumes engine evaluations are comparable between them, "
                   "which is approximate for a non-solver engine"),
        "checkpoints": described,
        "reference": {
            "optimum": "fresh engine top-1 of the same analysis the model is scored against",
            "recorded_best_disagreements": disagreements,
            "recorded_best_in_fresh_ranking": present,
            "note": ("the shard's teacher_best comes from another engine instance, so a "
                     "disagreement is engine noise between searches, not a dataset bug"),
        },
        "engine": {"path": str(Path(args.engine).resolve()), "max_nodes": args.max_nodes,
                   "multipv": args.parent_multipv, "threads": args.threads},
        "a": subsets("a"),
        "elapsed_seconds": time.monotonic() - started,
    }
    if "b" in loaded:
        report["b"] = subsets("b")
        report["paired"] = paired_statistics(rows["a"], rows["b"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    columns = ("index", "game_id", "ply", "engine_best", "recorded_best", "p_best")
    for label in loaded:
        columns += (f"model_action_{label}", f"p_model_{label}", f"regret_{label}",
                    f"bucket_{label}", f"in_top5_{label}", f"mask_excludes_best_{label}",
                    f"decision_critical_{label}", f"decided_{label}")
    csv = output.with_suffix(".csv")
    with csv.open("w", encoding="utf-8") as destination:
        destination.write(",".join(columns) + "\n")
        for position in range(len(rows["a"])):
            first = rows["a"][position]
            if first is None:
                continue
            line = [first["index"], first["game_id"], first["ply"], first["engine_best"],
                    first["recorded_best"], f"{first['p_best']:.4f}"]
            for label in loaded:
                row = rows[label][position]
                line += [row["model_action"], f"{row['p_model']:.4f}", f"{row['regret']:.4f}",
                         row["bucket"], int(row["in_top5"]), int(row["mask_excludes_best"]),
                         int(row["decision_critical"]), int(row["decided"])]
            destination.write(",".join(str(value) for value in line) + "\n")

    for label in loaded:
        for name in ("all", "decided_excluded", "decision_critical"):
            summary = report[label][name]
            if summary:
                print(f"{label} {name:18s} n={summary['rows']:5d} mean={summary['regret_mean']:.4f} "
                      f"median={summary['regret_median']:.4f} p90={summary['regret_p90']:.4f} "
                      f"severe={summary['severe_share']:.3f} "
                      f"outside_top5={summary['outside_teacher_top5']:.3f} "
                      f"mask_excludes_best={summary['mask_excludes_teacher_best']:.4f}", flush=True)
    if "paired" in report:
        paired = report["paired"]
        print(f"paired A-B mean={paired['mean_regret_delta']:+.4f} "
              f"CI={[round(value, 4) for value in paired['mean_regret_delta_ci95']]} "
              f"A_cheaper={paired['a_cheaper']} B_cheaper={paired['b_cheaper']} "
              f"ties={paired['ties']}", flush=True)
    print(f"report {output}\ncsv    {csv}", flush=True)


if __name__ == "__main__":
    main()
