"""Run the decoupled evaluation arms against Rapfi and compare them pairwise.

Each arm differs from another by exactly one switch, so a paired bootstrap
interval between two arms answers one question and nothing else:

    A1 vs A2   do the hard tactical rules (complete a five / block a five) help?
    A2 vs A3   does adding the soft tactical bias to the prior help?
    A3 vs A4   does MCTS + value help at all?   <- the only isolation of search
    A4 vs A5   does a larger search budget help?

A1 is the raw policy prior, A3 the policy prior plus bias, A4 the full search:
that chain isolates one switch per step. Plain policy against plain MCTS at the
same settings would be the same test as A3 vs A4.

Run a small pass first (--pairs 25) to see the direction, then widen.
"""
import argparse
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vk.evaluation import evaluate_rapfi, paired_delta              # noqa: E402
from vk.network import Network, architecture_of                     # noqa: E402
from vk.config import DEFAULTS                                      # noqa: E402
from vk.storage import load_model_state                             # noqa: E402

ARMS = {
    "A1-policy-nohard": {"search": "policy", "hard_rules": "none", "search_bias": "none"},
    "A2-policy-hard": {"search": "policy", "hard_rules": "forced", "search_bias": "none"},
    "A3-policy-bias": {"search": "policy", "hard_rules": "forced", "search_bias": "tactical"},
    "A4-mcts": {"search": "mcts", "hard_rules": "forced", "search_bias": "tactical"},
}
A5 = {"A5-mcts-wide": {"search": "mcts", "hard_rules": "forced", "search_bias": "tactical",
                       "simulations": 1600}}
QUESTIONS = (("A1-policy-nohard", "A2-policy-hard", "hard tactical rules"),
             ("A2-policy-hard", "A3-policy-bias", "soft tactical bias"),
             ("A3-policy-bias", "A4-mcts", "MCTS + value"),
             ("A4-mcts", "A5-mcts-wide", "more simulations"))


ARMS = {
    "A1-policy-forced": {"search": "policy", "candidates": "forced"},
    "A2-policy-legal": {"search": "policy", "candidates": "legal"},
    "A3-mcts-tactical": {"search": "mcts", "candidates": "tactical"},
    "A4-mcts-legal": {"search": "mcts", "candidates": "legal"},
}
A5 = {"A5-mcts-legal-1600": {"search": "mcts", "candidates": "legal", "simulations": 1600}}
QUESTIONS = (("A1-policy-forced", "A2-policy-legal", "forced tactical rule"),
             ("A2-policy-legal", "A4-mcts-legal", "MCTS + value"),
             ("A3-mcts-tactical", "A4-mcts-legal", "tactical candidate pruning"),
             ("A4-mcts-legal", "A5-mcts-legal-1600", "more simulations"))


def summary_line(name, report):
    statistics = report.get("search_statistics", {})
    return {"arm": name, "score": report["score"], "wins": report["wins"],
            "losses": report["losses"], "draws": report["draws"],
            "wilson_ci95": report["score_wilson_ci95"],
            "by_color": report["by_color"], "games": report["games"],
            "complete": report["complete"],
            "search_mode": report["search_mode"],
            "hard_rules": report["hard_rules"], "search_bias": report["search_bias"],
            "simulations": report["simulations"],
            "search_statistics": statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/freestyle-pretrain/best.pt")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--output", default="artifacts/compare-arms")
    parser.add_argument("--pairs", type=int, default=25)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS) + list(A5))
    parser.add_argument("--with-a5", action="store_true",
                        help="also run the 1600-simulation arm (only worth it if A4 beat A3)")
    parser.add_argument("--minutes", type=float, default=240)
    parser.add_argument("--max-nodes", type=int, default=200_000)
    parser.add_argument("--engine-threads", type=int, default=4)
    parser.add_argument("--engine-hash-mb", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opening-seed", type=int, default=91823)
    args = parser.parse_args()

    import torch
    state = load_model_state(args.checkpoint, "freestyle")
    cfg = dict(DEFAULTS, rule="freestyle", simulations=400)
    cfg.update({key: value for key, value in state["config"].items() if key in DEFAULTS})
    model = Network(architecture_of(state["config"])).to(args.device)
    model.load_state_dict(state["model"])
    del state

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = {name: options for name, options in {**ARMS, **A5}.items()}
    reports, started = {}, time.monotonic()
    for name in args.arms + (["A5-mcts-wide"] if args.with_a5 and
                             "A5-mcts-wide" not in args.arms else []):
        arm = dict(cfg, **catalog[name])
        print(f"=== {name}: search={arm['search']} hard_rules={arm['hard_rules']} "
              f"bias={arm['search_bias']} simulations={arm['simulations']} "
              f"pairs={args.pairs} ===", flush=True)
        report = evaluate_rapfi(model, arm, args.device, args.engine, args.engine_dir,
                                args.pairs, time.monotonic() + args.minutes * 60,
                                lambda: False, args.engine_threads, args.engine_hash_mb,
                                args.max_nodes, 5.0, args.opening_seed, with_records=True)
        reports[name] = report
        (output / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(summary_line(name, report), indent=2), flush=True)

    comparisons = []
    for left, right, question in QUESTIONS:
        if left not in reports or right not in reports:
            continue
        delta = paired_delta(reports[left]["records"], reports[right]["records"])
        comparisons.append({"question": question, "a": left, "b": right,
                            "meaning": f"positive means {left} scored higher than {right}",
                            **(delta or {})})
    summary = {"pairs": args.pairs, "games_per_arm": args.pairs * 2,
               "checkpoint": str(Path(args.checkpoint).resolve()),
               "engine": args.engine, "max_nodes": args.max_nodes,
               "opening_seed": args.opening_seed, "elapsed_seconds": time.monotonic() - started,
               "arms": {name: summary_line(name, report) for name, report in reports.items()},
               "comparisons": comparisons}
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)
    print(f"written {output / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
