"""The evaluation switches must actually change how the model picks moves.

Every arm of the comparison matrix is (search, candidates, simulations), so
these tests pin that each switch reaches the move-selection code and that the
report says which arm produced it.
"""
from pathlib import Path
import json
import textwrap
import numpy as np
import torch

from vk.evaluation import evaluate_rapfi, match, summary, paired_delta
from vk.network import Network
from vk.config import DEFAULTS


def fake_engine(path, winrate=0.5):
    """Legal-move engine: never repeats a point, one complete MultiPV group."""
    path.write_text(textwrap.dedent(f'''\
        import sys
        occupied = set()
        for raw in sys.stdin:
            command = raw.strip()
            if command.startswith("START"):
                print("OK", flush=True)
            elif command == "YXSHOWINFO":
                print("MESSAGE Rapfi eval-fake", flush=True)
            elif command.startswith("YXBOARD"):
                occupied = {{int(item.split(",")[1])*15 + int(item.split(",")[0])
                            for item in command.split()[1:-1]}}
            elif command.startswith("YXNBEST"):
                nbest = int(command.split()[1])
                legal = [action for action in range(225) if action not in occupied][:max(1, nbest)]
                for index, action in enumerate(legal):
                    x, y = action % 15, action // 15
                    print(f"INFO PV {{index}}", flush=True)
                    print(f"INFO NUMPV {{len(legal)}}", flush=True)
                    print("INFO DEPTH 3", flush=True)
                    print("INFO NODES 100", flush=True)
                    print(f"INFO WINRATE {winrate}", flush=True)
                    print(f"INFO BESTLINE {{x}},{{y}}", flush=True)
                    print("INFO PV DONE", flush=True)
                action = legal[0]
                print(f"{{action%15}},{{action//15}}", flush=True)
            elif command == "END":
                break
        '''), encoding="utf-8")
    return path


def cfg(**overrides):
    base = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
                simulations=2, workers=1, opening_plies=4)
    base.update(overrides)
    return base


def test_evaluate_rapfi_reports_the_arm_it_ran(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    model = Network("hybrid-8-1")
    report = evaluate_rapfi(model, cfg(search="policy", hard_rules="none",
                                       search_bias="none"), "cpu",
                            engine, tmp_path, pairs=1, threads=1, hash_mb=8,
                            max_nodes=10, timeout=2)
    assert report["games"] == 2 and report["complete"]
    assert set(report["by_color"]) == {"1", "-1"}
    assert report["search_mode"] == "policy" and report["hard_rules"] == "none"
    assert report["search_bias"] == "none"
    assert report["simulations"] == 0 and report["max_nodes"] == 10
    assert report["opening_mode"] == "sampled" and report["opening_seed"] == 91823


def test_policy_search_records_moves_without_search_statistics(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    model = Network("hybrid-8-1")
    report = evaluate_rapfi(model, cfg(search="policy", hard_rules="forced"), "cpu",
                            engine, tmp_path, pairs=1, threads=1, hash_mb=8,
                            max_nodes=10, timeout=2)
    statistics = report["search_statistics"]
    assert statistics["hard_mode_hist"] == {"policy_only": statistics["moves"]}
    # A policy arm has no MCTS measurements to publish, and must not pretend
    # they were taken and came out empty.
    assert "search_prior_kl_mean" not in statistics
    assert "root_visit_entropy_mean" not in statistics


def test_mcts_search_records_root_visit_shape(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    model = Network("hybrid-8-1")
    report = evaluate_rapfi(model, cfg(search="mcts", hard_rules="forced",
                                       search_bias="tactical", simulations=3),
                            "cpu", engine, tmp_path, pairs=1, threads=1, hash_mb=8,
                            max_nodes=10, timeout=2)
    assert report["search_mode"] == "mcts" and report["simulations"] == 3
    statistics = report["search_statistics"]
    assert set(statistics["hard_mode_hist"]) <= {"all_legal", "forced_win", "forced_defense"}
    assert statistics["root_visited_moves_mean"] is not None
    assert 0 <= statistics["root_max_visit_share_mean"] <= 1


def test_match_and_rapfi_share_the_paired_schedule(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    model = Network("hybrid-8-1")
    other = Network("hybrid-8-1")
    neural = match(model, cfg(simulations=1), "cpu", other, pairs=1)
    against_engine = evaluate_rapfi(model, cfg(search="policy"), "cpu", engine, tmp_path,
                                    pairs=1, threads=1, hash_mb=8, max_nodes=10, timeout=2)
    # Both arms play one opening pair as both colours, which is what makes the
    # per-pair delta meaningful.
    for report in (neural, against_engine):
        assert report["by_color"]["1"]["wins"] + report["by_color"]["1"]["losses"] == 1
        assert report["by_color"]["-1"]["wins"] + report["by_color"]["-1"]["losses"] == 1
    assert paired_delta([{"pair": 0, "color": 1, "result": 1}],
                        [{"pair": 0, "color": 1, "result": -1}])["delta_pp"] == 100.0


def test_paired_delta_needs_shared_openings():
    assert paired_delta([{"pair": 0, "color": 1, "result": 1}],
                        [{"pair": 7, "color": 1, "result": 1}]) is None


def test_summary_rejects_an_incomplete_run():
    report = summary([{"result": 1, "color": 1, "winner": 1, "pair": 0}], 2)
    assert not report["complete"] and report["games"] == 1
