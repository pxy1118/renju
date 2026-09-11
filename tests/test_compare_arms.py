"""The decoupled arms must be comparable, which requires per-pair records.

`paired_delta` can only subtract two arms on the *same* opening, so
`evaluate_rapfi(..., with_records=True)` has to expose one record per
``(pair, colour)`` and both arms have to walk the same schedule.
"""
from pathlib import Path
import textwrap

from vk.evaluation import evaluate_rapfi, paired_delta
from vk.network import Network
from vk.training import DEFAULTS


def fake_engine(path):
    path.write_text(textwrap.dedent('''\
        import sys
        occupied = set()
        for raw in sys.stdin:
            command = raw.strip()
            if command.startswith("START"):
                print("OK", flush=True)
            elif command == "YXSHOWINFO":
                print("MESSAGE Rapfi compare-fake", flush=True)
            elif command.startswith("YXBOARD"):
                occupied = {int(item.split(",")[1])*15 + int(item.split(",")[0])
                            for item in command.split()[1:-1]}
            elif command.startswith("YXNBEST"):
                legal = [action for action in range(225) if action not in occupied][:5]
                for index, action in enumerate(legal):
                    x, y = action % 15, action // 15
                    print(f"INFO PV {index}", flush=True)
                    print(f"INFO NUMPV {len(legal)}", flush=True)
                    print("INFO DEPTH 3", flush=True)
                    print("INFO WINRATE 0.5", flush=True)
                    print(f"INFO BESTLINE {x},{y}", flush=True)
                    print("INFO PV DONE", flush=True)
                action = legal[0]
                print(f"{action%15},{action//15}", flush=True)
            elif command == "END":
                break
        '''), encoding="utf-8")
    return path


def run(tmp_path, engine, **overrides):
    cfg = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
               simulations=2, workers=1, opening_plies=4)
    cfg.update(overrides)
    return evaluate_rapfi(Network("hybrid-8-1"), cfg, "cpu", engine, tmp_path, pairs=2,
                          threads=1, hash_mb=8, max_nodes=10, timeout=2,
                          with_records=True)


def test_records_cover_every_pair_and_colour(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    report = run(tmp_path, engine, search="policy", candidates="legal")
    records = report["records"]
    assert len(records) == 4 == report["games"]
    assert {(record["pair"], record["color"]) for record in records} == {
        (0, 1), (0, -1), (1, 1), (1, -1)}
    assert all(record["result"] in (-1, 0, 1) for record in records)
    assert "moves_detail" not in records[0], "records stay small on purpose"


def test_two_arms_are_paired_on_identical_openings(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    left = run(tmp_path, engine, search="policy", candidates="legal")
    right = run(tmp_path, engine, search="policy", candidates="forced")
    delta = paired_delta(left["records"], right["records"])
    assert delta["pairs"] == 4
    assert -100 <= delta["delta_pp"] <= 100
    assert delta["delta_ci95_pp"][0] <= delta["delta_pp"] <= delta["delta_ci95_pp"][1]


def test_arms_without_shared_openings_are_not_comparable(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    report = run(tmp_path, engine, search="policy", candidates="legal")
    assert paired_delta(report["records"][:2], report["records"][2:]) is None
