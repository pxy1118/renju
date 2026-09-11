"""The regret tool must measure raw winrate points, and measure them correctly.

The two paths through the tool are covered by making the engine's ranking the
only source of variation:

* ``in top-k``  -- the engine's ranking covers every legal point, so whatever
  the network picks is inside it and the move is scored straight off the root
  search, with no child query.
* ``outside top-k`` -- the engine ranks five moves that deliberately exclude
  the network's pick, so the child position must be analysed to recover the
  model move's value.

Both engines answer with winrates that differ from what the shard recorded, so
a bug that reads a stored number instead of the engine's would give a wrong
answer rather than passing.
"""
from pathlib import Path
import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from vk.datasets import FORMAT_VERSION, ShardWriter
from vk.candidates import forced_candidates
from vk.game import Game
from vk.network import Evaluator, Network, architecture_of
from vk.training import atomic_save

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "artifacts" / "diag_teacher_regret.py"
CENTRE = 112
BEST_WINRATE = 0.95
CHILD_WINRATE = 0.8


def fake_engine(path, start):
    """Rank the five lowest legal actions at or above ``start``.

    ``RapfiClient`` keeps only ``multipv`` records, so the engine can never
    return more than five moves; the window is what decides whether the model's
    pick (the centre, under the tactical candidate set) is inside it.
    """
    path.write_text(textwrap.dedent(f'''\
        import pathlib, sys
        start = {start!r}
        log = pathlib.Path("requests.txt")
        for raw in sys.stdin:
            command = raw.strip()
            if command.startswith("START"):
                print("OK", flush=True)
            elif command == "YXSHOWINFO":
                print("MESSAGE Rapfi regret-fake", flush=True)
            elif command.startswith("YXBOARD"):
                occupied = {{int(item.split(",")[1])*15 + int(item.split(",")[0])
                            for item in command.split()[1:-1]}}
            elif command.startswith("YXNBEST"):
                nbest = int(command.split()[1])
                with log.open("a") as handle:
                    handle.write(f"{{nbest}}\\n")
                if nbest == 1:
                    print("INFO PV 0", flush=True); print("INFO NUMPV 1", flush=True)
                    print("INFO DEPTH 3", flush=True)
                    print(f"INFO WINRATE {CHILD_WINRATE}", flush=True)
                    print("INFO BESTLINE 0,0", flush=True); print("INFO PV DONE", flush=True)
                    print("0,0", flush=True)
                    continue
                actions = [action for action in range(start, 225) if action not in occupied][:nbest]
                for index, action in enumerate(actions):
                    print(f"INFO PV {{index}}", flush=True)
                    print(f"INFO NUMPV {{len(actions)}}", flush=True)
                    print("INFO DEPTH 3", flush=True)
                    print(f"INFO WINRATE {{max(0.05, {BEST_WINRATE} - 0.05*index)}}", flush=True)
                    print(f"INFO BESTLINE {{action%15}},{{action//15}}", flush=True)
                    print("INFO PV DONE", flush=True)
                action = actions[0]
                print(f"{{action%15}},{{action//15}}", flush=True)
            elif command == "END":
                break
        '''), encoding="utf-8")
    return path


def make_dataset(root, topk_actions, best):
    writer = ShardWriter(root, shard_size=4)
    slots = np.full(5, 255, np.uint8)
    winrates = np.full(5, np.nan, np.float16)
    for slot, action in enumerate(topk_actions[:5]):
        slots[slot], winrates[slot] = np.uint8(action), np.float16(0.4)
    state = np.zeros((3, 15, 15), np.uint8)
    state[2] = 1                                   # Black to move
    writer.add("test", {"state": state, "policy": np.full(225, 1 / 225, np.float16),
                        "value": np.float16(0.0), "game_id": np.uint32(9),
                        "ply": np.uint16(1), "teacher_best": np.uint16(best),
                        "teacher_nodes": np.uint64(11),
                        "teacher_topk_actions": slots, "teacher_topk_winrates": winrates})
    writer.close()
    (root / "manifest.json").write_text(
        json.dumps({"format_version": FORMAT_VERSION, "rule": "freestyle"}), encoding="utf-8")


def make_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    model = Network("hybrid-8-1")
    atomic_save({"format": 1, "config": {"rule": "freestyle", "arch": "hybrid-8-1",
                                         "channels": 8, "blocks": 1},
                 "model": model.state_dict(), "step": 0}, path)


def deterministic_pick(checkpoint):
    """What this checkpoint plays on the empty board.

    An untrained network's ranking is arbitrary but reproducible, so the tests
    read it instead of guessing: the engine's window and the shard's recorded
    top-5 are then built around the move that is actually played.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Network(architecture_of(state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    game = Game("freestyle", player=1)
    logits, _ = Evaluator(model, "cpu")(game.encode())
    allowed = forced_candidates(game).mask
    return int(np.argmax(np.where(allowed, logits, -np.inf)))


def run(tmp_path, dataset, checkpoint, engine, output, extra=()):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset), "--checkpoint", str(checkpoint),
         "--engine", str(engine), "--engine-dir", str(tmp_path), "--split", "test",
         "--limit", "1", "--progress-every", "0", "--device", "cpu",
         # On an empty board the tactical mode offers only the centre, so the
         # model's move is the centre rather than an arbitrary empty point.
         "--candidates", "tactical",
         "--max-nodes", "10", "--threads", "1", "--hash-mb", "8", "--timeout", "5",
         "--output", str(output), "--cache", str(tmp_path / "cache.json"), *extra],
        capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(Path(output).read_text(encoding="utf-8"))
    if report["rows_skipped"]:
        pytest.fail(f"tool skipped the row:\n{completed.stdout}\n{completed.stderr}")
    return report


def test_in_topk_regret_uses_the_engines_winrates_on_the_raw_probability_scale(tmp_path):
    dataset, checkpoint = tmp_path / "data", tmp_path / "best.pt"
    make_checkpoint(checkpoint)
    pick = deterministic_pick(checkpoint)
    # The engine's window starts at the move the model plays, so its cost is read
    # off the same root search with no child query.
    engine = fake_engine(tmp_path / "engine.py", start=pick)
    make_dataset(dataset, topk_actions=[0, 1, 2, 3, 4], best=CENTRE)
    output = tmp_path / "regret.json"
    report = run(tmp_path, dataset, checkpoint, engine, output)
    assert report["rows_requested"] == 1 and report["rows_skipped"] == 0
    assert "raw winrate" in report["scale"]
    summary = report["a"]["all"]
    assert summary["outside_teacher_top5"] == 0.0
    assert summary["policy_top5_hit"] == 1.0
    # p_best and p_model both come from the engine's ranking, so the engine's own
    # best move costs zero. The shard recorded 0.4 for the tracked moves while
    # the engine ranks them 0.95, so a bug that read the stored number would
    # report a nonzero cost here.
    assert summary["regret_mean"] == pytest.approx(0.0, abs=1e-6)
    assert summary["mask_excludes_teacher_best"] == 0.0
    requests = (tmp_path / "requests.txt").read_text(encoding="utf-8").split()
    assert requests == ["5"], "an in-top-k move needs no child query"
    assert output.with_suffix(".csv").is_file()


def test_outside_topk_regret_asks_the_child_position(tmp_path):
    # The engine's window starts at action 0, so the model's centre move is not
    # ranked and the child position has to be analysed.
    engine = fake_engine(tmp_path / "engine.py", start=0)
    dataset, checkpoint = tmp_path / "data", tmp_path / "best.pt"
    make_dataset(dataset, topk_actions=[0, 1, 2, 3, 4], best=0)
    make_checkpoint(checkpoint)
    report = run(tmp_path, dataset, checkpoint, engine, tmp_path / "regret.json")
    summary = report["a"]["all"]
    # p_best = 0.95, p_model = 1 - 0.8 = 0.2, so regret = 0.75.
    assert summary["outside_teacher_top5"] == 1.0
    assert summary["policy_top5_hit"] == 0.0
    assert summary["regret_mean"] == pytest.approx(BEST_WINRATE - (1 - CHILD_WINRATE), abs=1e-4)
    requests = (tmp_path / "requests.txt").read_text(encoding="utf-8").split()
    assert requests == ["5", "1"], "one root query plus one child query"


def test_decided_positions_are_reported_but_not_headlined(tmp_path):
    engine = fake_engine(tmp_path / "engine.py", start=CENTRE)
    dataset, checkpoint = tmp_path / "data", tmp_path / "best.pt"
    make_dataset(dataset, topk_actions=[0, 1, 2, 3, 4], best=CENTRE)
    make_checkpoint(checkpoint)
    report = run(tmp_path, dataset, checkpoint, engine, tmp_path / "regret.json")
    # The engine's top-1 is 0.95, i.e. effectively won, so the row must be
    # excluded from the subset the conclusions are drawn from.
    assert report["a"]["all"]["rows"] == 1
    assert report["a"]["decided_excluded"] is None


def test_two_checkpoints_are_compared_on_identical_positions(tmp_path):
    """A retrained policy must be judged by a paired delta, not two separate means."""
    dataset = tmp_path / "data"
    baseline, candidate = tmp_path / "baseline.pt", tmp_path / "candidate.pt"
    make_checkpoint(baseline, seed=0)
    make_checkpoint(candidate, seed=1)
    # One window that ranks both models' picks, so neither needs a child query
    # and the two arms share every engine call.
    engine = fake_engine(tmp_path / "engine.py", start=min(deterministic_pick(baseline),
                                                          deterministic_pick(candidate)))
    make_dataset(dataset, topk_actions=[0, 1, 2, 3, 4], best=CENTRE)
    report = run(tmp_path, dataset, baseline, engine, tmp_path / "regret.json",
                 extra=("--compare-checkpoint", str(candidate)))
    assert set(report["checkpoints"]) == {"a", "b"}
    assert report["checkpoints"]["a"]["path"] != report["checkpoints"]["b"]["path"]
    paired = report["paired"]
    assert paired["rows"] == 1
    assert paired["a_cheaper"] + paired["b_cheaper"] + paired["ties"] == 1
    assert len(paired["mean_regret_delta_ci95"]) == 2
    assert "all" in report["b"]
    # The two arms share the root query; a child query only happens for an arm
    # whose move the engine did not rank, so two arms cost at most two extra.
    requests = (tmp_path / "requests.txt").read_text(encoding="utf-8").split()
    assert requests[0] == "5" and requests.count("5") == 1
    assert len(requests) <= 3
