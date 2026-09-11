"""The value-label diagnostic must show whether siblings are distinguishable.

It is the tool behind "the value head is saturated because one parent position
is only ever supervised with one scalar"; these tests pin the arithmetic and the
row construction that the claim rests on.
"""
from pathlib import Path
import json

import numpy as np
import pytest

from artifacts import value_dataset_diagnostic as diagnostic
from vk.datasets import FORMAT_VERSION, ShardWriter
from vk.game import Game


def make_dataset(root, rows=3, topk=True):
    writer = ShardWriter(root, shard_size=8)
    for index in range(rows):
        actions = np.full(5, 255, np.uint8)
        winrates = np.full(5, np.nan, np.float16)
        if topk:
            for slot, (action, winrate) in enumerate([(112, 0.8), (113, 0.5), (114, 0.79)]):
                actions[slot], winrates[slot] = np.uint8(action), np.float16(winrate)
        state = np.zeros((3, 15, 15), np.uint8)
        state[2] = 1
        writer.add("train", {"state": state, "policy": np.full(225, 1 / 225, np.float16),
                             "value": np.float16(0.6), "game_id": np.uint32(index),
                             "ply": np.uint16(1), "teacher_best": np.uint16(112),
                             "teacher_nodes": np.uint64(5),
                             "teacher_topk_actions": actions,
                             "teacher_topk_winrates": winrates})
    writer.close()
    (root / "manifest.json").write_text(
        json.dumps({"format_version": FORMAT_VERSION, "rule": "freestyle"}), encoding="utf-8")


def test_sibling_spread_is_the_gap_between_the_best_and_worst_analysed_move():
    winrates = np.array([[0.8, 0.5, 0.79], [0.5, 0.5, np.nan]], np.float64)
    spread = diagnostic.sibling_spread(winrates)
    assert spread["parents"] == 2
    # Rows are (max - min) over the analysed slots: 0.3 and 0.0.
    assert spread["mean_sibling_spread"] == pytest.approx(0.15, abs=1e-6)
    assert spread["median_sibling_spread"] == pytest.approx(0.15, abs=1e-6)


def test_saturation_reports_the_share_the_value_head_actually_sees():
    values = np.array([0.99, -0.98, 0.1, 0.95])
    report = diagnostic.saturation(values)
    assert report["rows"] == 4
    assert report["abs_gt_0.9_share"] == pytest.approx(0.75)
    assert report["abs_lt_0.5_share"] == pytest.approx(0.25)


def test_each_parent_expands_into_one_row_per_analysed_move():
    """Three analysed moves of one parent become three separately labelled children."""
    data = {"state": np.zeros((1, 3, 15, 15), np.uint8),
            "teacher_topk_actions": np.full((1, 5), 255, np.uint8),
            "teacher_topk_winrates": np.full((1, 5), np.nan, np.float16)}
    for slot, (action, winrate) in enumerate([(112, 0.8), (113, 0.5), (114, 0.79)]):
        data["teacher_topk_actions"][0, slot] = action
        data["teacher_topk_winrates"][0, slot] = winrate
    rows = list(diagnostic.build_rows(data, limit=0, stride=1))
    assert [int(row["action"]) for row in rows] == [112, 113, 114]
    # Chained labels: the child of a 0.8 move is worth 1 - 2*0.8 from the
    # opponent's side, i.e. -0.6 for the mover.
    assert [float(row["value"]) for row in rows] == pytest.approx([-0.6, 0.0, -0.58], abs=1e-3)
    # Every child state is a legal successor of the parent, one stone deeper.
    for row in rows:
        assert int(np.count_nonzero(row["state"][0]) + np.count_nonzero(row["state"][1])) == 1
    assert len({row["action"] for row in rows}) == 3


def test_a_parent_with_one_analysed_move_is_skipped(tmp_path):
    data = {"state": np.zeros((1, 3, 15, 15), np.uint8),
            "teacher_topk_actions": np.full((1, 5), 255, np.uint8),
            "teacher_topk_winrates": np.full((1, 5), np.nan, np.float16)}
    data["teacher_topk_actions"][0, 0] = 112
    data["teacher_topk_winrates"][0, 0] = 0.8
    assert list(diagnostic.build_rows(data, limit=0, stride=1)) == []
