"""Critical-regret weighting: the budget must move to positions that differ.

The gap between the best and second-best analysed move says how much the choice
costs. Weighting is applied in the batch *sampler*, so the objective is
unchanged and only the attention shifts.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from vk.datasets import DEFAULT_GAP_WEIGHTS, gap_weights, topk_gap, weighted_indices
from vk.pretraining import pretrain


def winrates(rows):
    """Top-k winrate rows; ``None`` means "no data", i.e. NaN slots."""
    out = np.full((len(rows), 5), np.nan, np.float16)
    for index, values in enumerate(rows):
        for slot, value in enumerate(values):
            out[index, slot] = np.float16(value)
    return out


def actions(rows):
    out = np.full((len(rows), 5), 255, np.uint8)
    for index, values in enumerate(rows):
        for slot, value in enumerate(values):
            out[index, slot] = np.uint8(value)
    return out


def test_gap_is_the_top1_top2_difference():
    rates = winrates([[0.7, 0.6, 0.1], [0.5, 0.5, 0.4]])
    gap = topk_gap(rates)
    assert gap[0] == pytest.approx(0.1, abs=1e-3)
    assert gap[1] == pytest.approx(0.0, abs=1e-3)


def test_rows_without_second_move_have_no_gap_and_no_evidence():
    rates = winrates([[0.7], [0.6, 0.55]])
    slots = actions([[3], [4, 5]])
    gap = topk_gap(rates, slots)
    assert np.isnan(gap[0]) and gap[1] == pytest.approx(0.05, abs=1e-3)
    weights = gap_weights(rates, actions=slots)
    # The row with no usable gap keeps a weight but not a boosted one.
    assert weights[0] < weights[1]


def test_weights_follow_the_documented_buckets():
    rates = winrates([[0.5, 0.4995],    # gap < 0.01  -> 0.25
                      [0.5, 0.485],     # 0.01-0.03    -> 0.5
                      [0.5, 0.46],      # 0.03-0.05    -> 1.0
                      [0.5, 0.42],      # 0.05-0.10    -> 2.0
                      [0.5, 0.30]])     # > 0.10       -> 4.0
    weights = gap_weights(rates)
    assert list(weights) == [0.25, 0.5, 1.0, 2.0, 4.0]
    assert DEFAULT_GAP_WEIGHTS[-1][1] == 4.0


def test_weighted_indices_prefers_heavy_rows_without_repeating():
    weights = np.array([0.0, 0.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    drawn = weighted_indices(weights, 2, rng)
    assert sorted(drawn.tolist()) == [2, 3]
    with pytest.raises(ValueError, match="weights"):
        weighted_indices(np.zeros(4), 1, rng)


def make_dataset(root, rows=32, rule="freestyle"):
    """A format-3 set whose gap varies by row, so weighting has something to do."""
    root.mkdir(parents=True, exist_ok=True)
    state = np.zeros((rows, 3, 15, 15), np.uint8)
    state[:, 2] = 1
    policy = np.zeros((rows, 225), np.float16)
    policy[:, 112] = 1
    rates = np.full((rows, 5), np.nan, np.float16)
    slots = np.full((rows, 5), 255, np.uint8)
    for row in range(rows):
        # Half the rows are "every move equal", half are a real decision.
        gaps = 0.001 if row % 2 == 0 else 0.2
        for slot in range(3):
            rates[row, slot] = np.float16(0.9 - slot * gaps)
            slots[row, slot] = np.uint8(100 + slot)
    for split in ("train", "validation", "test"):
        (root / split).mkdir(exist_ok=True)
        np.savez_compressed(root / split / "shard-00000.npz", state=state, policy=policy,
                            value=np.full(rows, 0.5, np.float16),
                            game_id=np.arange(rows, dtype=np.uint32),
                            ply=np.zeros(rows, np.uint16),
                            teacher_best=np.full(rows, 100, np.uint16),
                            teacher_nodes=np.full(rows, 5, np.uint64),
                            teacher_topk_actions=slots, teacher_topk_winrates=rates)
    (root / "manifest.json").write_text(
        json.dumps({"format_version": 3, "rule": rule}), encoding="utf-8")


def test_pretrain_can_weight_batches_by_the_gap(tmp_path):
    dataset = tmp_path / "data"
    make_dataset(dataset)
    report = pretrain(dataset, tmp_path / "weighted", steps=4, batch_size=8,
                      arch="hybrid-8-1", device="cpu", warmup_steps=1,
                      learning_rate=0.01, final_learning_rate=0.001, value_weight=0.0,
                      critical_weighting=True)
    assert report["critical_weighting"] is True
    assert report["sampling"]["teacher"]["mean_weight"] == pytest.approx(2.125, abs=1e-6)
    assert report["sampling"]["teacher"]["usable_rows"] == 32
    record = json.loads((tmp_path / "weighted" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["sampling"]["teacher"]["mean_weight"] == pytest.approx(2.125, abs=1e-6)


def test_weighting_needs_topk_data_and_says_so(tmp_path):
    """A format-2 source has no gap to weight by; failing loudly beats silently."""
    dataset = tmp_path / "v2"
    make_dataset(dataset)
    for split in ("train", "validation", "test"):
        path = dataset / split / "shard-00000.npz"
        with np.load(path, allow_pickle=False) as shard:
            arrays = {key: shard[key] for key in shard.files
                      if not key.startswith("teacher_topk")}
        np.savez_compressed(path, **arrays)
    (dataset / "manifest.json").write_text(
        json.dumps({"format_version": 2, "rule": "freestyle"}), encoding="utf-8")
    with pytest.raises(ValueError, match="top-k winrates"):
        pretrain(dataset, tmp_path / "unweighted", steps=1, batch_size=4,
                 arch="hybrid-8-1", device="cpu", warmup_steps=1,
                 value_weight=0.0, critical_weighting=True)


def test_unweighted_pretraining_is_unchanged(tmp_path):
    dataset = tmp_path / "data"
    make_dataset(dataset)
    report = pretrain(dataset, tmp_path / "uniform", steps=2, batch_size=4,
                      arch="hybrid-8-1", device="cpu", warmup_steps=1,
                      learning_rate=0.01, final_learning_rate=0.001, value_weight=0.0)
    assert report["critical_weighting"] is False and report["sampling"] is None
