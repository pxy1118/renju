"""Replay sampling is where surprise weighting either bites or does not."""
import numpy as np
import pytest

from vk.config import DEFAULTS
from vk.replay import ReplayBuffer
from vk.records import DTYPE, blank, set_head


def records(rows, surprise=0.0, full_search=True, policy_valid=True):
    batch = blank(rows)
    batch["state"][:, 2] = 1
    batch["policy"][:, 0] = 1.0
    batch["policy_valid"] = 1 if policy_valid else 0
    batch["policy_weight"] = 1.0 if policy_valid else 0.0
    batch["policy_surprise"] = surprise
    batch["value_surprise"] = surprise
    batch["full_search"] = 1 if full_search else 0
    set_head(batch, "final", np.ones(rows, np.float32))
    return batch


def test_capacity_is_a_hard_ceiling_and_drops_the_oldest_rows():
    buffer = ReplayBuffer(capacity=5)
    buffer.extend(records(3))
    buffer.extend(records(4))
    assert len(buffer) == 5
    assert len(buffer.array()) == 5
    buffer.extend(records(1))
    assert len(buffer) == 5
    buffer.clear()
    assert len(buffer) == 0 and len(buffer.array()) == 0


def test_replay_only_accepts_the_current_schema():
    buffer = ReplayBuffer(capacity=4)
    with pytest.raises(ValueError, match="current schema"):
        buffer.extend(np.zeros(2, np.dtype([("state", np.uint8)])))
    buffer.extend(blank(0))
    assert len(buffer) == 0


def test_state_round_trips_through_a_checkpoint():
    buffer = ReplayBuffer(capacity=10)
    buffer.extend(records(3))
    restored = ReplayBuffer.from_state(10, buffer.to_state())
    assert len(restored) == 3 and restored.array().dtype == DTYPE
    assert np.array_equal(restored.array()["policy"], buffer.array()["policy"])


def test_an_empty_buffer_refuses_to_sample():
    buffer = ReplayBuffer(capacity=4)
    with pytest.raises(ValueError, match="empty"):
        buffer.sample(2, np.random.default_rng(0), dict(DEFAULTS))


def test_surprising_rows_are_drawn_more_often():
    buffer = ReplayBuffer(capacity=1000)
    buffer.extend(records(50, surprise=0.0))
    buffer.extend(records(50, surprise=1.0))
    cfg = dict(DEFAULTS, surprise_uniform_share=0.5, surprise_cap=5.0, surprise_ref=1.0)
    rng = np.random.default_rng(7)
    drawn = buffer.sample(4000, rng, cfg)
    high = int(np.sum(drawn["policy_surprise"] > 0.5))
    assert 0.25 * 4000 < high < 0.9 * 4000, high


def test_a_capped_outlier_cannot_take_over_the_batch():
    buffer = ReplayBuffer(capacity=1000)
    buffer.extend(records(100, surprise=0.0))
    buffer.extend(records(1, surprise=1000.0))
    cfg = dict(DEFAULTS, surprise_uniform_share=0.5, surprise_cap=5.0, surprise_ref=1.0)
    weights = buffer.weights(cfg)
    # (1 - share) + share * cap: the floor and the ceiling of the mix.
    assert weights.max() == pytest.approx(3.0), "the cap bounds the outlier"
    assert weights[0] == pytest.approx(0.5), "the uniform share keeps a floor"
    drawn = buffer.sample(2000, np.random.default_rng(11), cfg)
    odd = int(np.sum(drawn["policy_surprise"] > 500))
    assert odd < 2000 * 0.2, "one row must never dominate the batch"


def test_stats_and_weights_describe_the_buffer():
    buffer = ReplayBuffer(capacity=10)
    buffer.extend(records(4, surprise=1.0, full_search=False))
    buffer.extend(records(4, surprise=0.0, policy_valid=False))
    report = buffer.stats(dict(DEFAULTS))
    assert report["size"] == 8 and report["capacity"] == 10
    assert report["full_search_share"] == pytest.approx(0.5)
    assert report["policy_valid_share"] == pytest.approx(0.5)
    assert report["weight_max"] >= report["weight_mean"]
