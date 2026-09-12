"""The position schema is the contract between production, storage and training."""
import numpy as np
import pytest

from vk.game import Game
from vk.records import (ACTION_NONE, DTYPE, FIELDS, SCHEMA_VERSION, SOURCE_TEACHER_LEGACY,
                        VALUE_HEADS, WINNER_UNKNOWN, augment, augment_batch, blank,
                        head_mask, head_targets, normalize_legacy, set_head, stack,
                        value_valid_mask)


def test_the_dtype_carries_every_declared_field():
    assert DTYPE.names == FIELDS
    assert blank(3).dtype == DTYPE
    assert SCHEMA_VERSION == 4


def test_a_blank_batch_says_everything_is_unset():
    rows = blank(4)
    assert not rows["policy_valid"].any()
    assert not rows["policy"].any()
    assert np.isnan(rows["value"]).all()
    assert not rows["value_valid"].any()
    assert np.isnan(rows["search_value"]).all()
    assert (rows["winner"] == WINNER_UNKNOWN).all()
    assert (rows["teacher_topk_actions"] == ACTION_NONE).all()
    assert np.allclose(rows["weight"], 1.0)


def test_value_valid_is_the_mask_and_nan_is_its_shadow():
    rows = blank(2)
    set_head(rows, "final", np.array([1.0, -1.0], np.float32))
    set_head(rows, "mid", np.array([np.nan, 0.5], np.float32))
    mask = value_valid_mask(rows)
    assert mask[0].tolist() == [True, False, False]
    assert mask[1].tolist() == [True, True, False]
    assert head_mask(rows, "final").all() and not head_mask(rows, "short").any()
    assert np.isnan(head_targets(rows, "short")).all()
    assert head_targets(rows, "final") == pytest.approx([1.0, -1.0])


def test_the_schema_round_trips_through_stacking():
    state = Game().encode().astype(np.uint8)
    rows = stack([{"state": state, "policy": np.full(225, 1 / 225, np.float16),
                   "policy_valid": 1, "ply": index, "winner": 1} for index in range(3)])
    assert rows.dtype == DTYPE
    assert rows["ply"].tolist() == [0, 1, 2]
    assert rows["policy_valid"].tolist() == [1, 1, 1]
    assert not rows["value_valid"].any(), "a stacked row without a value stays invalid"


def test_augmentation_moves_board_and_policy_together():
    state = np.zeros((3, 15, 15), np.uint8)
    state[0, 2, 4] = 1
    policy = np.zeros(225)
    policy[34] = 1
    seen = set()
    for rotation in range(4):
        for mirror in (False, True):
            augmented_state, augmented_policy = augment(state, policy, rotation, mirror)
            assert np.array_equal(augmented_state[0].ravel(), augmented_policy)
            seen.add(augmented_policy.tobytes())
    assert len(seen) == 8

    states = np.stack([state] * 6)
    policies = np.stack([policy] * 6)
    out_states, out_policies = augment_batch(states, policies, np.random.default_rng(3))
    for index in range(6):
        assert np.array_equal(out_states[index][0].ravel(), out_policies[index])


def legacy_arrays(rows=2, version=3):
    arrays = {"state": np.zeros((rows, 3, 15, 15), np.uint8),
              "policy": np.full((rows, 225), 1 / 225, np.float16),
              "value": np.array([0.6, -0.4], np.float16)[:rows],
              "game_id": np.arange(rows, dtype=np.uint32),
              "ply": np.arange(rows, dtype=np.uint16),
              "teacher_best": np.full(rows, 112, np.uint16),
              "teacher_nodes": np.full(rows, 5000, np.uint64)}
    if version >= 3:
        arrays["teacher_topk_actions"] = np.full((rows, 5), ACTION_NONE, np.uint8)
        arrays["teacher_topk_winrates"] = np.full((rows, 5), np.nan, np.float16)
    return arrays


def test_a_legacy_shard_becomes_a_fully_marked_batch():
    rows = normalize_legacy(legacy_arrays(), 3)
    assert rows.dtype == DTYPE
    assert rows["source"].tolist() == [SOURCE_TEACHER_LEGACY] * 2
    assert rows["policy_valid"].all() and rows["full_search"].all()
    assert head_targets(rows, "final") == pytest.approx([0.6, -0.4], abs=1e-3)
    assert not head_mask(rows, "mid").any(), "the old files never recorded a horizon"
    assert rows["winner"].tolist() == [WINNER_UNKNOWN] * 2
    assert rows["simulations"].tolist() == [5000, 5000]


def test_normalising_an_unknown_version_fails_loudly():
    with pytest.raises(ValueError, match="Cannot normalise"):
        normalize_legacy(legacy_arrays(), 9)
    with pytest.raises(ValueError, match="missing fields"):
        normalize_legacy({"state": np.zeros((1, 3, 15, 15), np.uint8)}, 2)
