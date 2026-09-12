"""Target construction is arithmetic, so it is tested as arithmetic."""
import numpy as np
import pytest

from vk.records import VALUE_HEADS, head_bit, head_mask, head_targets
from vk.targets import (normalized_weights, policy_target, soft_target,
                        surprise_weights, value_targets)


def test_pruning_drops_children_the_search_barely_touched():
    visits = np.zeros(225)
    visits[[3, 7, 8]] = [100, 1, 40]
    target = policy_target(visits, visits / visits.sum(), prune_prop=0.02, prune_min_count=2)
    assert target[3] > 0 and target[8] > 0
    assert target[7] == 0, "one visit out of a hundred cannot be a real choice"
    assert target.sum() == pytest.approx(1.0)


def test_pruning_keeps_the_uncertainty_among_visited_moves():
    visits = np.zeros(225)
    visits[[0, 1, 2]] = [50, 40, 30]
    target = policy_target(visits, None, prune_prop=0.02, prune_min_count=2)
    assert np.count_nonzero(target) == 3
    assert target[0] > target[1] > target[2]


def test_pruning_an_empty_distribution_is_safe():
    assert not policy_target(np.zeros(225), None).any()


def test_pruning_falls_back_to_the_most_visited_move():
    visits = np.zeros(225)
    visits[[5, 6]] = [1, 1]
    target = policy_target(visits, None, prune_prop=1.0, prune_min_count=1)
    assert np.count_nonzero(target) >= 1 and target.sum() == pytest.approx(1.0)


def test_noise_correction_removes_the_exploration_component():
    visits = np.zeros(225)
    visits[[1, 2]] = [90, 10]
    noise = np.zeros(225)
    noise[[1, 2]] = [0.5, 0.5]
    corrected = policy_target(visits, None, noise, noise_weight=0.25,
                              prune_prop=0.0, prune_min_count=1)
    uncorrected = policy_target(visits, None, noise, noise_weight=0.0,
                                prune_prop=0.0, prune_min_count=1)
    assert corrected[2] == 0, "the noise-only move must not survive the correction"
    assert uncorrected[2] > 0
    assert corrected[1] == pytest.approx(1.0)


def test_noise_correction_falls_back_when_it_would_zero_everything():
    visits = np.zeros(225)
    visits[4] = 10
    noise = np.zeros(225)
    noise[4] = 1.0
    target = policy_target(visits, None, noise, noise_weight=1.0,
                           prune_prop=0.0, prune_min_count=1)
    assert target[4] == pytest.approx(1.0)


def test_soft_target_smooths_without_changing_the_order():
    target = np.zeros(225)
    target[[0, 1, 2]] = [0.8, 0.15, 0.05]
    identity = soft_target(target, 1.0)
    soft = soft_target(target, 3.0)
    assert np.allclose(identity, target, atol=1e-6)
    assert soft[0] < target[0] and soft[2] > target[2]
    assert soft.sum() == pytest.approx(1.0)
    assert np.argmax(soft) == 0


def test_value_targets_read_the_outcome_inside_the_horizon():
    values, valid, stats = value_targets([0.5, 0.5, 0.5, 0.5], winner=1,
                                         short_plies=4, mid_plies=4)
    assert list(values[:, VALUE_HEADS.index("final")]) == pytest.approx([1, -1, 1, -1])
    assert valid.all()
    # Both horizons span the whole game, so nothing bootstraps and the horizon
    # heads agree with the final head exactly.
    assert stats["short_bootstrap"] == 0 and stats["mid_bootstrap"] == 0
    assert list(values[:, VALUE_HEADS.index("short")]) == pytest.approx([1, -1, 1, -1])


def test_value_targets_bootstrap_with_a_sign_flip_per_odd_ply():
    search_values = [0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
    values, valid, stats = value_targets(search_values, winner=0, short_plies=2,
                                         mid_plies=4)
    assert stats["short_bootstrap"] == 4 and stats["mid_bootstrap"] == 2
    # Ply 0 reaches ply 2 two plies later: the side to move is the same, so the
    # bootstrap keeps its sign.
    assert values[0, VALUE_HEADS.index("short")] == pytest.approx(0.0)
    assert values[1, VALUE_HEADS.index("short")] == pytest.approx(0.2)
    # An odd horizon flips it: ply 0 reaches ply 1, where the other player moves.
    values, _, _ = value_targets([0.0, 0.6, 0.0, 0.0], winner=0, short_plies=1,
                                 mid_plies=1)
    assert values[0, VALUE_HEADS.index("short")] == pytest.approx(-0.6)
    assert values[0, VALUE_HEADS.index("mid")] == pytest.approx(-0.6)


def test_value_targets_mark_unknown_outcomes_and_missing_searches_invalid():
    values, valid, stats = value_targets([np.nan, np.nan, np.nan], winner=127,
                                         short_plies=1, mid_plies=2)
    assert not valid.any() and np.isnan(values).all()
    # A horizon that lands on a ply with no search value cannot be supervised;
    # one that reaches the end of the game still can.
    values, valid, _ = value_targets([0.0, np.nan, 0.0, 0.0], winner=1,
                                     short_plies=1, mid_plies=2)
    short = head_bit("short")
    assert not bool(valid[0] & short), "the bootstrap for ply 0 is missing"
    assert bool(valid[3] & short), "ply 3 sees the game end inside its horizon"
    assert all(bool(row & head_bit("final")) for row in valid)


def test_surprise_weights_are_bounded_and_monotone():
    cfg = {"surprise_policy_weight": 1.0, "surprise_value_weight": 0.5,
           "surprise_ref": 1.0, "surprise_cap": 5.0, "surprise_uniform_share": 0.5}
    weights = surprise_weights([0.0, 1.0, 100.0], [0.0, 0.0, 0.0], cfg)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] > weights[0] and weights[2] > weights[1]
    assert weights[2] == pytest.approx(3.0), "the cap must bound an outlier"
    assert not np.isnan(surprise_weights([np.nan], [np.nan], cfg)).any()


def test_zero_reference_degrades_to_uniform_weights():
    cfg = {"surprise_policy_weight": 1.0, "surprise_value_weight": 0.0,
           "surprise_ref": 0.0, "surprise_cap": 5.0, "surprise_uniform_share": 0.5}
    weights = surprise_weights([3.0, 1.0], [0.0, 0.0], cfg)
    # With no reference there is no surprise signal, so every row is equal and
    # the sampler is uniform again.
    assert weights[0] == weights[1]


def test_normalized_weights_reject_garbage():
    assert normalized_weights([1.0, 3.0]) == pytest.approx([0.5, 1.5])
    for bad in ([1.0, -1.0], [0.0, 0.0], [np.nan, 1.0]):
        with pytest.raises(ValueError):
            normalized_weights(bad)
