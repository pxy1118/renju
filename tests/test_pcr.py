"""Playout cap randomization decides which moves are worth supervising."""
import numpy as np
import pytest

from vk.config import DEFAULTS
from vk.game import Game
from vk.network import Inference
from vk.selfplay import play_game, search_plan


def uniform(state):
    return Inference.leaf_value(np.zeros(225), 0.0)


def test_the_full_budget_is_never_exceeded_by_the_cheap_one():
    cfg = dict(DEFAULTS, simulations=8, cheap_search_simulations=64, cheap_search_prob=0.5)
    rng = np.random.default_rng(0)
    budgets = [search_plan(cfg, rng) for _ in range(50)]
    assert max(budget for budget, _ in budgets) == 8
    assert {budget for budget, full in budgets if not full} == {8}


def test_the_cap_probability_is_honoured_and_reproducible():
    cfg = dict(DEFAULTS, simulations=100, cheap_search_simulations=10,
               cheap_search_prob=0.75)
    first = [search_plan(cfg, np.random.default_rng(5)) for _ in range(1)]
    rng = np.random.default_rng(5)
    first = [search_plan(cfg, rng) for _ in range(400)]
    rng = np.random.default_rng(5)
    again = [search_plan(cfg, rng) for _ in range(400)]
    assert first == again
    cheap = sum(1 for _, full in first if not full)
    assert 250 < cheap < 350, cheap
    assert {budget for budget, full in first if full} == {100}
    assert {budget for budget, full in first if not full} == {10}


def test_a_cheap_move_carries_no_policy_target_by_default():
    cfg = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
               simulations=8, cheap_search_simulations=2, cheap_search_prob=1.0,
               opening_plies=0)
    records, stats = play_game(cfg, uniform, 3, game_id=3)
    assert len(records) > 0
    assert stats["cheap_search_share"] == 1.0
    assert not records["policy_valid"].any()
    assert not records["policy"].any()
    assert (records["full_search"] == 0).all()
    # Value data is still produced: that is the point of the cheap searches.
    assert (records["value_valid"] & 1).all()


def test_a_full_move_carries_a_policy_target():
    cfg = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
               simulations=8, cheap_search_prob=0.0, opening_plies=0)
    records, stats = play_game(cfg, uniform, 4, game_id=4)
    assert stats["cheap_search_share"] == 0.0
    assert records["policy_valid"].any()
    assert (records["full_search"] == 1).all()


def test_a_forced_move_is_played_but_not_supervised():
    """One legal move under the hard rules teaches nothing about move choice."""
    game = Game("freestyle", player=1)
    game.board[110:114] = -1          # White threatens five on the row
    game.board[114] = 1               # ... but one end is already taken
    game.player = 1
    assert game.legal().any()
    cfg = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
               simulations=4, cheap_search_prob=0.0, opening_plies=0)
    records, stats = play_game(cfg, uniform, 1, opening=game, game_id=1)
    assert records[0]["policy_valid"] == 0, "a single candidate is not a choice"
    assert not records[0]["policy"].any()
    assert stats["forced_defense_count"] >= 1
    assert stats["moves"][:1] == [109], "the only defence must be played"
    # The caller's position must be untouched: the game is played on a copy.
    assert game.board[114] == 1 and not game.history
