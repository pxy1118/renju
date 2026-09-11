import numpy as np
import pytest

from vk.game import Game
from vk.openings import balanced_opening, opening_moves


@pytest.mark.parametrize("rule", ["freestyle", "renju"])
def test_opening_is_legal_and_bounded(rule):
    for seed in range(50):
        game = balanced_opening(rule, seed, 8)
        assert len(game.history) <= 8
        # Replay validates every move against the rule in force at that moment.
        replay = Game(rule)
        for action in game.history:
            assert replay.legal()[action]
            replay.move(int(action))
        assert np.array_equal(replay.board, game.board)


@pytest.mark.parametrize("rule", ["freestyle", "renju"])
def test_opening_is_reproducible_but_not_constant(rule):
    assert opening_moves(balanced_opening(rule, 5, 8)) == opening_moves(balanced_opening(rule, 5, 8))
    distinct = {tuple(opening_moves(balanced_opening(rule, seed, 8))) for seed in range(30)}
    assert len(distinct) > 25


def test_opening_moves_reports_history_in_order():
    game = balanced_opening("freestyle", 11, 6)
    assert opening_moves(game) == [int(a) for a in game.history]
    assert len(opening_moves(game)) == 6


@pytest.mark.parametrize("rule", ["freestyle", "renju"])
def test_zero_plies_leaves_the_board_empty(rule):
    game = balanced_opening(rule, 3, 0)
    assert game.history == [] and not game.board.any()


def test_renju_opening_respects_the_forced_first_move():
    game = balanced_opening("renju", 9, 4)
    assert game.history[0] == 112


def test_opening_never_exceeds_the_requested_neighbourhood():
    """After the random prefix, moves stay near stones already on the board."""
    game = balanced_opening("freestyle", 21, 10, radius=3)
    for ply, action in enumerate(game.history):
        if ply < 4:
            continue
        prior = np.zeros((15, 15), bool)
        board = Game("freestyle")
        for earlier in game.history[:ply]:
            board.move(int(earlier), validate=False)
            r, c = divmod(int(earlier), 15)
            prior[max(0, r - 3):min(15, r + 4), max(0, c - 3):min(15, c + 4)] = True
        assert prior.reshape(-1)[action]
