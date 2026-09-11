import numpy as np
import pytest

from vk.candidates import immediate_wins, tactical_candidates
from vk.game import Game
from vk.teacher import symmetry


@pytest.mark.parametrize("direction,start", [
    ((0, 1), (7, 5)), ((1, 0), (5, 7)), ((1, 1), (5, 5)), ((1, -1), (5, 9))])
def test_forced_wins_all_lines(direction, start):
    game = Game()
    dr, dc = direction
    for index in range(4):
        game.board[(start[0] + index * dr) * 15 + start[1] + index * dc] = 1
    candidates = tactical_candidates(game)
    assert candidates.mode == "forced_win"
    assert np.array_equal(candidates.mask, immediate_wins(game))


def test_broken_four_and_unique_boundary_defense():
    game = Game()
    game.board[[105, 106, 108, 109]] = 1
    assert np.flatnonzero(tactical_candidates(game).mask).tolist() == [107]
    defense = Game()
    defense.board[105:109] = -1
    result = tactical_candidates(defense)
    assert result.mode == "forced_defense"
    assert np.flatnonzero(result.mask).tolist() == [109]


def test_multiple_threats_are_all_preserved():
    game = Game()
    game.board[105:109] = -1
    game.board[[7, 22, 37, 52]] = -1
    result = tactical_candidates(game)
    assert result.mask[109] and result.mask[67]
    assert result.mask.sum() == 2


def test_center_normal_range_legality_and_all_symmetries():
    empty = tactical_candidates(Game())
    assert np.flatnonzero(empty.mask).tolist() == [112]
    game = Game()
    game.board[[31, 112, 173]] = [1, -1, 1]
    original = tactical_candidates(game).mask.reshape(15, 15)
    assert not original.reshape(-1)[game.board != 0].any()
    for index in range(8):
        transformed = Game(board=symmetry(game.board.reshape(15, 15), index), player=game.player)
        actual = tactical_candidates(transformed).mask.reshape(15, 15)
        assert np.array_equal(actual, symmetry(original, index))


def test_line_distance_four_is_included():
    game = Game()
    game.board[112] = 1
    candidates = tactical_candidates(game).mask.reshape(15, 15)
    assert candidates[7, 11] and candidates[11, 7] and candidates[11, 11]
    assert not candidates[7, 12]
