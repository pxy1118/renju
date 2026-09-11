import numpy as np
import pytest

from vk.candidates import four_threats, immediate_wins, tactical_candidates
from vk.game import Game
from vk.teacher import symmetry


def place(points, color=1, rule="freestyle"):
    game = Game(rule, player=color)
    for r, c in points:
        game.board[r * 15 + c] = color
    return game


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


@pytest.mark.parametrize("rule", ["freestyle", "renju"])
def test_straight_three_is_a_forcing_threat(rule):
    """A straight three can grow into an open four, which nothing can block."""
    game = place([(7, 5), (7, 6), (7, 7)], rule=rule)
    assert not immediate_wins(game).any()
    assert np.flatnonzero(four_threats(game)).tolist() == [7 * 15 + 4, 7 * 15 + 8]
    candidates = tactical_candidates(game)
    assert candidates.mode == "strategic"
    assert np.flatnonzero(candidates.mask).tolist() == [7 * 15 + 4, 7 * 15 + 8]


def test_broken_three_threatens_the_gap_not_the_ends():
    # X.XX: only the gap turns it into four, and that four has both ends open.
    assert np.flatnonzero(four_threats(place([(7, 3), (7, 5), (7, 6)]))).tolist() == [7 * 15 + 4]
    # XX.X symmetrically.
    assert np.flatnonzero(four_threats(place([(7, 3), (7, 4), (7, 6)]))).tolist() == [7 * 15 + 5]


def test_a_blocking_move_outranks_our_own_forcing_four():
    """Regression: a forcing four does not stop an immediate five.

    Black has a three that can become an open four, but White is one move from
    five. Playing the four only delays the loss, so the node must be a
    ``forced_defense`` and the four must NOT be offered as an alternative --
    otherwise search can pick a move that loses on the spot.
    """
    game = place([(7, 5), (7, 6), (7, 7)], color=1)
    for c in (3, 4, 5, 6):
        game.board[9 * 15 + c] = -1                       # White: four in a row
    game.player = 1

    assert four_threats(game).any(), "the forcing four must exist"
    assert immediate_wins(game, -1).any(), "White must threaten five"
    candidates = tactical_candidates(game)
    assert candidates.mode == "forced_defense"
    # White's four is open at both ends, so both completing points must be kept.
    assert np.flatnonzero(candidates.mask).tolist() == [9 * 15 + 2, 9 * 15 + 7]
    for four_move in (7 * 15 + 4, 7 * 15 + 8):
        assert not candidates.mask[four_move]


def test_blocked_and_edge_threes_are_not_forced():
    """A four whose far end is taken, or that runs off the board, is sleeping."""
    open_three = place([(7, 3), (7, 4), (7, 5)])
    assert np.flatnonzero(four_threats(open_three)).tolist() == [7 * 15 + 2, 7 * 15 + 6]
    # White blocks (7,2); now the far end of the four is never empty.
    blocked_low = place([(7, 3), (7, 4), (7, 5)])
    blocked_low.board[7 * 15 + 2] = -1
    assert not four_threats(blocked_low).any()
    blocked_high = place([(7, 3), (7, 4), (7, 5)])
    blocked_high.board[7 * 15 + 6] = -1
    assert not four_threats(blocked_high).any()
    # Against the board edge there is no room for a second way to five.
    assert not four_threats(place([(0, 12), (0, 13), (0, 14)])).any()
    assert not four_threats(Game()).any()
    assert not four_threats(place([(7, 5), (7, 6)])).any()


def test_four_threats_respects_the_mover_and_symmetry():
    game = place([(5, 4), (6, 5), (7, 6)])
    assert not four_threats(game, -1).any()
    diagonal = four_threats(game, 1).reshape(15, 15)
    assert diagonal[4, 3] and diagonal[8, 7]
    for index in range(8):
        transformed = Game(board=symmetry(game.board.reshape(15, 15), index), player=game.player)
        assert np.array_equal(four_threats(transformed, 1).reshape(15, 15),
                              symmetry(diagonal, index))


@pytest.mark.parametrize("points,expected", [
    ([(5, 7), (6, 7), (7, 7)], [(4, 7), (8, 7)]),
    ([(7, 5), (7, 6), (7, 7)], [(7, 4), (7, 8)]),
    ([(5, 5), (6, 6), (7, 7)], [(4, 4), (8, 8)]),
    ([(5, 9), (6, 8), (7, 7)], [(4, 10), (8, 6)]),
])
def test_straight_three_threatens_both_ends_in_every_direction(points, expected):
    game = place(points)
    assert sorted(divmod(int(a), 15) for a in np.flatnonzero(four_threats(game))) == expected


def test_immediate_win_outranks_a_forcing_four():
    """With a five available there is nothing to force: forced_win comes first."""
    game = place([(7, 5), (7, 6), (7, 7), (7, 8)])
    assert immediate_wins(game).any()
    assert not four_threats(game).any()
    assert tactical_candidates(game).mode == "forced_win"
