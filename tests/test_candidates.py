import numpy as np
import pytest

from vk.candidates import (forced_candidates, four_threats, hard_candidates,
                           immediate_wins, legal_candidates, tactical_bias)
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
    candidates = hard_candidates(game, "forced")
    assert candidates.mode == "forced_win"
    assert np.array_equal(candidates.mask, immediate_wins(game))


def test_broken_four_and_unique_boundary_defense():
    game = Game()
    game.board[[105, 106, 108, 109]] = 1
    assert np.flatnonzero(hard_candidates(game, "forced").mask).tolist() == [107]
    defense = Game()
    defense.board[105:109] = -1
    result = hard_candidates(defense, "forced")
    assert result.mode == "forced_defense"
    assert np.flatnonzero(result.mask).tolist() == [109]


def test_multiple_threats_are_all_preserved():
    game = Game()
    game.board[105:109] = -1
    game.board[[7, 22, 37, 52]] = -1
    result = hard_candidates(game, "forced")
    assert result.mask[109] and result.mask[67]
    assert result.mask.sum() == 2


def test_center_normal_range_legality_and_all_symmetries():
    empty = hard_candidates(Game(), "forced")
    assert empty.mode == "all_legal" and empty.mask.sum() == 225
    game = Game()
    game.board[[31, 112, 173]] = [1, -1, 1]
    original = tactical_bias(game).reshape(15, 15)
    assert not original.reshape(-1)[game.board != 0].any()
    for index in range(8):
        transformed = Game(board=symmetry(game.board.reshape(15, 15), index), player=game.player)
        actual = tactical_bias(transformed).reshape(15, 15)
        assert np.array_equal(actual, symmetry(original, index))


def test_the_bias_never_removes_a_legal_point():
    """The whole point of the split: a heuristic mistake stays recoverable."""
    game = place([(7, 7), (5, 5)])
    bias = tactical_bias(game)
    legal = game.legal()
    assert np.all(bias[~legal] == 0)
    # Every legal point still has a finite score, biased or not.
    assert np.isfinite(bias[legal]).all()
    assert (bias[legal] >= 0).all()
    assert (bias[legal] > 0).any()


def test_line_distance_four_is_biased_but_not_exclusive():
    game = Game()
    game.board[112] = 1
    bias = tactical_bias(game).reshape(15, 15)
    assert bias[7, 11] > 0 and bias[11, 7] > 0 and bias[11, 11] > 0
    assert bias[7, 12] == 0
    assert bias[0, 0] == 0
    # The far corner is still legal and still scores exactly zero bias.
    assert game.legal()[0] and bias.reshape(-1)[0] == 0


def test_an_empty_board_biases_only_the_centre():
    bias = tactical_bias(Game())
    assert np.flatnonzero(bias).tolist() == [112]


@pytest.mark.parametrize("rule", ["freestyle", "renju"])
def test_straight_three_is_a_biased_threat(rule):
    """A straight three can grow into an open four, which nothing can block."""
    game = place([(7, 5), (7, 6), (7, 7)], rule=rule)
    assert not immediate_wins(game).any()
    assert np.flatnonzero(four_threats(game)).tolist() == [7 * 15 + 4, 7 * 15 + 8]
    bias = tactical_bias(game)
    assert bias[7 * 15 + 4] > bias[7 * 15 + 3]
    assert bias[7 * 15 + 8] > bias[7 * 15 + 9]
    # Nothing is forced, so every legal point survives the hard rules.
    assert hard_candidates(game, "forced").mode == "all_legal"


def test_broken_three_threatens_the_gap_not_the_ends():
    # X.XX: only the gap turns it into four, and that four has both ends open.
    assert np.flatnonzero(four_threats(place([(7, 3), (7, 5), (7, 6)]))).tolist() == [7 * 15 + 4]
    # XX.X symmetrically.
    assert np.flatnonzero(four_threats(place([(7, 3), (7, 4), (7, 6)]))).tolist() == [7 * 15 + 5]


def test_a_blocking_move_outranks_our_own_forcing_four():
    """Regression: a forcing four does not stop an immediate five.

    Black has a three that can become an open four, but White is one move from
    five. Playing the four only delays the loss, so the node must be a
    forced_defense and the four must NOT be offered as an alternative --
    otherwise search can pick a move that loses on the spot.
    """
    game = place([(7, 5), (7, 6), (7, 7)], color=1)
    for c in (3, 4, 5, 6):
        game.board[9 * 15 + c] = -1                       # White: four in a row
    game.player = 1

    assert four_threats(game).any(), "the forcing four must exist"
    assert immediate_wins(game, -1).any(), "White must threaten five"
    candidates = hard_candidates(game, "forced")
    assert candidates.mode == "forced_defense"
    # White four is open at both ends, so both completing points must be kept.
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
    assert hard_candidates(game, "forced").mode == "forced_win"


def test_a_fromcing_three_does_not_choose_the_action_set():
    """A forcing three is strong, not forced: it must not exclude other moves."""
    game = place([(7, 5), (7, 6), (7, 7)])
    assert four_threats(game).any()
    result = forced_candidates(game)
    assert result.mode == "all_legal"
    assert np.array_equal(result.mask, game.legal())


def test_hard_rules_only_keep_determined_moves():
    win = place([(7, 5), (7, 6), (7, 7), (7, 8)])
    result = hard_candidates(win, "forced")
    assert result.mode == "forced_win"
    assert np.array_equal(np.flatnonzero(result.mask), np.flatnonzero(immediate_wins(win)))

    defence = Game()
    defence.board[105:109] = -1
    result = hard_candidates(defence, "forced")
    assert result.mode == "forced_defense"
    assert np.flatnonzero(result.mask).tolist() == [109]

    empty = hard_candidates(Game(), "forced")
    assert empty.mode == "all_legal" and int(empty.mask.sum()) == 225
    # Under Renju the empty board is not "unconstrained": the centre is forced
    # by the rule, so even the unpruned mode offers exactly one point.
    renju = hard_candidates(Game("renju"), "forced")
    assert renju.mode == "all_legal"
    assert np.flatnonzero(renju.mask).tolist() == [112]


def test_hard_rules_can_be_switched_off_and_typos_are_rejected():
    game = place([(7, 5), (7, 6), (7, 7), (7, 8)])
    assert hard_candidates(game, "none").mode == "all_legal"
    assert np.array_equal(hard_candidates(game, "none").mask, game.legal())
    with pytest.raises(ValueError, match="Unknown hard rule mode"):
        hard_candidates(game, "tactical")


def test_legal_candidates_offer_every_empty_point():
    game = place([(7, 7), (8, 8)])
    result = legal_candidates(game)
    assert result.mode == "all_legal"
    assert int(result.mask.sum()) == 223
    assert not result.mask[7 * 15 + 7] and not result.mask[8 * 15 + 8]
