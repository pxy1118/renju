import numpy as np
import pytest
from az.game import Game, forbidden, fours, DIRECTIONS


def position(black=(), white=(), rule="renju", player=1):
    g = Game(rule, player=player)
    for r,c in black:
        g.board[r*15+c] = 1
    for r,c in white:
        g.board[r*15+c] = -1
    return g


@pytest.mark.parametrize("rule", ["freestyle", "renju"])
@pytest.mark.parametrize("dr,dc", DIRECTIONS)
@pytest.mark.parametrize("color", [1,-1])
def test_four_directions(rule,dr,dc,color):
    g = Game(rule,player=color)
    for k in range(4):
        g.board[(5+k*dr)*15+7+k*dc] = color
    g.move((5+4*dr)*15+7+4*dc)
    assert g.winner == color


def test_opening_and_occupied():
    g = Game("renju")
    assert np.flatnonzero(g.legal()).tolist() == [112]
    with pytest.raises(ValueError):
        g.move(0)
    g.move(112)
    assert g.legal().sum() == 224
    with pytest.raises(ValueError):
        g.move(112)


def test_overline_and_simultaneous_five():
    g = position([(7,c) for c in (3,4,5,6,8)])
    assert forbidden(g.board.tobytes(),112) == "overline"
    assert not g.legal()[112]
    for r in (3,4,5,6):
        g.board[r*15+7] = 1
    assert forbidden(g.board.tobytes(),112) is None
    g.move(112)
    assert g.winner == 1


@pytest.mark.parametrize("rule,color", [("freestyle",1),("freestyle",-1),("renju",-1)])
def test_overline_wins_other_modes(rule,color):
    g = Game(rule,player=color)
    for c in (3,4,5,6,8):
        g.board[7*15+c] = color
    g.move(112)
    assert g.winner == color


def test_double_four_and_single_open_four():
    g = position([(7,5),(7,6),(7,8)])
    assert forbidden(g.board.tobytes(),112) is None
    b = g.board.copy()
    b[112] = 1
    assert len(fours(b,112,(0,1))) == 1
    for r in (5,6,8):
        g.board[r*15+7] = 1
    assert forbidden(g.board.tobytes(),112) == "double_four"


def test_same_direction_double_four():
    # X.XXX.X has two distinct fours: X.XXX and XXX.X.
    g = position([(7,c) for c in (3,5,7,9)])
    assert forbidden(g.board.tobytes(),7*15+6) == "double_four"


def test_true_and_blocked_double_three():
    g = position([(7,6),(7,8),(6,7),(8,7)])
    assert forbidden(g.board.tobytes(),112) == "double_three"
    g.board[7*15+5] = -1
    g.board[7*15+9] = -1
    assert forbidden(g.board.tobytes(),112) is None


def test_recursive_false_three():
    # Horizontal three's sole open-four continuation (7,5) is itself
    # a forbidden double-three; it must not count as a real three.
    g = position([(7,6),(7,8),(6,7),(8,7),(6,5),(8,5),(6,4),(8,6)], [(7,10)])
    after = g.board.copy()
    after[112] = 1
    assert forbidden(after.tobytes(),7*15+5) == "double_three"
    assert forbidden(g.board.tobytes(),112) is None


def test_edge_three_is_not_open():
    g = position([(0,1),(0,2),(1,0),(2,0)])
    assert forbidden(g.board.tobytes(),0) is None


def test_draw_and_copy_and_encoding():
    g = Game()
    # Two-by-one stagger avoids five in any direction.
    g.board = np.array([1 if (r+2*c)%4 < 2 else -1 for r in range(15) for c in range(15)],np.int8)
    g.board[0] = 0
    g.move(0)
    assert g.winner == 0
    clone = g.copy()
    clone.board[0] = -1
    assert g.board[0] == 1
    assert g.encode()[2].sum() == 0
    with pytest.raises(ValueError):
        g.move(1)


def test_mask_equals_full_forbidden_and_symmetry():
    rng = np.random.default_rng(4)
    g = Game("renju")
    for _ in range(22):
        if g.adjudicate() is not None:
            break
        a = int(rng.choice(np.flatnonzero(g.legal())))
        g.move(a)
    g.player = 1
    mask = g.legal()
    for a in np.flatnonzero(g.board == 0):
        assert mask[a] == (forbidden(g.board.tobytes(),int(a)) is None)
    rotated = Game("renju",np.rot90(g.board.reshape(15,15)),1)
    assert np.array_equal(rotated.legal().reshape(15,15),np.rot90(mask.reshape(15,15)))


def test_no_legal_moves_loses_before_full_board():
    g = Game("renju")
    g.board = np.array([1 if (r+2*c)%4 < 2 else -1 for r in range(15) for c in range(15)],np.int8)
    for c in (4,5,6,8,9):
        g.board[7*15+c] = 1
    g.board[112] = 0
    assert not g.legal().any()
    assert g.adjudicate() == -1
