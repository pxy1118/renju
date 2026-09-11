"""Shared tactical move candidates used by every local MCTS caller."""
from dataclasses import dataclass
import numpy as np

from .game import DIRECTIONS, SIZE, lengths, forbidden


@dataclass(frozen=True)
class CandidateSet:
    mask: np.ndarray
    mode: str


def _legal_for_color(game, color):
    mask = game.board == 0
    if game.rule == "renju" and color == 1:
        position = game.board.tobytes()
        for action in np.flatnonzero(mask):
            mask[action] = forbidden(position, int(action)) is None
    return mask


def immediate_wins(game, color=None):
    """Return legal placements that immediately win for ``color``."""
    color = game.player if color is None else int(color)
    legal = game.legal() if color == game.player else _legal_for_color(game, color)
    wins = np.zeros(SIZE * SIZE, dtype=bool)
    board = game.board.copy()
    for action in np.flatnonzero(legal):
        action = int(action)
        board[action] = color
        spans = lengths(board, action, color)
        wins[action] = 5 in spans if game.rule == "renju" and color == 1 else max(spans) >= 5
        board[action] = 0
    return wins


def four_threats(game, color=None):
    """Legal placements after which ``color`` has a forced four.

    A forced four is four in a row with both outer ends still empty, so the
    opponent's single stone cannot cover both ways to five. An immediate-five
    test cannot see this -- the move creates a four, not a five -- which is why
    a straight three can be lethal while ``immediate_wins`` reports nothing.

    A run is only forced when it is exactly four long with both ends empty: a
    four against the board edge, or one whose far end is blocked, is a sleeping
    four the opponent can defuse with one move.
    """
    color = game.player if color is None else int(color)
    legal = game.legal() if color == game.player else _legal_for_color(game, color)
    board = game.board
    threats = np.zeros(SIZE * SIZE, dtype=bool)
    for action in np.flatnonzero(legal):
        action, r, c = int(action), *divmod(int(action), SIZE)
        for dr, dc in DIRECTIONS:
            runs = []
            for sign in (-1, 1):
                extent = 0
                while True:
                    rr, cc = r + sign * (extent + 1) * dr, c + sign * (extent + 1) * dc
                    if not (0 <= rr < SIZE and 0 <= cc < SIZE) or board[rr * SIZE + cc] != color:
                        break
                    extent += 1
                runs.append(extent)
            if sum(runs) + 1 != 4:
                continue
            ends = []
            for sign, extent in zip((-1, 1), runs):
                rr, cc = r + sign * (extent + 1) * dr, c + sign * (extent + 1) * dc
                if not (0 <= rr < SIZE and 0 <= cc < SIZE) or board[rr * SIZE + cc] != 0:
                    break
                ends.append(rr * SIZE + cc)
            if len(ends) == 2:
                threats[action] = True
                break
    return threats


def tacticals(game):
    """Legal placements that block every one of the opponent's immediate wins.

    Usually a single point; two or more only when the opponent has a double
    four, in which case the position is already lost and the set is what is
    left to try.
    """
    legal = game.legal()
    return immediate_wins(game, -game.player) & legal


def tactical_candidates(game):
    """Rapfi-inspired, symmetry-equivariant candidate set for one node."""
    legal = game.legal()
    own = immediate_wins(game)
    if own.any():
        return CandidateSet(own & legal, "forced_win")

    # Defence outranks our own four. A forcing four does not stop an immediate
    # five: the opponent simply blocks and then wins, so offering our four here
    # would let search pick a move that loses on the spot.
    defences = tacticals(game)
    if defences.any():
        return CandidateSet(defences, "forced_defense")

    # Now a four of our own is safe to play: the opponent must answer it, so it
    # costs nothing and may win outright.
    forcing = four_threats(game)
    if forcing.any():
        return CandidateSet(forcing, "strategic")

    occupied = np.flatnonzero(game.board)
    if not len(occupied):
        mask = np.zeros(SIZE * SIZE, dtype=bool)
        mask[(SIZE * SIZE) // 2] = True
        return CandidateSet(mask & legal, "empty_center")

    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for point in occupied:
        r, c = divmod(int(point), SIZE)
        mask[max(0, r - 3):min(SIZE, r + 4), max(0, c - 3):min(SIZE, c + 4)] = True
        for dr, dc in DIRECTIONS:
            for distance in range(-4, 5):
                rr, cc = r + distance * dr, c + distance * dc
                if 0 <= rr < SIZE and 0 <= cc < SIZE:
                    mask[rr, cc] = True
    result = mask.reshape(-1) & legal
    if not result.any():
        result = legal.copy()
        mode = "fallback_all"
    else:
        mode = "square3_line4"
    return CandidateSet(result, mode)
