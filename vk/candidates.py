"""Tactical move handling shared by every local MCTS caller.

Two mechanisms, deliberately separated:

*  A *hard* rule only covers cases where the move is determined: complete a
   five, or block the opponent five. Those may exclude legal points, because
   there is nothing to choose.
*  A *soft* bias covers everything empirical: a forcing four, a neighbourhood
   around the stones, the centre opening. Those add to the prior and never
   remove a legal point from consideration, so a heuristic that is wrong costs
   prior mass instead of making the right move unreachable.
"""
from dataclasses import dataclass
import numpy as np

from .game import DIRECTIONS, SIZE, lengths, forbidden

HARD_RULES = ("forced", "none")
SEARCH_BIAS = ("none", "tactical")


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
    """Return legal placements that immediately win for color."""
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
    """Legal placements after which color has a forced four.

    A forced four is four in a row with both outer ends still empty, so the
    opponent single stone cannot cover both ways to five. An immediate-five
    test cannot see this -- the move creates a four, not a five -- which is why
    a straight three can be lethal while immediate_wins reports nothing.

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
    """Legal placements that block every one of the opponent immediate wins.

    Usually a single point; two or more only when the opponent has a double
    four, in which case the position is already lost and the set is what is
    left to try.
    """
    legal = game.legal()
    return immediate_wins(game, -game.player) & legal


def legal_candidates(game):
    """Every legal placement; no tactical treatment at all."""
    return CandidateSet(game.legal(), "all_legal")


def forced_candidates(game):
    """Only the moves whose value is determined, with no heuristic pruning.

    Three cases: complete a five, block the opponent five, or nothing is
    determined and every legal point is offered.
    """
    own = immediate_wins(game)
    if own.any():
        return CandidateSet(own, "forced_win")
    defences = tacticals(game)
    if defences.any():
        return CandidateSet(defences, "forced_defense")
    return legal_candidates(game)


def hard_candidates(game, mode="forced"):
    """The legal set search may actually consider.

    forced: the deterministic cases above, every legal point otherwise.
    none:   every legal point, always.
    """
    if mode == "forced":
        return forced_candidates(game)
    if mode == "none":
        return legal_candidates(game)
    raise ValueError(f"Unknown hard rule mode: {mode!r} (known: {list(HARD_RULES)})")


def tactical_bias(game, four=2.0, neighbour=0.5, radius=3):
    """Additive logit bias for empirical tactical shape, never an exclusion.

    The bias is what the old candidate mask was trying to express: forcing fours
    and the neighbourhood of the stones deserve more prior mass, but a legal
    point that the heuristic does not like must still be reachable -- that is
    how a heuristic mistake stays recoverable.
    """
    bias = np.zeros(SIZE * SIZE, dtype=np.float64)
    legal = game.legal()
    occupied = np.flatnonzero(game.board)
    if not len(occupied):
        # An empty board has no neighbourhood to bias; the centre keeps the
        # prior mass the old candidate set gave it, without excluding anything.
        bias[SIZE * SIZE // 2] += float(four)
        return bias
    if four > 0:
        threats = four_threats(game)
        if threats.any():
            bias[threats] += float(four)
    if neighbour > 0:
        window = np.zeros((SIZE, SIZE), bool)
        for point in occupied:
            r, c = divmod(int(point), SIZE)
            window[max(0, r - radius):min(SIZE, r + radius + 1),
                   max(0, c - radius):min(SIZE, c + radius + 1)] = True
            for dr, dc in DIRECTIONS:
                for distance in range(-4, 5):
                    rr, cc = r + distance * dr, c + distance * dc
                    if 0 <= rr < SIZE and 0 <= cc < SIZE:
                        window[rr, cc] = True
        bias[window.reshape(-1)] += float(neighbour)
    return bias * legal
