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


def tactical_candidates(game):
    """Rapfi-inspired, symmetry-equivariant candidate set for one node."""
    legal = game.legal()
    own = immediate_wins(game)
    if own.any():
        return CandidateSet(own & legal, "forced_win")

    threats = immediate_wins(game, -game.player)
    defenses = threats & legal
    if defenses.any():
        return CandidateSet(defenses, "forced_defense")

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
