"""Balanced opening sampler shared by self-play, arena and engine evaluation.

A game that starts from the empty board is not a fair test of either colour:
freestyle Black keeps a decisive first-move advantage, so self-play built from
the empty board degenerates into "Black always wins" and the value head never
sees a contest. Sampling a short balanced opening before play begins removes
that systematic skew while keeping both sides on identical ground.

The sampler is deliberately deterministic: the same ``(rule, seed, plies,
radius)`` arguments always yield the same position, so a training run stays
reproducible and an arena pair can give both games the same start.
"""
import numpy as np

from .game import Game, SIZE

# The first moves are drawn from the whole legal set so openings vary; the tail
# is restricted to the neighbourhood of the stones already on the board so the
# sampler cannot strand a stone in an empty corner.
RANDOM_PLIES = 4
NEIGHBOURHOOD = 3


def opening_moves(game):
    """The board's history as a list of points, oldest first."""
    return [int(action) for action in game.history]


def balanced_opening(rule, seed, plies=8, radius=NEIGHBOURHOOD, random_plies=RANDOM_PLIES):
    """Play ``plies`` opening moves and return the resulting game.

    ``game.legal()`` already encodes each rule's first-move constraint (Renju
    forces the centre opening), so the sampler adds no rule knowledge of its
    own. A game that ends early is returned as it stands rather than padded.
    """
    game = Game(rule)
    rng = np.random.default_rng(int(seed))
    for ply in range(int(plies)):
        if game.adjudicate() is not None:
            break
        legal = game.legal()
        mask = legal
        if ply >= random_plies:
            occupied = np.flatnonzero(game.board)
            if len(occupied):
                window = np.zeros((SIZE, SIZE), bool)
                for point in occupied:
                    r, c = divmod(int(point), SIZE)
                    window[max(0, r - radius):min(SIZE, r + radius + 1),
                           max(0, c - radius):min(SIZE, c + radius + 1)] = True
                local = window.reshape(-1) & legal
                if local.any():
                    mask = local
        actions = np.flatnonzero(mask)
        if not len(actions):
            break
        game.move(int(rng.choice(actions)), validate=False)
    return game
