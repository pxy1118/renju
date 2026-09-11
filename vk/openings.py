"""Opening procedures shared by self-play, arena and engine evaluation.

A game that starts from the empty board is not a fair test of either colour:
freestyle Black keeps a decisive first-move advantage, so self-play built from
the empty board degenerates into "Black always wins" and the value head never
sees a contest. Sampling a short opening before play begins is meant to remove
that systematic skew while keeping both sides on identical ground.

It does not, by itself: drawing stones at random does not blunt a first-player
win, which is why the teacher data is 93% Black wins. ``balanced_opening`` is
the cheap uniform sampler; ``book_opening`` reads a set of openings that were
*verified* balanced by ``artifacts/opening_book_generate.py``, and is what makes
a colour-split result mean something.

Both samplers are deliberately deterministic: the same arguments always yield
the same position, so a training run stays reproducible and an arena pair can
give both games the same start.
"""
from pathlib import Path
import json
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


def load_opening_book(path, rule=None):
    """Read and validate an opening book; returns the parsed document."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Opening book not found: {path}")
    book = json.loads(path.read_text(encoding="utf-8"))
    if book.get("format") != "renju-opening-book":
        raise ValueError(f"Not an opening book: {path}")
    if not book.get("openings"):
        raise ValueError(f"Opening book is empty: {path}")
    if rule is not None and book.get("rule") != rule:
        raise ValueError(f"Opening book is for {book.get('rule')!r}, not {rule!r}")
    for index, entry in enumerate(book["openings"]):
        moves = entry.get("moves")
        if not moves or any(not isinstance(action, int) or not 0 <= action < SIZE * SIZE
                            for action in moves):
            raise ValueError(f"Opening {index} has an invalid move list: {moves!r}")
    return book


def book_opening(book, index):
    """One opening from a loaded book, replayed onto a fresh game.

    ``index`` wraps, so a caller can walk a book cyclically with a seed stream.
    The returned position carries its ``history``, which is what lets
    ``MCTS.advance`` keep reusing the same subtree as in a sampled opening.
    """
    entries = book["openings"]
    entry = entries[int(index) % len(entries)]
    game = Game(book["rule"])
    for action in entry["moves"]:
        game.move(int(action))
    return game


def sampled_opening(rule, seed, plies=8, sample_plies=12, client=None):
    """The teacher's own opening procedure: follow Rapfi's MultiPV for a while.

    ``vk.teacher._play_teacher_game`` draws the first move at random and then
    samples from Rapfi's top-5 until ``sample_plies``, which is why its value
    labels are less saturated than the ones this module's uniform sampler
    produces. Evaluating against openings from that same distribution is a
    separate arm, not the default: a model trained on teacher games and
    evaluated on teacher openings would not be comparable with runs that used
    ``balanced_opening``.
    """
    if client is None:
        raise ValueError("sampled_opening needs a Rapfi client")
    if rule != "freestyle":
        # Renju's first move is forced and the teacher generator draws its
        # openings another way; do not silently pretend the procedures match.
        raise ValueError("The teacher opening procedure is only defined for freestyle")
    game = Game(rule)
    rng = np.random.default_rng(int(seed))
    game.move(int(rng.choice(np.flatnonzero(game.legal()))))
    reply = np.flatnonzero(game.legal())
    if len(reply) == 1:
        game.move(int(reply[0]))
    for ply in range(int(plies)):
        if game.adjudicate() is not None:
            break
        legal = np.flatnonzero(game.legal())
        if not len(legal):
            break
        action = int(rng.choice(legal))
        if ply < sample_plies:
            moves = [move for move in client.analyze(game, 5).moves if move.action in set(legal)]
            weights = np.array([move.winrate + 1e-9 for move in moves], np.float64)
            if len(moves):
                weights /= weights.sum()
                action = int(rng.choice([move.action for move in moves], p=weights))
        game.move(action, validate=False)
    return game


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
