"""Placement-only freestyle / Renju. RIF 9.1-9.3, automatic adjudication.

Fours are deduplicated by their stone sets (an open four has two endpoints,
not two fours). Threes require a legal continuation to a straight four.
Recursive legality always adds stones, so recursion is finite.
"""
from functools import lru_cache
import numpy as np

SIZE = 15
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def line(point, direction):
    """Board points through ``point`` along ``direction``, clipped to the board.

    The board is deliberately not a parameter: the geometry does not depend on
    it, and taking one only invites callers to pass their real board while
    silently having it ignored.
    """
    return _line(int(point),direction)


@lru_cache(maxsize=900)
def _line(point, direction):
    r, c = divmod(point, SIZE)
    dr, dc = direction
    return tuple(rr * SIZE + cc for k in range(-14, 15)
                 if 0 <= (rr := r + k * dr) < SIZE and 0 <= (cc := c + k * dc) < SIZE)


def lengths(board, point, color):
    result = []
    r,c = divmod(point,15)
    for dr,dc in DIRECTIONS:
        count = 1
        for sign in (-1,1):
            rr,cc = r+sign*dr,c+sign*dc
            while 0 <= rr < 15 and 0 <= cc < 15 and board[rr*15+cc] == color:
                count += 1
                rr,cc = rr+sign*dr,cc+sign*dc
        result.append(count)
    return result


def fours(board, point, direction):
    cells = line(point, direction)
    groups = {}
    center = cells.index(point)
    for start in range(max(0,center-4),min(center,len(cells)-5)+1):
        segment = cells[start:start+5]
        stones = frozenset(x for x in segment if board[x] == 1)
        empty = [x for x in segment if board[x] == 0]
        if len(stones) != 4 or len(empty) != 1:
            continue
        # The completing five must be exactly five in this direction.
        if start > 0 and board[cells[start-1]] == 1:
            continue
        if start+5 < len(cells) and board[cells[start+5]] == 1:
            continue
        groups.setdefault(stones, set()).add(empty[0])
    return groups


@lru_cache(maxsize=100000)
def forbidden(position: bytes, point: int):
    b = np.frombuffer(position, dtype=np.int8).copy()
    if b[point] != 0:
        return "occupied"
    b[point] = 1
    spans = lengths(b, point, 1)
    if 5 in spans:
        return None  # RIF 9.2: simultaneous five takes precedence.
    if max(spans) > 5:
        return "overline"
    if sum(len(fours(b, point, d)) for d in DIRECTIONS) >= 2:
        return "double_four"
    threes = set()
    for d in DIRECTIONS:
        cells = line(point, d)
        center = cells.index(point)
        nearby = cells[max(0,center-3):center+4]
        if sum(b[x] == 1 for x in nearby) < 3:
            continue
        for q in nearby:
            if b[q] != 0:
                continue
            b[q] = 1
            candidates = [s - {q} for s, ends in fours(b, point, d).items()
                          if q in s and len(ends) == 2]
            # RIF three continuation cannot simultaneously make five.
            if candidates and 5 in lengths(b,q,1):
                candidates = []
            b[q] = 0
            if candidates and forbidden(b.tobytes(), q) is None:
                threes.update(candidates)
                if len(threes) >= 2:
                    return "double_three"
    return None


class Game:
    def __init__(self, rule="freestyle", board=None, player=1, history=None):
        if rule not in ("freestyle", "renju"):
            raise ValueError("Unknown rule")
        self.rule = rule
        self.board = np.zeros(225, np.int8) if board is None else np.array(board, np.int8).reshape(225).copy()
        self.player = player
        self.winner = None
        self.history = list(history) if history is not None else []

    def copy(self):
        g = Game(self.rule, self.board, self.player, self.history)
        g.winner = self.winner
        return g

    def legal(self):
        mask = self.board == 0
        if self.winner is not None:
            return np.zeros(225, bool)
        if self.rule == "renju" and not self.board.any():
            mask[:] = False
            mask[112] = True
        elif self.rule == "renju" and self.player == 1:
            position = self.board.tobytes()
            # A forbidden move needs >= 4 nearby black stones. This safe
            # filter skips expensive recursion for sparse positions.
            for a in np.flatnonzero(mask):
                r, c = divmod(int(a), 15)
                nearby = self.board.reshape(15, 15)[max(0,r-4):r+5, max(0,c-4):c+5]
                if np.count_nonzero(nearby == 1) >= 4:
                    mask[a] = forbidden(position, int(a)) is None
        return mask

    def adjudicate(self):
        if self.winner is None:
            if np.all(self.board != 0):
                self.winner = 0
            elif not self.legal().any():
                self.winner = -self.player
        return self.winner

    def move(self, action, validate=True):
        action = int(action)
        if self.winner is not None or not 0 <= action < 225 or self.board[action] != 0:
            raise ValueError("Illegal move")
        if validate and not self.legal()[action]:
            raise ValueError("Illegal move under selected rule")
        color = self.player
        self.board[action] = color
        self.history.append(action)
        spans = lengths(self.board, action, color)
        if (5 in spans if self.rule == "renju" and color == 1 else max(spans) >= 5):
            self.winner = color
        elif np.all(self.board != 0):
            self.winner = 0
        self.player = -color

    def encode(self):
        b = self.board.reshape(15, 15)
        return np.stack((b == self.player, b == -self.player,
                         np.full((15,15), self.player == 1))).astype(np.float32)
