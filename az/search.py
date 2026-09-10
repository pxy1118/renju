import numpy as np


class SearchStopped(Exception):
    pass


class Node:
    def __init__(self):
        self.p = None
        self.n = np.zeros(225, np.int32)
        self.w = np.zeros(225, np.float32)
        self.children = {}


class MCTS:
    """Each edge Q is measured from its parent state's player perspective."""
    def __init__(self, evaluator, simulations=200, cpuct=2.0, rng=None):
        self.evaluate = evaluator
        self.simulations = simulations
        self.cpuct = cpuct
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = Node()
        self.key = None
        self.completed = 0

    @staticmethod
    def state_key(game):
        return game.rule, game.player, game.board.tobytes(), game.winner

    def expand(self, node, game):
        legal = game.legal()
        if not legal.any():
            game.adjudicate()
            return float(game.winner * game.player)
        logits, value = self.evaluate(game.encode())
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (225,) or not np.isfinite(logits).all() or not np.isfinite(value):
            raise RuntimeError("Invalid network policy/value output")
        p = np.zeros(225, np.float64)
        p[legal] = np.maximum(np.exp(logits[legal] - logits[legal].max()),np.finfo(np.float64).tiny)
        node.p = p / p.sum()
        return float(value)

    def policy(self, game, noise=False, stop=lambda: False):
        if stop():
            raise SearchStopped()
        key = self.state_key(game)
        if self.key != key:
            self.root, self.key = Node(), key
        if game.adjudicate() is not None:
            raise ValueError("Cannot search terminal position")
        if self.root.p is None:
            self.expand(self.root, game)
        prior = self.root.p.copy()
        if noise:
            indices = np.flatnonzero(prior)
            prior[indices] = 0.75 * prior[indices] + 0.25 * self.rng.dirichlet(np.full(len(indices), 0.3))
        for _ in range(self.simulations):
            if stop():
                raise SearchStopped()
            node, state, path = self.root, game.copy(), []
            while node.p is not None and state.winner is None:
                p = prior if node is self.root else node.p
                q = np.divide(node.w, node.n, out=np.zeros(225, np.float32), where=node.n > 0)
                score = q + self.cpuct * p * np.sqrt(1 + node.n.sum()) / (1 + node.n)
                score[p == 0] = -np.inf
                a = int(np.argmax(score))
                path.append((node, a))
                state.move(a, validate=False)
                node = node.children.setdefault(a, Node())
            value = float(state.winner * state.player) if state.winner is not None else self.expand(node, state)
            for parent, a in reversed(path):
                value = -value
                parent.n[a] += 1
                parent.w[a] += value
            self.completed += 1
        visits = self.root.n.astype(np.float64)
        return (visits / visits.sum()).astype(np.float32)

    def advance(self, action, game):
        self.root = self.root.children.get(int(action), Node())
        self.key = self.state_key(game)
