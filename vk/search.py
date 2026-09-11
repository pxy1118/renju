import numpy as np
from .candidates import tactical_candidates


class SearchStopped(Exception):
    pass


class Node:
    def __init__(self):
        self.p = None
        self.n = np.zeros(225, np.int32)
        self.w = np.zeros(225, np.float32)
        self.children = {}
        self.candidate_mode = None
        self.candidate_count = 0


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
        self.last_stats = {}

    @staticmethod
    def state_key(game):
        return game.rule, game.player, game.board.tobytes(), game.winner

    def expand(self, node, game):
        candidates = tactical_candidates(game)
        legal = candidates.mask
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
        node.candidate_mode = candidates.mode
        node.candidate_count = int(legal.sum())
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
        max_depth = 0
        values = []
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
            max_depth = max(max_depth, len(path))
            value = float(state.winner * state.player) if state.winner is not None else self.expand(node, state)
            values.append(abs(value))
            for parent, a in reversed(path):
                value = -value
                parent.n[a] += 1
                parent.w[a] += value
            self.completed += 1
        visits = self.root.n.astype(np.float64)
        policy = visits / visits.sum()
        support = (policy > 0) & (self.root.p > 0)
        kl = float(np.sum(policy[support] * np.log(policy[support] / self.root.p[support])))
        self.last_stats = {
            "candidate_count": self.root.candidate_count,
            "candidate_mode": self.root.candidate_mode,
            "forced_win": int(self.root.candidate_mode == "forced_win"),
            "forced_defense": int(self.root.candidate_mode == "forced_defense"),
            "strategic": int(self.root.candidate_mode == "strategic"),
            "max_depth": max_depth,
            "search_prior_kl": kl,
            "value_abs_mean": float(np.mean(values)) if values else 0.0,
        }
        return policy.astype(np.float32)

    def advance(self, action, game):
        self.root = self.root.children.get(int(action), Node())
        self.key = self.state_key(game)
