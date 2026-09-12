"""PUCT search with hard tactical rules, a soft prior bias and PCR budgets.

The search produces a training target, not a move: the raw visit counts, the
pruned and noise-corrected policy target, the root value, and the shape of the
root distribution that says whether the search had anything to choose between.
"""
from dataclasses import dataclass, field
import numpy as np

from .candidates import hard_candidates, tactical_bias
from .targets import policy_target


class SearchStopped(Exception):
    pass


def prior_options(cfg):
    """The candidate and bias switches: what a move selector needs to know."""
    return dict(hard_rules=cfg.get("hard_rules", "forced"),
                bias=cfg.get("search_bias", "tactical"),
                bias_four=cfg.get("search_bias_four", 2.0),
                bias_neighbour=cfg.get("search_bias_neighbour", 0.5))


def search_options(cfg):
    """Every decoupled search switch, read from one configuration."""
    return dict(prior_options(cfg),
                noise_weight=cfg.get("root_noise_weight", 0.25),
                noise_total=cfg.get("root_noise_total_concentration", 10.83),
                prune_prop=cfg.get("policy_target_prune_prop", 0.02),
                prune_min_count=cfg.get("policy_target_prune_min_count", 2),
                correct_noise=cfg.get("policy_noise_correction", True))


class Node:
    __slots__ = ("p", "n", "w", "children", "hard_mode", "candidate_count", "network_value",
                 "bias_moves")

    def __init__(self):
        self.p = None
        self.n = np.zeros(225, np.int32)
        self.w = np.zeros(225, np.float32)
        self.children = {}
        self.hard_mode = None
        self.candidate_count = 0
        self.network_value = None
        self.bias_moves = 0


@dataclass
class SearchResult:
    """Everything one root search produced, in one explicit object."""
    visits: np.ndarray
    policy: np.ndarray
    value: float
    network_value: float
    q_values: np.ndarray
    q_spread: float
    simulations: int
    full_search: bool
    noise: object
    stats: dict = field(default_factory=dict)


class MCTS:
    """Each edge Q is measured from its parent state player perspective."""

    def __init__(self, evaluator, simulations=200, cpuct=2.0, rng=None, hard_rules="forced",
                 bias="tactical", bias_four=2.0, bias_neighbour=0.5, noise_weight=0.25,
                 noise_total=10.83, prune_prop=0.02, prune_min_count=2,
                 correct_noise=True):
        self.evaluate = evaluator
        self.simulations = int(simulations)
        self.cpuct = cpuct
        self.rng = rng if rng is not None else np.random.default_rng()
        if hard_rules not in ("forced", "none"):
            raise ValueError(f"Unknown hard rule mode: {hard_rules!r}")
        if bias not in ("none", "tactical"):
            raise ValueError(f"Unknown search bias: {bias!r}")
        self.hard_rules = hard_rules
        self.bias = bias
        self.bias_four = float(bias_four)
        self.bias_neighbour = float(bias_neighbour)
        self.noise_weight = float(noise_weight)
        self.noise_total = float(noise_total)
        self.prune_prop = float(prune_prop)
        self.prune_min_count = int(prune_min_count)
        self.correct_noise = bool(correct_noise)
        self.root = Node()
        self.key = None
        self.completed = 0
        self.last_stats = {}
        self.last_result = None

    @staticmethod
    def state_key(game):
        return game.rule, game.player, game.board.tobytes(), game.winner

    def prior_bias(self, game):
        """Soft tactical bias for this node; zeros when bias is disabled."""
        if self.bias == "none":
            return np.zeros(225, np.float64)
        return tactical_bias(game, self.bias_four, self.bias_neighbour)

    def expand(self, node, game):
        candidates = hard_candidates(game, self.hard_rules)
        legal = candidates.mask
        if not legal.any():
            game.adjudicate()
            return float(game.winner * game.player)
        inference = self.evaluate(game.encode())
        logits = np.asarray(inference.policy, dtype=np.float64)
        leaf = float(inference.leaf())
        if logits.shape != (225,) or not np.isfinite(logits).all() or not np.isfinite(leaf):
            raise RuntimeError("Invalid network policy/value output")
        bias_vector = self.prior_bias(game)
        node.bias_moves = int((bias_vector > 0).sum())
        biased = logits + bias_vector
        p = np.zeros(225, np.float64)
        p[legal] = np.maximum(np.exp(biased[legal] - biased[legal].max()),
                              np.finfo(np.float64).tiny)
        node.p = p / p.sum()
        node.hard_mode = candidates.mode
        node.candidate_count = int(legal.sum())
        node.network_value = leaf
        return leaf

    def search(self, game, noise=False, budget=None, full_search=True, stop=lambda: False):
        """Run one root search and return its result."""
        if stop():
            raise SearchStopped()
        key = self.state_key(game)
        if self.key != key:
            self.root, self.key = Node(), key
        if game.adjudicate() is not None:
            raise ValueError("Cannot search terminal position")
        budget = self.simulations if budget is None else int(budget)
        if self.root.p is None:
            self.expand(self.root, game)
        prior = self.root.p.copy()
        noise_full = None
        if noise and self.noise_weight > 0:
            indices = np.flatnonzero(prior)
            alpha = self.noise_total / max(1, len(indices))
            draw = self.rng.dirichlet(np.full(len(indices), alpha))
            noise_full = np.zeros(225, np.float64)
            noise_full[indices] = draw
            prior[indices] = (1.0 - self.noise_weight) * prior[indices] + self.noise_weight * draw
        max_depth = 0
        values = []
        for _ in range(max(0, budget)):
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
        result = self._result(game, prior, noise_full, values, max_depth, budget, full_search)
        self.last_result = result
        self.last_stats = result.stats
        return result

    def _result(self, game, prior, noise_full, values, max_depth, budget, full_search):
        visits = self.root.n.astype(np.float64)
        if not visits.any():
            # Zero simulations (or all of them stopped early): there is no visit
            # distribution to report, and dividing by its zero sum would warn.
            root_leaf = 0.0 if self.root.network_value is None else float(self.root.network_value)
            return SearchResult(
                visits=visits, policy=np.zeros(225, np.float32), value=root_leaf,
                network_value=root_leaf, q_values=np.zeros(225, np.float32),
                q_spread=0.0, simulations=0, full_search=bool(full_search), noise=noise_full,
                stats=self._stats(max_depth, visits, 0.0, prior, values))
        target = policy_target(visits, self.root.p, noise_full,
                               self.noise_weight if noise_full is not None else 0.0,
                               self.prune_prop, self.prune_min_count, self.correct_noise)
        q = np.divide(self.root.w, self.root.n, out=np.zeros(225, np.float32), where=self.root.n > 0)
        visited = self.root.n > 0
        spread = float(q[visited].max() - q[visited].min()) if visited.sum() >= 2 else 0.0
        root_value = float(q[visited].mean()) if visited.any() else 0.0
        root_leaf = 0.0 if self.root.network_value is None else float(self.root.network_value)
        return SearchResult(visits=visits, policy=target, value=root_value,
                            network_value=root_leaf, q_values=q,
                            q_spread=spread, simulations=int(visits.sum()),
                            full_search=bool(full_search), noise=noise_full,
                            stats=self._stats(max_depth, visits, spread, prior, values,
                                              root_value, target))

    def _stats(self, max_depth, visits, spread, prior, values, root_value=0.0, target=None):
        share = visits / visits.sum() if visits.sum() > 0 else np.zeros(225)
        support = (share > 0) & (prior > 0)
        kl = float(np.sum(share[support] * np.log(share[support] / prior[support]))) if support.any() else 0.0
        # How far the *supervised* target moved away from what the network
        # already believed: this is the training signal the search added.
        surprise = 0.0
        if target is not None and self.root.p is not None:
            echo = (target > 0) & (self.root.p > 0)
            if echo.any():
                surprise = float(np.sum(target[echo] * np.log(target[echo] / self.root.p[echo])))
        visited = int((visits > 0).sum())
        nonzero = share[share > 0]
        mode = self.root.hard_mode
        return {
            "hard_rules": self.hard_rules, "hard_mode": mode,
            "search_bias": self.bias,
            "candidate_count": self.root.candidate_count,
            "forced_win": int(mode == "forced_win"),
            "forced_defense": int(mode == "forced_defense"),
            "bias_moves": int(getattr(self.root, "bias_moves", 0)),
            "max_depth": max_depth,
            "search_prior_kl": kl,
            "policy_surprise": surprise,
            "value_surprise": float(abs(root_value - (0.0 if self.root.network_value is None
                                                     else float(self.root.network_value)))),
            "value_abs_mean": float(np.mean(values)) if values else 0.0,
            "root_value": float(root_value),
            "network_value": float(0.0 if self.root.network_value is None else self.root.network_value),
            "q_spread": float(spread),
            # How far the search actually spread. Visits per move is not a
            # signal (every simulation adds exactly one root visit), so report
            # the shape of the visit distribution instead.
            "root_visited_moves": visited,
            "root_visited_share": visited / max(1, self.root.candidate_count),
            "root_max_visit_share": float(nonzero.max()) if len(nonzero) else 0.0,
            "root_visit_entropy": float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0,
            "mean_visits_per_visited_move": float(visits.sum() / visited) if visited else 0.0,
        }

    def policy(self, game, noise=False, stop=lambda: False):
        """The pruned training target of one search; the common short call."""
        return self.search(game, noise=noise, stop=stop).policy

    def advance(self, action, game):
        self.root = self.root.children.get(int(action), Node())
        self.key = self.state_key(game)


def policy_move(game, evaluator, hard_rules="forced", bias="tactical",
                bias_four=2.0, bias_neighbour=0.5):
    """Play the network own ranking, with no search and no value head.

    Deterministic tactics still apply: an available five is completed and a
    five for the opponent is blocked. Everything else is the policy prior plus
    the soft tactical bias, restricted to the hard candidate set.
    """
    restricted = hard_candidates(game, hard_rules)
    logits = np.asarray(evaluator(game.encode()).policy, dtype=np.float64)
    if logits.shape != (225,) or not np.isfinite(logits).all():
        raise RuntimeError("Invalid network policy output")
    allowed = restricted.mask.copy()
    if not allowed.any():
        # The position is terminal; the caller only needs *a* legal action and
        # the game ends before it is used.
        allowed = game.legal()
    if not allowed.any():
        raise ValueError("Cannot choose a move: no legal placement")
    scores = logits + (tactical_bias(game, bias_four, bias_neighbour)
                       if bias == "tactical" else 0.0)
    scores = np.where(allowed, scores, -np.inf)
    return int(np.argmax(scores)), restricted
