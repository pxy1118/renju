import math
import time
import numpy as np
from .game import Game, lengths
from .network import Network, Evaluator
from .search import MCTS, SearchStopped
from .training import load_checkpoint


def tactical(game, rng):
    legal = np.flatnonzero(game.legal())
    for color in (game.player, -game.player):
        for a in legal:
            b = game.board.copy()
            b[a] = color
            spans = lengths(b, int(a), color)
            if (5 in spans if game.rule == "renju" and color == 1 else max(spans) >= 5):
                return int(a)
    # Fixed local heuristic, used only as an evaluation opponent.
    b = game.board.reshape(15,15)
    scores = []
    for a in legal:
        r, c = divmod(int(a),15)
        scores.append(10*np.count_nonzero(b[max(0,r-2):r+3,max(0,c-2):c+3]) - abs(r-7)-abs(c-7))
    best = np.flatnonzero(np.array(scores) == max(scores))
    return int(legal[rng.choice(best)])


def summary(results, expected):
    n = len(results)
    wins = sum(r["result"] == 1 for r in results)
    draws = sum(r["result"] == 0 for r in results)
    losses = n-wins-draws
    score = (wins+0.5*draws)/n if n else 0.0
    # Hoeffding interval for bounded per-game score; report pairs separately
    # below because games sharing an opening are not independent.
    pair_scores = [(results[i]["result"]+results[i+1]["result"]+2)/4
                   for i in range(0, n-1, 2)]
    mean = float(np.mean(pair_scores)) if pair_scores else 0.5
    radius = math.sqrt(math.log(40)/(2*len(pair_scores))) if pair_scores else 1
    return {"games": n, "wins": wins, "draws": draws, "losses": losses, "score": score,
            "paired_score_ci95": [max(0,mean-radius), min(1,mean+radius)],
            "ci_method": "Hoeffding on independent seeded opening pairs; conservative",
            "complete": n == expected,
            "by_color": {str(c): {"wins": sum(r["result"] == 1 for r in results if r["color"] == c),
                                   "draws": sum(r["result"] == 0 for r in results if r["color"] == c),
                                   "losses": sum(r["result"] == -1 for r in results if r["color"] == c)} for c in (1,-1)}}


def match(model, cfg, device, opponent, pairs, deadline=float("inf"), stop=lambda: False):
    results = []
    stopped = lambda: stop() or time.monotonic() >= deadline
    for pair in range(pairs):
        opening_rng = np.random.default_rng(91823+pair)
        opening = Game(cfg["rule"])
        for _ in range(4):
            opening.move(int(opening_rng.choice(np.flatnonzero(opening.legal()))), validate=False)
        for color in (1,-1):
            if stopped():
                return summary(results, pairs*2)
            rng = np.random.default_rng(4141+pair)
            g = opening.copy()
            ours = MCTS(Evaluator(model, device), cfg["simulations"], cfg["cpuct"])
            theirs = MCTS(Evaluator(opponent, device), cfg["simulations"], cfg["cpuct"]) if not isinstance(opponent,str) else None
            try:
                while g.adjudicate() is None:
                    if stopped():
                        raise SearchStopped()
                    if g.player == color:
                        a = int(np.argmax(ours.policy(g, stop=stopped)))
                    elif theirs:
                        a = int(np.argmax(theirs.policy(g, stop=stopped)))
                    elif opponent == "random":
                        a = int(rng.choice(np.flatnonzero(g.legal())))
                    else:
                        a = tactical(g, rng)
                    g.move(a, validate=False)
                    ours.advance(a, g)
                    if theirs:
                        theirs.advance(a, g)
            except SearchStopped:
                return summary(results, pairs*2)
            results.append({"result": g.winner*color, "color": color})
            print(f"evaluation opponent={opponent if isinstance(opponent,str) else 'historical'} games={len(results)}/{pairs*2}", flush=True)
    return summary(results, pairs*2)


def evaluate_suite(model, cfg, device, root, pairs, deadline=float("inf"), stop=lambda: False):
    from pathlib import Path
    report = {}
    best = Path(root) / "best.pt"
    opponents = []
    if best.exists():
        state = load_checkpoint(best, cfg["rule"])
        other = Network(state["config"]["channels"], state["config"]["blocks"]).to(device)
        other.load_state_dict(state["model"])
        opponents.append(("historical", other))
    opponents.extend([("random", "random"), ("tactical", "tactical")])
    for name, opponent in opponents:
        if stop() or time.monotonic() >= deadline:
            break
        report[name] = match(model, cfg, device, opponent, pairs, deadline, stop)
    return report
