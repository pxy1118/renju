import math
import multiprocessing as mp
import queue
import signal
import time
import traceback
import numpy as np
from .game import Game, lengths
from .network import Network, Evaluator, architecture_of
from .openings import balanced_opening
from .search import MCTS, SearchStopped
from .training import load_checkpoint
from .candidates import immediate_wins, tactical_candidates


def wilson_interval(score, games, z=1.959963984540054):
    if games <= 0:
        return 0.0, 1.0
    denominator = 1 + z * z / games
    center = (score + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


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
    wilson = wilson_interval(mean, len(pair_scores))
    radius = math.sqrt(math.log(40)/(2*len(pair_scores))) if pair_scores else 1
    winners = [record.get("winner", record["result"] * record["color"]) for record in results]
    window_rates = []
    for start in range(max(0, len(winners) - 127)):
        window = winners[start:start + 128]
        if len(window) == 128:
            window_rates.append(max(window.count(1), window.count(-1)) / 128)
    collapse_rate = max(window_rates) if window_rates else None
    return {"games": n, "wins": wins, "draws": draws, "losses": losses, "score": score,
            "paired_score_ci95": [max(0,mean-radius), min(1,mean+radius)],
            "score_wilson_ci95": list(wilson), "wilson_lower": wilson[0],
            "ci_method": "Hoeffding and score-Wilson intervals on independent seeded opening pairs",
            "max_same_board_color_win_rate_128": collapse_rate,
            "color_collapse_detected": collapse_rate > 0.85 if collapse_rate is not None else None,
            "complete": n == expected,
            "by_color": {str(c): {"wins": sum(r["result"] == 1 for r in results if r["color"] == c),
                                   "draws": sum(r["result"] == 0 for r in results if r["color"] == c),
                                   "losses": sum(r["result"] == -1 for r in results if r["color"] == c)} for c in (1,-1)}}


def _arena_actor(worker, tasks, requests, response, results, cancel, cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        def evaluate(model_id, state):
            requests.put((worker, model_id, state))
            while not cancel.is_set():
                try:
                    return response.get(timeout=0.2)
                except queue.Empty:
                    pass
            raise SearchStopped()

        while not cancel.is_set():
            task = tasks.get()
            if task is None:
                return
            pair, color, opening = task
            game = opening.copy()
            ours = MCTS(lambda state: evaluate(0, state), cfg["simulations"], cfg["cpuct"])
            theirs = MCTS(lambda state: evaluate(1, state), cfg["simulations"], cfg["cpuct"])
            try:
                while game.adjudicate() is None:
                    tree = ours if game.player == color else theirs
                    action = int(np.argmax(tree.policy(game, stop=cancel.is_set)))
                    game.move(action, validate=False)
                    ours.advance(action, game)
                    theirs.advance(action, game)
            except SearchStopped:
                return
            results.put(("game", pair, {"result": game.winner * color,
                                         "color": color, "winner": game.winner}))
    except BaseException:
        results.put(("error", worker, traceback.format_exc()))


def _batched_match(model, cfg, device, opponent, pairs, deadline, stop, sequential):
    """Run neural-vs-neural arena games concurrently with central batched inference."""
    ctx = mp.get_context("spawn")
    tasks, requests, results = ctx.Queue(), ctx.Queue(), ctx.Queue()
    cancel = ctx.Event()
    worker_count = min(cfg.get("workers", 1), pairs * 2)
    replies = [ctx.Queue() for _ in range(worker_count)]
    for pair in range(pairs):
        # One shared, balanced opening per pair: both colours face the same
        # position, which is what makes the paired comparison meaningful.
        opening = balanced_opening(cfg["rule"], 91823 + pair, cfg.get("opening_plies", 8))
        for color in (1, -1):
            tasks.put((pair, color, opening))
    for _ in replies:
        tasks.put(None)
    actors = [ctx.Process(target=_arena_actor,
                          args=(i, tasks, requests, replies[i], results, cancel, cfg))
              for i in range(worker_count)]
    evaluators = (Evaluator(model, device), Evaluator(opponent, device))
    records = {}
    reported = 0
    cancelled_at = None

    def completed():
        complete = []
        for pair in sorted(records):
            if len(records[pair]) == 2:
                complete.extend(sorted(records[pair], key=lambda item: -item["color"]))
        return complete

    def drain():
        nonlocal reported
        while True:
            try:
                kind, key, payload = results.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                raise RuntimeError(payload)
            records.setdefault(key, []).append(payload)
        done = completed()
        if len(done) != reported:
            reported = len(done)
            print(f"evaluation opponent=historical games={reported}/{pairs * 2}", flush=True)
        return done

    try:
        for actor in actors:
            actor.start()
        while any(actor.is_alive() for actor in actors):
            done = drain()
            if sequential and done:
                interim = summary(done, pairs * 2)
                if interim["wilson_lower"] > 0.5 or interim["score_wilson_ci95"][1] <= 0.5:
                    interim["sequential_decision"] = ("accept" if interim["wilson_lower"] > 0.5
                                                       else "reject")
                    cancel.set()
            if stop() or time.monotonic() >= deadline:
                cancel.set()
            if cancel.is_set():
                if cancelled_at is None:
                    cancelled_at = time.monotonic()
                if time.monotonic() - cancelled_at > 5:
                    break
            try:
                batch = [requests.get(timeout=0.01)]
            except queue.Empty:
                continue
            until = time.monotonic() + 0.003
            while len(batch) < worker_count and time.monotonic() < until:
                try:
                    batch.append(requests.get(timeout=max(0.0001, until - time.monotonic())))
                except queue.Empty:
                    break
            if cancel.is_set():
                continue
            for model_id in (0, 1):
                selected = [item for item in batch if item[1] == model_id]
                if not selected:
                    continue
                logits, values = evaluators[model_id].batch([item[2] for item in selected])
                for (worker, _, _), policy, value in zip(selected, logits, values):
                    replies[worker].put((policy, float(value)))
        done = drain()
    finally:
        cancel.set()
        for actor in actors:
            if actor.pid is not None:
                actor.join(timeout=3)
                if actor.is_alive():
                    actor.terminate()
                    actor.join()
        for channel in (tasks, requests, results, *replies):
            channel.cancel_join_thread()
            channel.close()
    report = summary(done, pairs * 2)
    if sequential:
        if report["wilson_lower"] > 0.5:
            report["sequential_decision"] = "accept"
        elif report["score_wilson_ci95"][1] <= 0.5:
            report["sequential_decision"] = "reject"
    return report


def match(model, cfg, device, opponent, pairs, deadline=float("inf"), stop=lambda: False,
          sequential=False):
    if not isinstance(opponent, str) and cfg.get("workers", 1) > 1:
        return _batched_match(model, cfg, device, opponent, pairs, deadline, stop, sequential)
    results = []
    stopped = lambda: stop() or time.monotonic() >= deadline
    for pair in range(pairs):
        opening = balanced_opening(cfg["rule"], 91823 + pair, cfg.get("opening_plies", 8))
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
            results.append({"result": g.winner*color, "color": color, "winner": g.winner})
            print(f"evaluation opponent={opponent if isinstance(opponent,str) else 'historical'} games={len(results)}/{pairs*2}", flush=True)
        if sequential:
            interim = summary(results, pairs * 2)
            if interim["wilson_lower"] > 0.5:
                interim["sequential_decision"] = "accept"
                return interim
            if interim["score_wilson_ci95"][1] <= 0.5:
                interim["sequential_decision"] = "reject"
                return interim
    return summary(results, pairs*2)


def tactical_gate(model, cfg, device):
    positions = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        game = Game(cfg["rule"])
        r, c = (7, 5) if dc >= 0 else (7, 9)
        for i in range(4):
            game.board[(r + i * dr) * 15 + c + i * dc] = 1
        positions.append((game, immediate_wins(game)))
    gap = Game(cfg["rule"])
    gap.board[[105, 106, 108, 109]] = 1
    positions.append((gap, immediate_wins(gap)))
    defense = Game(cfg["rule"])
    defense.board[105:109] = -1
    positions.append((defense, tactical_candidates(defense).mask))
    correct = 0
    evaluator = Evaluator(model, device)
    for game, accepted in positions:
        action = int(np.argmax(MCTS(evaluator, max(1, min(8, cfg["simulations"])), cfg["cpuct"]).policy(game)))
        correct += int(accepted[action])
    return {"correct": correct, "total": len(positions), "accuracy": correct / len(positions),
            "passed": correct == len(positions)}


def evaluate_suite(model, cfg, device, root, pairs, deadline=float("inf"), stop=lambda: False):
    from pathlib import Path
    report = {}
    best = Path(root) / "best.pt"
    opponents = []
    if best.exists():
        state = load_checkpoint(best, cfg["rule"])
        other = Network(architecture_of(state["config"])).to(device)
        other.load_state_dict(state["model"])
        opponents.append(("historical", other))
    opponents.extend([("random", "random"), ("tactical", "tactical")])
    for name, opponent in opponents:
        if stop() or time.monotonic() >= deadline:
            break
        report[name] = match(model, cfg, device, opponent, pairs, deadline, stop)
    return report


def evaluate_rapfi(model, cfg, device, engine, engine_dir, pairs,
                   deadline=float("inf"), stop=lambda: False,
                   threads=4, hash_mb=256, max_nodes=200_000, timeout=5.0):
    from .rapfi import RapfiClient
    results = []
    stopped = lambda: stop() or time.monotonic() >= deadline
    with RapfiClient(engine, engine_dir, threads, hash_mb, max_nodes, timeout, 2) as rapfi:
        for pair in range(pairs):
            opening = balanced_opening("freestyle", 91823 + pair, cfg.get("opening_plies", 8))
            for color in (1, -1):
                if stopped():
                    return summary(results, pairs * 2)
                game, ours = opening.copy(), MCTS(Evaluator(model, device), cfg["simulations"], cfg["cpuct"])
                while game.adjudicate() is None:
                    if stopped():
                        return summary(results, pairs * 2)
                    action = (int(np.argmax(ours.policy(game, stop=stopped))) if game.player == color
                              else rapfi.analyze(game, 1).moves[0].action)
                    game.move(action)
                    ours.advance(action, game)
                results.append({"result": game.winner * color, "color": color, "winner": game.winner})
                print(f"evaluation opponent=rapfi games={len(results)}/{pairs * 2}", flush=True)
    return summary(results, pairs * 2)
