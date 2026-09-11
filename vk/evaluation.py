import math
import multiprocessing as mp
import queue
import signal
import time
import traceback
import numpy as np
from .game import Game, lengths
from .network import Network, Evaluator, architecture_of
from .openings import balanced_opening, book_opening, load_opening_book, sampled_opening
from .search import MCTS, SearchStopped, policy_move
from .training import load_checkpoint
from .candidates import immediate_wins, tactical_candidates


def wilson_interval(score, games, z=1.959963984540054):
    if games <= 0:
        return 0.0, 1.0
    denominator = 1 + z * z / games
    center = (score + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_interval(values, samples=2000, seed=20260910):
    """Percentile bootstrap CI of the mean of ``values`` (list or array)."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def paired_delta(results_a, results_b, samples=2000, seed=20260910):
    """Paired score difference between two arms on identical openings.

    Both arms must have been run on the same ``(pair, color)`` schedule, which
    is what makes this a *paired* comparison: per-pair differences remove the
    opening-to-opening variance that inflates the two separate Wilson intervals.
    """
    scores = {}
    for label, results in (("a", results_a), ("b", results_b)):
        for record in results:
            scores.setdefault((record["pair"], record["color"]), {})[label] = (record["result"] + 1) / 2
    shared = sorted(key for key, value in scores.items() if "a" in value and "b" in value)
    if not shared:
        return None
    deltas = [scores[key]["a"] - scores[key]["b"] for key in shared]
    mean = float(np.mean(deltas))
    return {"pairs": len(shared), "delta_pp": mean * 100,
            "delta_ci95_pp": [value * 100 for value in bootstrap_interval(deltas, samples, seed)],
            "meaning": "positive means arm A scored higher than arm B on the same openings"}


def opening(rule, seed, mode="sampled", plies=8, client=None, sample_plies=12, book=None):
    if mode == "sampled":
        return balanced_opening(rule, seed, plies)
    if mode == "teacher":
        return sampled_opening(rule, seed, plies, sample_plies, client)
    if mode == "book":
        if book is None:
            raise ValueError("opening_mode='book' needs a loaded opening book")
        return book_opening(book, seed)
    if mode == "none":
        return Game(rule)
    raise ValueError(f"Unknown opening mode: {mode!r} "
                     f"(known: ['sampled', 'teacher', 'book', 'none'])")


def load_book(cfg):
    """The book a run's config points at, or ``None`` for the sampled modes.

    Centralised so a missing or mismatched book fails before any game starts,
    rather than in the middle of a match.
    """
    if cfg.get("opening_mode") != "book":
        return None
    path = cfg.get("opening_book")
    if not path:
        raise ValueError("opening_mode='book' requires opening_book to point at a book file")
    return load_opening_book(path, rule=cfg.get("rule"))


def play_move(game, model, evaluator, cfg, search="mcts", candidates=None, tree=None,
              stop=lambda: False):
    """One move by the model: either MCTS+value or the raw policy ranking."""
    candidates = candidates or cfg.get("candidates", "tactical")
    if search == "policy":
        action, _ = policy_move(game, evaluator, candidates)
        return action
    if search != "mcts":
        raise ValueError(f"Unknown search mode: {search!r} (known: ['mcts', 'policy'])")
    if tree is None:
        tree = MCTS(evaluator, cfg["simulations"], cfg["cpuct"], candidates=candidates)
    return int(np.argmax(tree.policy(game, stop=stop)))


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


def search_statistics(results):
    """Aggregate per-move search diagnostics across completed games.

    Keys no move reported (the MCTS numbers under ``search="policy"``) are left
    out rather than published as null, so an absent field means "this arm has no
    such measurement" instead of "the measurement was empty".
    """
    moves = [move for record in results for move in record.get("moves_detail", [])]
    if not moves:
        return {}
    def mean(key):
        values = [move[key] for move in moves if move.get(key) is not None]
        return float(np.mean(values)) if values else None
    modes = {}
    for move in moves:
        modes[move.get("candidate_mode")] = modes.get(move.get("candidate_mode"), 0) + 1
    statistics = {"moves": len(moves), "candidate_mode_hist": modes,
                  "ply_median": float(np.median(
                      [len(record.get("moves", [])) for record in results if record.get("moves")]))}
    for key in ("candidate_count", "value_abs_mean", "search_prior_kl", "root_visited_moves",
                "root_max_visit_share", "root_visit_entropy", "mean_visits_per_visited_move"):
        value = mean(key)
        if value is not None:
            statistics[f"{key}_mean" if key != "candidate_count" else "candidate_count_mean"] = value
    return statistics


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
    report = {"games": n, "wins": wins, "draws": draws, "losses": losses, "score": score,
              "paired_score_ci95": [max(0,mean-radius), min(1,mean+radius)],
              "score_wilson_ci95": list(wilson), "wilson_lower": wilson[0],
              "ci_method": "Hoeffding and score-Wilson intervals on independent seeded opening pairs",
              "max_same_board_color_win_rate_128": collapse_rate,
              "color_collapse_detected": collapse_rate > 0.85 if collapse_rate is not None else None,
              "complete": n == expected,
              "by_color": {str(c): {"wins": sum(r["result"] == 1 for r in results if r["color"] == c),
                                     "draws": sum(r["result"] == 0 for r in results if r["color"] == c),
                                     "losses": sum(r["result"] == -1 for r in results if r["color"] == c)} for c in (1,-1)}}
    statistics = search_statistics(results)
    if statistics:
        report["search_statistics"] = statistics
    return report


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
            pair, color, start = task
            game = start.copy()
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
                                         "color": color, "winner": game.winner, "pair": pair}))
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
        start = balanced_opening(cfg["rule"], 91823 + pair, cfg.get("opening_plies", 8))
        for color in (1, -1):
            tasks.put((pair, color, start))
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
          sequential=False, client=None):
    if not isinstance(opponent, str) and cfg.get("workers", 1) > 1:
        return _batched_match(model, cfg, device, opponent, pairs, deadline, stop, sequential)
    results = []
    stopped = lambda: stop() or time.monotonic() >= deadline
    mode = cfg.get("opening_mode", "sampled")
    book = load_book(cfg)
    for pair in range(pairs):
        start = opening(cfg["rule"], 91823 + pair, mode, cfg.get("opening_plies", 8), client, book=book)
        for color in (1,-1):
            if stopped():
                return summary(results, pairs*2)
            rng = np.random.default_rng(4141+pair)
            g = start.copy()
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
            results.append({"result": g.winner*color, "color": color, "winner": g.winner,
                            "pair": pair, "moves": list(g.history)})
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


def tactical_gate(model, cfg, device, mode="mcts"):
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
        if mode == "policy":
            # A policy-only checkpoint has an untrained value head, so routing
            # the gate through MCTS would test what was deliberately skipped.
            action, _ = policy_move(game, evaluator, "forced")
        else:
            action = int(np.argmax(MCTS(evaluator, max(1, min(8, cfg["simulations"])),
                                        cfg["cpuct"], candidates=cfg.get("candidates", "tactical")
                                        ).policy(game)))
        correct += int(accepted[action])
    return {"correct": correct, "total": len(positions), "accuracy": correct / len(positions),
            "mode": mode, "passed": correct == len(positions)}


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
                   threads=4, hash_mb=256, max_nodes=200_000, timeout=5.0,
                   opening_seed=91823, with_records=False):
    from .rapfi import RapfiClient
    results = []
    search = cfg.get("search", "mcts")
    candidates = cfg.get("candidates", "tactical")
    mode = cfg.get("opening_mode", "sampled")
    book = load_book(cfg)
    stopped = lambda: stop() or time.monotonic() >= deadline

    def finish():
        report = summary(results, pairs * 2)
        report.update({"search_mode": search, "candidates": candidates, "opening_mode": mode,
                       "opening_seed": opening_seed, "max_nodes": max_nodes,
                       "opening_book": str(cfg.get("opening_book")) if book else None,
                       "simulations": cfg["simulations"] if search == "mcts" else 0})
        if with_records:
            # Per-pair (pair, color) results are what `paired_delta` needs to
            # compare two arms on identical openings; callers that only read the
            # aggregate leave them out to keep the report small.
            report["records"] = [{"pair": record["pair"], "color": record["color"],
                                  "result": record["result"], "winner": record["winner"]}
                                 for record in results]
        return report

    with RapfiClient(engine, engine_dir, threads, hash_mb, max_nodes, timeout, 2, rule="freestyle") as rapfi:
        for pair in range(pairs):
            start = opening("freestyle", opening_seed + pair, mode, cfg.get("opening_plies", 8),
                            rapfi, book=book)
            for color in (1, -1):
                if stopped():
                    return finish()
                game = start.copy()
                evaluator = Evaluator(model, device)
                tree = MCTS(evaluator, cfg["simulations"], cfg["cpuct"], candidates=candidates) \
                    if search == "mcts" else None
                moves_detail = []
                while game.adjudicate() is None:
                    if stopped():
                        return finish()
                    if game.player == color:
                        action = play_move(game, model, evaluator, cfg, search, candidates, tree, stopped)
                        if tree is not None:
                            stats = tree.last_stats
                            moves_detail.append({"candidate_count": stats["candidate_count"],
                                                 "candidate_mode": stats["candidate_mode"],
                                                 "value_abs_mean": stats["value_abs_mean"],
                                                 "search_prior_kl": stats["search_prior_kl"],
                                                 "root_visited_moves": stats["root_visited_moves"],
                                                 "root_max_visit_share": stats["root_max_visit_share"],
                                                 "root_visit_entropy": stats["root_visit_entropy"],
                                                 "mean_visits_per_visited_move":
                                                     stats["mean_visits_per_visited_move"]})
                        else:
                            moves_detail.append({"candidate_mode": "policy_only"})
                    else:
                        action = rapfi.analyze(game, 1).moves[0].action
                    game.move(action)
                    if tree is not None:
                        tree.advance(action, game)
                results.append({"result": game.winner * color, "color": color,
                                "winner": game.winner, "pair": pair,
                                "moves": list(game.history), "moves_detail": moves_detail})
                print(f"evaluation opponent=rapfi search={search} candidates={candidates} "
                      f"games={len(results)}/{pairs * 2}", flush=True)
    return finish()
