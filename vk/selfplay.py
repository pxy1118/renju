"""Spawned CPU actors request inference from one parent GPU process.

One game produces one structured batch of positions. Each move decides its own
search budget -- a full search whose policy target is worth supervising, or a
cheap search that only exists to advance the game and produce value data -- and
the value targets are built once the game is over, from the per-move search
values that were collected along the way.
"""
import multiprocessing as mp
import queue
import signal
import time
import traceback
import numpy as np

from .config import value_horizons
from .game import Game
from .network import Inference
from .openings import balanced_opening, book_opening, load_opening_book, opening_moves
from .records import SOURCE_SELFPLAY, blank
from .search import MCTS, SearchStopped, search_options
from .targets import surprise_weights, value_targets


def search_plan(cfg, rng):
    """This move's budget: cheap most of the time, full for a policy target.

    The expensive searches are what make a policy target worth learning from;
    the cheap ones exist so the same compute buys more games, and therefore
    more value data. Returns (budget, full_search).
    """
    full = int(cfg["simulations"])
    if rng.random() >= float(cfg.get("cheap_search_prob", 0.0)):
        return full, True
    # A cheap search is only ever cheaper than the full one: with a small full
    # budget (smoke runs, tiny tests) the two budgets coincide.
    cheap = min(full, max(1, int(cfg.get("cheap_search_simulations", 1))))
    return cheap, False


def play_game(cfg, evaluator, seed, stop=lambda: False, book=None, opening=None,
              game_id=0):
    """Play one self-play game and return its position records plus statistics.

    book takes precedence over opening, and opening over the sampled opener.
    Both are copied, never mutated: a paired arena match hands the same start to
    both colours.
    """
    if book is not None:
        g = book_opening(book, seed)
    elif opening is not None:
        # A real copy: Game.copy duplicates the board and the move list, while a
        # shallow copy would hand the caller's position back mutated.
        g = opening.copy()
    else:
        g = balanced_opening(cfg["rule"], seed, cfg.get("opening_plies", 8))
    rng = np.random.default_rng(seed)
    tree = MCTS(evaluator, cfg["simulations"], cfg.get("cpuct", 2.0), rng,
                **search_options(cfg))
    opening_played = opening_moves(g)
    moves = list(opening_played)
    per_move, search_stats = [], []
    while g.adjudicate() is None:
        budget, full = search_plan(cfg, rng)
        result = tree.search(g, noise=full, budget=budget, full_search=full, stop=stop)
        forced = result.stats["candidate_count"] <= 1
        valid = (full or float(cfg.get("cheap_search_target_weight", 0.0)) > 0) and not forced
        per_move.append({
            "state": g.encode().astype(np.uint8),
            "policy": result.policy if valid else np.zeros(225, np.float32),
            "policy_valid": 1 if valid else 0,
            "policy_weight": 1.0 if full else float(cfg.get("cheap_search_target_weight", 0.0)),
            "search_value": result.value,
            "q_spread": result.q_spread,
            "policy_surprise": result.stats["policy_surprise"],
            "value_surprise": result.stats["value_surprise"],
            # The budget this move was given, not the resulting visit count:
            # subtree reuse carries the parent one visit over, and the record
            # should say how much search the move was actually worth.
            "simulations": budget,
            "full_search": 1 if full else 0,
        })
        search_stats.append(result.stats)
        visits = np.asarray(result.visits, np.float64)
        if visits.sum() <= 0:
            action = int(np.argmax(result.policy if result.policy.any() else g.legal()))
        elif len(moves) < int(cfg.get("temperature_moves", 20)):
            probabilities = visits / visits.sum()
            action = int(rng.choice(225, p=probabilities))
        else:
            action = int(np.argmax(visits))
        g.move(action, validate=False)
        tree.advance(action, g)
        moves.append(action)
    records, horizon = _records(cfg, per_move, search_stats, g, opening_played, moves,
                                int(game_id))
    return records, _game_stats(per_move, search_stats, g, opening_played, moves, tree, horizon)


def _records(cfg, per_move, search_stats, game, opening_played, moves, game_id):
    """The finished game as one explicitly named record batch."""
    plies = len(per_move)
    horizons = value_horizons(cfg)
    search_values = np.array([move["search_value"] for move in per_move], np.float64)
    values, valid, horizon_stats = value_targets(search_values, game.winner,
                                                 horizons["short"], horizons["mid"])
    records = blank(plies)
    for index, move in enumerate(per_move):
        records["state"][index] = move["state"]
        records["policy"][index] = move["policy"]
        records["policy_valid"][index] = move["policy_valid"]
        records["policy_weight"][index] = move["policy_weight"]
        records["search_value"][index] = move["search_value"]
        records["q_spread"][index] = move["q_spread"]
        records["policy_surprise"][index] = move["policy_surprise"]
        records["value_surprise"][index] = move["value_surprise"]
        records["simulations"][index] = move["simulations"]
        records["full_search"][index] = move["full_search"]
        records["game_id"][index] = game_id
        records["ply"][index] = index
        records["winner"][index] = game.winner
        records["source"][index] = SOURCE_SELFPLAY
    records["value"] = values
    records["value_valid"] = valid
    records["weight"] = surprise_weights(records["policy_surprise"],
                                         records["value_surprise"], cfg)
    return records, horizon_stats


def _game_stats(per_move, search_stats, game, opening_played, moves, tree, horizon):
    def mean(key, source=search_stats):
        values = [item[key] for item in source]
        return float(np.mean(values)) if values else 0.0
    full = [move["full_search"] for move in per_move]
    return {
        "winner": game.winner, "moves": moves, "simulations": tree.completed,
        "opening": [int(action) for action in opening_played],
        "candidate_count_mean": mean("candidate_count"),
        "forced_win_count": sum(item["forced_win"] for item in search_stats),
        "forced_defense_count": sum(item["forced_defense"] for item in search_stats),
        "bias_moves_mean": mean("bias_moves"),
        "search_max_depth": max((item["max_depth"] for item in search_stats), default=0),
        "search_prior_kl_mean": mean("search_prior_kl"),
        "q_spread_mean": mean("q_spread"),
        "root_value_mean": mean("root_value"),
        "value_abs_mean": mean("value_abs_mean"),
        "cheap_search_share": float(1.0 - np.mean(full)) if full else 0.0,
        "policy_valid_share": float(np.mean([move["policy_valid"] for move in per_move]))
        if per_move else 0.0,
        "policy_surprise_mean": mean("policy_surprise"),
        "value_surprise_mean": mean("value_surprise"),
        # How often a horizon target had to bootstrap the search value instead
        # of reading the true result: near zero means the horizon is too long
        # for this game length to teach anything the final head does not.
        "horizon_bootstrap_share": float((horizon["short_bootstrap"]
                                          + horizon["mid_bootstrap"])
                                         / max(1, 2 * horizon["rows"])),
    }


def _actor(worker, tasks, requests, response, results, cancel, cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        def evaluate(state):
            requests.put((worker, state))
            while not cancel.is_set():
                try:
                    policy, value = response.get(timeout=0.2)
                    return Inference.leaf_value(policy, value)
                except queue.Empty:
                    pass
            raise SearchStopped()

        # A book is passed by path, never as an object: actors are spawned
        # processes, so a loaded dict would not survive the trip.
        book = None
        if cfg.get("opening_mode") == "book":
            path = cfg.get("opening_book")
            if not path:
                raise ValueError("opening_mode='book' requires an opening_book path")
            book = load_opening_book(path, rule=cfg.get("rule"))
        while not cancel.is_set():
            seed = tasks.get()
            if seed is None:
                return
            try:
                records, stats = play_game(cfg, evaluate, seed, cancel.is_set, book=book,
                                           game_id=int(seed))
                results.put(("game", records, stats))
            except SearchStopped:
                return
    except BaseException:
        results.put(("error", traceback.format_exc(), None))


def collect(cfg, evaluator, seeds, deadline, stop=lambda: False):
    ctx = mp.get_context("spawn")
    tasks, requests, results = ctx.Queue(), ctx.Queue(), ctx.Queue()
    cancel = ctx.Event()
    replies = [ctx.Queue() for _ in range(min(cfg["workers"], len(seeds)))]
    for seed in seeds:
        tasks.put(int(seed))
    # Explicit sentinels avoid a Windows Queue feeder race where actors could
    # observe a transient empty queue and exit before all seeds were visible.
    for _ in replies:
        tasks.put(None)
    actors = [ctx.Process(target=_actor, args=(i, tasks, requests, replies[i], results, cancel, cfg))
              for i in range(len(replies))]
    batches, games = [], []
    inference_positions, batch_count, largest_batch = 0, 0, 0
    started = last_report = time.monotonic()
    cancelled_at = None

    def drain():
        while True:
            try:
                kind, data, stats = results.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                raise RuntimeError(data)
            batches.append(data)
            games.append(stats)

    try:
        for actor in actors:
            actor.start()
        while any(p.is_alive() for p in actors):
            drain()
            if stop() or time.monotonic() >= deadline:
                cancel.set()
                if cancelled_at is None:
                    cancelled_at = time.monotonic()
            if cancelled_at is not None and time.monotonic()-cancelled_at > 5:
                break
            if time.monotonic() - last_report >= 20:
                print(f"selfplay games={len(games)}/{len(seeds)} inferred={inference_positions} "
                      f"elapsed={time.monotonic()-started:.1f}s remaining={max(0,deadline-time.monotonic()):.1f}s", flush=True)
                last_report = time.monotonic()
            try:
                batch = [requests.get(timeout=0.01)]
            except queue.Empty:
                continue
            until = time.monotonic() + 0.003
            while len(batch) < len(replies) and time.monotonic() < until:
                try:
                    batch.append(requests.get(timeout=max(0.0001, until-time.monotonic())))
                except queue.Empty:
                    break
            if cancel.is_set():
                continue
            inference = evaluator.batch([state for _, state in batch])
            for (worker, _), p, v in zip(batch, inference.policy, inference.leaf()):
                replies[worker].put((p, float(v)))
            inference_positions += len(batch)
            batch_count += 1
            largest_batch = max(largest_batch, len(batch))
        drain()
        for actor in actors:
            if actor.exitcode not in (0,None) and not cancel.is_set():
                raise RuntimeError(f"Self-play worker exited with code {actor.exitcode}")
    finally:
        cancel.set()
        for actor in actors:
            if actor.pid is not None:
                actor.join(timeout=3)
                if actor.is_alive():
                    actor.terminate()
                    actor.join()
        for q in [tasks, requests, results, *replies]:
            q.cancel_join_thread()
            q.close()
    records = np.concatenate(batches) if batches else blank(0)
    return records, games, {"inference_positions": inference_positions, "batches": batch_count,
                            "average_inference_batch_size": inference_positions/batch_count if batch_count else 0,
                            "largest_inference_batch_size": largest_batch,
                            "seconds": time.monotonic()-started}
