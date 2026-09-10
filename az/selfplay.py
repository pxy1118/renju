"""Spawned CPU actors request inference from one parent GPU process."""
import multiprocessing as mp
import queue
import signal
import time
import traceback
import numpy as np
from .game import Game
from .search import MCTS, SearchStopped


def play_game(rule, evaluator, simulations, seed, stop=lambda: False, cpuct=2.0, temperature_moves=20):
    rng = np.random.default_rng(seed)
    g = Game(rule)
    tree = MCTS(evaluator, simulations, cpuct, rng)
    samples = []
    moves = []
    while g.adjudicate() is None:
        pi = tree.policy(g, noise=True, stop=stop)
        samples.append((g.encode().astype(np.uint8), pi.astype(np.float16), g.player))
        probabilities = pi.astype(np.float64)
        probabilities /= probabilities.sum()
        a = int(rng.choice(225, p=probabilities)) if len(moves) < temperature_moves else int(np.argmax(pi))
        g.move(a, validate=False)
        tree.advance(a, g)
        moves.append(a)
    return [(x, p, float(player*g.winner)) for x, p, player in samples], {
        "winner": g.winner, "moves": moves, "simulations": tree.completed}


def _actor(worker, tasks, requests, response, results, cancel, cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        def evaluate(state):
            requests.put((worker, state))
            while not cancel.is_set():
                try:
                    return response.get(timeout=0.2)
                except queue.Empty:
                    pass
            raise SearchStopped()
        while not cancel.is_set():
            try:
                seed = tasks.get(timeout=0.1)
            except queue.Empty:
                return
            try:
                data, stats = play_game(cfg["rule"], evaluate, cfg["simulations"], seed,
                                        cancel.is_set, cfg["cpuct"], cfg["temperature_moves"])
                results.put(("game", data, stats))
            except SearchStopped:
                return
    except BaseException:
        results.put(("error", traceback.format_exc(), None))


def collect(cfg, evaluator, seeds, deadline, stop=lambda: False):
    ctx = mp.get_context("spawn")
    tasks, requests, results = ctx.Queue(), ctx.Queue(), ctx.Queue()
    cancel = ctx.Event()
    for seed in seeds:
        tasks.put(int(seed))
    replies = [ctx.Queue() for _ in range(min(cfg["workers"], len(seeds)))]
    actors = [ctx.Process(target=_actor, args=(i, tasks, requests, replies[i], results, cancel, cfg))
              for i in range(len(replies))]
    samples, games = [], []
    inference_positions, batches, largest_batch = 0, 0, 0
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
            samples.extend(data)
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
            logits, values = evaluator.batch([state for _, state in batch])
            for (worker, _), p, v in zip(batch, logits, values):
                replies[worker].put((p, float(v)))
            inference_positions += len(batch)
            batches += 1
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
    return samples, games, {"inference_positions": inference_positions, "batches": batches,
                             "average_inference_batch_size": inference_positions/batches if batches else 0,
                             "largest_inference_batch_size": largest_batch,
                             "seconds": time.monotonic()-started}
