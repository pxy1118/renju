"""Shared test helpers: a valid position batch without a self-play run."""
import numpy as np

from vk.game import Game
from vk.records import blank, set_head


def position_batch(rows=1, winner=1, policy=None, state=None):
    """A schema-valid batch of finished-position records.

    Tests that monkeypatch the self-play collector still need data the training
    loop and the replay buffer accept, rather than a hand-rolled tuple.
    """
    records = blank(rows)
    for index in range(rows):
        records["state"][index] = (Game().encode() if state is None else state).astype(np.uint8)
        records["policy"][index] = (np.full(225, 1 / 225, np.float16) if policy is None
                                    else policy)
        records["policy_valid"][index] = 1
        records["policy_weight"][index] = 1.0
        records["ply"][index] = index
        records["winner"][index] = winner
        records["full_search"][index] = 1
    set_head(records, "final", np.full(rows, float(winner), np.float32))
    return records


def game_stats(winner=1, moves=()):
    """The per-game statistics the collector returns next to its records."""
    return {"winner": winner, "moves": list(moves), "simulations": 1, "opening": []}


def collect_perf(seconds=1):
    return {"seconds": seconds, "batches": 1, "inference_positions": 1,
            "average_inference_batch_size": 1, "largest_inference_batch_size": 1}
