"""Replay over the position schema, sampled by how surprising a row was.

The buffer is a list of structured-array chunks with a fixed total capacity:
appending a round is a pointer push, trimming drops whole old chunks, and the
flattened view that sampling needs is rebuilt only when the buffer changed.
"""
import numpy as np

from .records import DTYPE, blank, concatenate
from .targets import normalized_weights, surprise_weights


class ReplayBuffer:
    def __init__(self, capacity):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("Replay capacity must be at least one position")
        self.capacity = capacity
        self._chunks = []
        self._count = 0
        self._flat = None

    def __len__(self):
        return self._count

    def clear(self):
        self._chunks, self._count, self._flat = [], 0, None

    def extend(self, records):
        records = np.asarray(records)
        if len(records) == 0:
            return
        if records.dtype != DTYPE:
            raise ValueError("Replay only accepts position records of the current schema")
        self._chunks.append(records)
        self._count += len(records)
        self._flat = None
        self._trim()

    def _trim(self):
        while len(self._chunks) > 1 and self._count - len(self._chunks[0]) >= self.capacity:
            self._count -= len(self._chunks.pop(0))
        if self._count > self.capacity:
            extra = self._count - self.capacity
            self._chunks[0] = self._chunks[0][extra:].copy()
            self._count -= extra

    def array(self):
        """One flat view of every retained row (cached until the buffer changes)."""
        if self._flat is None:
            self._flat = concatenate(list(self._chunks)) if self._chunks else blank(0)
        return self._flat

    def weights(self, cfg):
        data = self.array()
        if not len(data):
            return np.zeros(0, np.float32)
        return surprise_weights(data["policy_surprise"], data["value_surprise"], cfg)

    def sample(self, size, rng, cfg):
        """size rows drawn with probability proportional to the surprise weight."""
        data = self.array()
        if not len(data):
            raise ValueError("Cannot sample from an empty replay buffer")
        weights = normalized_weights(self.weights(cfg).astype(np.float64))
        probabilities = weights / weights.sum()
        indices = rng.choice(len(data), size=int(size), replace=True, p=probabilities)
        return data[indices]

    def stats(self, cfg=None):
        data = self.array()
        report = {"size": int(len(data)), "capacity": self.capacity,
                  "chunks": len(self._chunks)}
        if not len(data):
            return report
        report["full_search_share"] = float((data["full_search"] != 0).mean())
        report["policy_valid_share"] = float((data["policy_valid"] != 0).mean())
        report["sources"] = {str(int(key)): int((data["source"] == key).sum())
                             for key in np.unique(data["source"])}
        if cfg is not None:
            weights = self.weights(cfg)
            report["weight_mean"] = float(weights.mean())
            report["weight_max"] = float(weights.max())
        return report

    def to_state(self):
        return self.array().copy()

    @classmethod
    def from_state(cls, capacity, state):
        buffer = cls(capacity)
        if state is not None and len(state):
            buffer.extend(np.asarray(state))
        return buffer
