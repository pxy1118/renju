"""The DAgger collector must emit standard format-3 data plus a sidecar.

The engine is faked at the ``analyze`` boundary: the collector's job is to turn
a policy into annotated positions and diagnostics, and running a real engine
here would test Rapfi rather than this code.
"""
from pathlib import Path
import csv
import json

import numpy as np
import torch

from artifacts import dagger_collect
from vk.datasets import (FORMAT_VERSION, combine_datasets, load_split, topk_valid)
from vk.game import Game
from vk.network import Network
from vk.pretraining import pretrain
from vk.rapfi import RapfiAnalysis, RapfiMove
from vk.storage import atomic_save


class FakeRapfi:
    """Answers the first legal moves, with winrates that make one move best."""

    version = "dagger-fake"

    def __init__(self, winrates=(0.9, 0.8, 0.7, 0.6, 0.5)):
        self.winrates = winrates
        self.calls = []

    def analyze(self, game, multipv=5):
        self.calls.append((np.count_nonzero(game.board), multipv))
        legal = [int(action) for action in np.flatnonzero(game.legal())]
        moves = [RapfiMove(action, self.winrates[index % len(self.winrates)], 100, 3, (action,))
                 for index, action in enumerate(legal[:multipv])]
        return RapfiAnalysis(tuple(moves), self.version)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def make_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    model = Network("hybrid-8-1")
    atomic_save({"format": 1, "config": {"rule": "freestyle", "arch": "hybrid-8-1",
                                         "channels": 8, "blocks": 1},
                 "model": model.state_dict(), "step": 0}, path)


def fill_missing_splits(root):
    """Copy the train shard into validation and test so a small set has all three.

    ``pretrain`` requires every split to exist; a handful of DAgger games all
    map to train, so the collected data alone cannot satisfy it.
    """
    source = next((Path(root) / "train").glob("shard-*.npz"))
    with np.load(source, allow_pickle=False) as shard:
        arrays = {key: shard[key] for key in shard.files}
    for split, game_id in (("validation", 8), ("test", 9)):
        (Path(root) / split).mkdir(parents=True, exist_ok=True)
        arrays["game_id"] = np.full(len(arrays["state"]), game_id, np.uint32)
        np.savez_compressed(Path(root) / split / "shard-00000.npz", **arrays)


def test_dagger_output_is_standard_v3_with_a_diagnostics_sidecar(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    make_checkpoint(checkpoint)
    fake = FakeRapfi()
    monkeypatch.setattr(dagger_collect, "RapfiClient", lambda *args, **kwargs: fake)

    class Args:
        pass
    args = Args()
    for key, value in dict(positions=6, games=8, seed=7, rule="freestyle", search="policy",
                           hard_rules="forced", search_bias="none", simulations=4, cpuct=2.0,
               opening_plies=4,
                           annotation_multipv=5, max_nodes=1000, threads=1, hash_mb=8,
                           timeout=5.0, regret_threshold=0.05, shard_size=4, device="cpu",
                           engine="engine.py", engine_dir=str(tmp_path),
                           checkpoint=str(checkpoint),
                           output=str(tmp_path / "dagger")).items():
        setattr(args, key, value)
    manifest = dagger_collect.collect(args)

    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["positions_written"] == 6 and manifest["games"] >= 1
    assert manifest["player"] == "model"
    rows = load_split(tmp_path / "dagger", "train")
    assert len(rows["state"]) == 6
    # Every row carries real top-k winrates: the annotation is the whole point
    # of DAgger data, so sentinel rows here would be a silent regression.
    assert topk_valid(rows["teacher_topk_actions"]).all()
    assert not np.isnan(rows["teacher_topk_winrates"]).all()
    assert (rows["teacher_topk_winrates"] <= 1).all()
    # The model plays its own search, so the recorded positions are not all
    # Rapfi's own line; still, the value target is the annotation's top-1.
    state = rows["state"][0]
    game = Game("freestyle", np.where(state[0], 1, np.where(state[1], -1, 0)).astype(np.int8), 1)
    assert np.count_nonzero(game.board) >= 4

    with (tmp_path / "dagger" / "diagnostics.csv").open(encoding="utf-8") as source:
        diagnostics = list(csv.DictReader(source))
    assert len(diagnostics) == 6
    assert {"sample_id", "model_action", "teacher_best", "regret", "hard"} <= set(diagnostics[0])
    assert manifest["diagnostics"]["outside_topk"] >= 0
    assert all(row["hard"] in ("0", "1") for row in diagnostics)


def test_dagger_data_merges_and_trains_alongside_the_teacher_set(tmp_path, monkeypatch):
    """The merge path is what makes the collected data usable, so exercise it."""
    checkpoint = tmp_path / "best.pt"
    make_checkpoint(checkpoint)
    monkeypatch.setattr(dagger_collect, "RapfiClient", lambda *args, **kwargs: FakeRapfi())

    class Args:
        pass
    args = Args()
    for key, value in dict(positions=6, games=8, seed=3, rule="freestyle", search="policy",
                           hard_rules="forced", search_bias="none", simulations=4, cpuct=2.0,
               opening_plies=4,
                           annotation_multipv=5, max_nodes=1000, threads=1, hash_mb=8,
                           timeout=5.0, regret_threshold=0.05, shard_size=4, device="cpu",
                           engine="engine.py", engine_dir=str(tmp_path),
                           checkpoint=str(checkpoint),
                           output=str(tmp_path / "dagger")).items():
        setattr(args, key, value)
    dagger_collect.collect(args)

    teacher = tmp_path / "teacher"
    rows = 4
    for split in ("train", "validation", "test"):
        (teacher / split).mkdir(parents=True, exist_ok=True)
        state = np.zeros((rows, 3, 15, 15), np.uint8)
        state[:, 2] = 1
        policy = np.zeros((rows, 225), np.float16)
        policy[:, 112] = 1
        actions = np.full((rows, 5), 255, np.uint8)
        actions[:, 0] = 112
        winrates = np.full((rows, 5), np.nan, np.float16)
        winrates[:, 0] = 0.7
        np.savez_compressed(teacher / split / "shard-00000.npz",
                            state=state, policy=policy, value=np.zeros(rows, np.float16),
                            game_id=np.arange(rows, dtype=np.uint32),
                            ply=np.zeros(rows, np.uint16),
                            teacher_best=np.full(rows, 112, np.uint16),
                            teacher_nodes=np.full(rows, 5, np.uint64),
                            teacher_topk_actions=actions, teacher_topk_winrates=winrates)
    (teacher / "manifest.json").write_text(
        json.dumps({"format_version": FORMAT_VERSION, "rule": "freestyle"}), encoding="utf-8")
    merged = tmp_path / "merged"
    manifest = combine_datasets([teacher, tmp_path / "dagger"], merged)
    assert manifest["positions_written"] == 3 * rows + 6
    assert [source["rows"] for source in manifest["sources"]] == [3 * rows, 6]
    fill_missing_splits(merged)

    report = pretrain(merged, tmp_path / "run", steps=4, batch_size=4, arch="hybrid-8-1",
                      device="cpu", warmup_steps=1, learning_rate=0.01,
                      final_learning_rate=0.001, value_weight=0.0)
    assert "value_mae" not in report["acceptance"]
    assert report["gate_mode"] == "policy"
    assert report["test"]["topk_valid_rows"] > 0
