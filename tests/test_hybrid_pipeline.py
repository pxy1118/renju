import json
from pathlib import Path
import textwrap
import numpy as np
import torch

from vk.datasets import FORMAT_VERSION, read_shard
from vk.game import Game
from vk.pretraining import pretrain
from vk.records import ACTION_NONE, VALUE_HEADS, SOURCE_TEACHER, head_mask
from vk.teacher import generate_teacher_dataset
from vk.config import DEFAULTS
from vk.storage import load_checkpoint
from vk.training import train

from conftest import collect_perf, game_stats, position_batch


def dynamic_engine(path):
    path.write_text(textwrap.dedent('''\
        for raw in __import__("sys").stdin:
            command = raw.strip()
            if command.startswith("START"):
                print("OK", flush=True)
            elif command == "YXSHOWINFO":
                print("MESSAGE Rapfi dataset-fake", flush=True)
            elif command.startswith("YXBOARD"):
                occupied = {int(item.split(",")[1])*15 + int(item.split(",")[0])
                            for item in command.split()[1:-1]}
            elif command.startswith("YXNBEST"):
                legal = [action for action in range(225) if action not in occupied][:5]
                for index, action in enumerate(legal):
                    x, y = action % 15, action // 15
                    print(f"INFO PV {index}", flush=True); print(f"INFO NUMPV {len(legal)}", flush=True)
                    print("INFO DEPTH 3", flush=True); print(f"INFO NODES {100+index}", flush=True)
                    print(f"INFO WINRATE {0.8-index*0.1}", flush=True)
                    print(f"INFO BESTLINE {x},{y}", flush=True); print("INFO PV DONE", flush=True)
                action = legal[0]; print(f"{action%15},{action//15}", flush=True)
            elif command == "END":
                break
        '''), encoding="utf-8")
    return path


def test_small_teacher_generation_is_safe_and_exact(tmp_path):
    engine = dynamic_engine(tmp_path / "engine.py")
    (tmp_path / "config.toml").write_text("key='value'", encoding="utf-8")
    (tmp_path / "weights.bin").write_bytes(b"fake")
    output = tmp_path / "teacher"
    manifest = generate_teacher_dataset(engine, tmp_path, output, positions=6, workers=1,
                                        threads=1, hash_mb=8, max_nodes=10, timeout=1,
                                        retries=0, shard_size=4)
    assert manifest["positions_written"] == 6
    assert manifest["rapfi_version"] == "dataset-fake"
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["value_heads"] == list(VALUE_HEADS)
    assert {item["path"] for item in manifest["engine_files"]} == {"config.toml", "weights.bin"}
    shards = list(output.rglob("*.npz"))
    assert shards
    with np.load(shards[0], allow_pickle=False) as shard:
        assert shard["state"].dtype == np.uint8 and shard["state"].shape[1:] == (3, 15, 15)
        assert shard["policy"].dtype == np.float16 and shard["policy"].shape[1] == 225
        assert shard["value"].shape[1] == len(VALUE_HEADS)
        assert len(shard["state"]) <= 4
        # The stored per-action winrates must be the engine's raw numbers, not
        # the softmax-of-odds values the policy target is built from.
        assert shard["teacher_topk_winrates"].dtype == np.float16
        assert np.allclose(shard["teacher_topk_winrates"][0].astype(np.float64),
                           [0.8, 0.7, 0.6, 0.5, 0.4], atol=1e-3)
        assert not np.isclose(shard["policy"][0].max(), 0.8)
    # A freshly generated teacher row knows its game result and its horizon
    # targets; only a shard normalised from an old value file has to leave them
    # unset, which tests/test_datasets.py pins separately.
    rows = read_shard(shards[0])
    assert (rows["source"] == SOURCE_TEACHER).all()
    assert head_mask(rows, "final").all() and head_mask(rows, "short").any()
    assert set(int(value) for value in rows["winner"]) <= {1, -1, 0}


def make_dataset(root, rows=64):
    manifest = {"format": "renju-rapfi-teacher-npz", "format_version": 3,
                "rule": "freestyle", "seed": 1}
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = np.zeros((rows, 3, 15, 15), np.uint8)
    state[:, 2] = 1
    policy = np.zeros((rows, 225), np.float16)
    policy[:, 112] = 1
    values = np.zeros(rows, np.float16)
    topk_actions = np.full((rows, 5), 255, np.uint8)
    topk_actions[:, 0] = 112
    topk_winrates = np.full((rows, 5), np.nan, np.float16)
    topk_winrates[:, 0] = 0.6
    for split in ("train", "validation", "test"):
        folder = root / split
        folder.mkdir()
        np.savez_compressed(folder / "shard-00000.npz", state=state, policy=policy,
                            value=values, game_id=np.arange(rows, dtype=np.uint32),
                            ply=np.zeros(rows, np.uint16),
                            teacher_best=np.full(rows, 112, np.uint16),
                            teacher_nodes=np.full(rows, 100, np.uint64),
                            teacher_topk_actions=topk_actions,
                            teacher_topk_winrates=topk_winrates)


def test_64_sample_overfit_save_and_weight_only_init(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    dataset, pretrained = tmp_path / "data", tmp_path / "pretrained"
    make_dataset(dataset)
    report = pretrain(dataset, pretrained, steps=80, batch_size=64, arch="hybrid-8-1",
                      device="cpu", warmup_steps=1,
                      learning_rate=0.01, final_learning_rate=0.001)
    assert report["test"]["top1"] == 1 and (pretrained / "best.pt").is_file()

    import vk.training as module
    sample = position_batch(1, policy=np.eye(1, 225, 112, dtype=np.float32)[0])
    monkeypatch.setattr(module, "collect",
                        lambda *args: (sample, [game_stats()], collect_perf()))
    cfg = dict(DEFAULTS, arch="hybrid-8-1", channels=8, blocks=1,
               simulations=1, workers=1,
               games_per_round=1, train_steps=1, batch_size=1, replay_capacity=10,
               promotion_every=99)
    run = tmp_path / "hybrid"
    result = train(cfg, run, "cpu", 30, max_rounds=1,
                   init_checkpoint=pretrained / "best.pt")
    state = load_checkpoint(result["checkpoint"], "freestyle")
    assert len(state["replay"]) == 1 and state["teacher"] is not None
    assert state["champion_model"] is not None
