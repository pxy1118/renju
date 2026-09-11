import json
from pathlib import Path
import textwrap
import numpy as np
import torch

from az.game import Game
from az.pretraining import pretrain
from az.teacher import generate_teacher_dataset
from az.training import DEFAULTS, load_checkpoint, train


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
    assert {item["path"] for item in manifest["engine_files"]} == {"config.toml", "weights.bin"}
    shards = list(output.rglob("*.npz"))
    assert shards
    with np.load(shards[0], allow_pickle=False) as shard:
        assert shard["state"].dtype == np.uint8 and shard["state"].shape[1:] == (3, 15, 15)
        assert shard["policy"].dtype == np.float16 and shard["policy"].shape[1] == 225
        assert len(shard["state"]) <= 4


def make_dataset(root, rows=64):
    manifest = {"format": "renju-rapfi-teacher-npz", "format_version": 2,
                "rule": "freestyle", "seed": 1}
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    state = np.zeros((rows, 3, 15, 15), np.uint8)
    state[:, 2] = 1
    policy = np.zeros((rows, 225), np.float16)
    policy[:, 112] = 1
    values = np.zeros(rows, np.float16)
    for split in ("train", "validation", "test"):
        folder = root / split
        folder.mkdir()
        np.savez_compressed(folder / "shard-00000.npz", state=state, policy=policy,
                            value=values, game_id=np.arange(rows, dtype=np.uint32),
                            ply=np.zeros(rows, np.uint16),
                            teacher_best=np.full(rows, 112, np.uint16),
                            teacher_nodes=np.full(rows, 100, np.uint64))


def test_64_sample_overfit_save_and_weight_only_init(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    dataset, pretrained = tmp_path / "data", tmp_path / "pretrained"
    make_dataset(dataset)
    report = pretrain(dataset, pretrained, steps=80, batch_size=64, channels=4,
                      blocks=1, device="cpu", warmup_steps=1,
                      learning_rate=0.01, final_learning_rate=0.001)
    assert report["test"]["top1"] == 1 and (pretrained / "best.pt").is_file()

    import az.training as module
    sample = [(Game().encode(), np.eye(1, 225, 112, dtype=np.float32)[0], 1.0)]
    stats = {"winner": 1, "moves": [], "simulations": 1}
    perf = {"seconds": 1, "batches": 1, "inference_positions": 1,
            "average_inference_batch_size": 1, "largest_inference_batch_size": 1}
    monkeypatch.setattr(module, "collect", lambda *args: (sample, [stats], perf))
    cfg = dict(DEFAULTS, channels=4, blocks=1, simulations=1, workers=1,
               games_per_round=1, train_steps=1, batch_size=1, replay_capacity=10,
               promotion_every=99)
    run = tmp_path / "hybrid"
    result = train(cfg, run, "cpu", 30, max_rounds=1,
                   init_checkpoint=pretrained / "best.pt")
    state = load_checkpoint(result["checkpoint"], "freestyle")
    assert len(state["replay"]) == 1 and state["teacher"] is not None
    assert state["champion_model"] is not None
