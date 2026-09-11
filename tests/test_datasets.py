"""Format-3 shard storage: sentinel rows, merging and split integrity."""
from pathlib import Path
import json
import numpy as np
import pytest

from vk.datasets import (FORMAT_VERSION, LEGACY_ACTION, LEGACY_VERSION, ShardWriter,
                         combine_datasets, load_all, load_split, sentinel_topk, topk_valid)


def record(game_id, ply=1, value=0.5, topk=True, best=112):
    entry = {"state": np.zeros((3, 15, 15), np.uint8),
             "policy": np.full(225, 1 / 225, np.float16),
             "value": np.float16(value), "game_id": np.uint32(game_id),
             "ply": np.uint16(ply), "teacher_best": np.uint16(best),
             "teacher_nodes": np.uint64(7)}
    if topk:
        actions, winrates = sentinel_topk(1)
        actions[0], winrates[0] = np.uint8(best), np.float16(0.9)
        entry["teacher_topk_actions"], entry["teacher_topk_winrates"] = actions[0], winrates[0]
    return entry


def write_v3(root, game_ids, manifest=None):
    writer = ShardWriter(root, shard_size=2)
    for game_id in game_ids:
        writer.add("train" if game_id % 10 < 8 else "test", record(game_id))
    writer.close()
    data = {"format_version": FORMAT_VERSION, "rule": "freestyle", "seed": 1}
    data.update(manifest or {})
    (Path(root) / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def write_v2(root, game_ids):
    """A legacy shard: seven fields, no top-k, written by hand."""
    (Path(root) / "test").mkdir(parents=True, exist_ok=True)
    rows = [record(game_id, topk=False) for game_id in game_ids]
    arrays = {key: np.stack([row[key] for row in rows])
              for key in ("state", "policy", "value", "game_id", "ply",
                          "teacher_best", "teacher_nodes")}
    np.savez_compressed(Path(root) / "test" / "shard-00000.npz", **arrays)
    (Path(root) / "manifest.json").write_text(
        json.dumps({"format_version": LEGACY_VERSION, "rule": "freestyle"}), encoding="utf-8")


def test_v3_roundtrip_keeps_every_field(tmp_path):
    write_v3(tmp_path / "data", range(6))
    loaded = load_split(tmp_path / "data", "train")
    assert loaded["state"].shape == (6, 3, 15, 15)
    assert loaded["policy"].shape == (6, 225)
    assert loaded["teacher_topk_actions"].dtype == np.uint8
    assert loaded["teacher_topk_winrates"].dtype == np.float16
    assert topk_valid(loaded["teacher_topk_actions"]).all()
    assert loaded["teacher_topk_winrates"][0][0] == np.float16(0.9)


def test_v2_rows_load_as_explicit_sentinels(tmp_path):
    """A format-2 source has no per-action winrate; rows say so, not ``None``."""
    write_v2(tmp_path / "legacy", [1, 2, 3])
    loaded = load_split(tmp_path / "legacy", "test")
    assert not topk_valid(loaded["teacher_topk_actions"]).any()
    assert (loaded["teacher_topk_actions"] == LEGACY_ACTION).all()
    assert np.isnan(loaded["teacher_topk_winrates"]).all()


def test_writer_refuses_records_missing_its_version_fields(tmp_path):
    writer = ShardWriter(tmp_path / "data", shard_size=1)
    with pytest.raises(ValueError, match="requires fields"):
        writer.add("train", record(1, topk=False))
        writer.flush("train")


def test_combine_merges_both_versions_and_retags_games(tmp_path):
    write_v3(tmp_path / "v3", [1, 2, 3, 4])
    write_v2(tmp_path / "v2", [1, 2])
    manifest = combine_datasets([tmp_path / "v3", tmp_path / "v2"], tmp_path / "merged")
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["positions_written"] == 6
    assert [source["rows"] for source in manifest["sources"]] == [4, 2]
    assert [source["game_id_offset"] for source in manifest["sources"]] == [0, 10 ** 7]

    merged = load_all(tmp_path / "merged")
    ids = sorted(int(value) for value in merged["game_id"])
    assert ids == [1, 2, 3, 4, 10 ** 7 + 1, 10 ** 7 + 2]
    valid = topk_valid(merged["teacher_topk_actions"])
    assert int(valid.any(axis=1).sum()) == 4, "only the format-3 source carries real winrates"
    assert not valid[4:].any()


def test_combine_is_idempotent_and_keeps_games_in_one_split(tmp_path):
    write_v3(tmp_path / "v3", [1, 2, 3, 4, 9])
    write_v2(tmp_path / "v2", [1, 2, 8])
    first = combine_datasets([tmp_path / "v3", tmp_path / "v2"], tmp_path / "merged")
    second = combine_datasets([tmp_path / "merged"], tmp_path / "again")
    assert second["counts"] == first["counts"]

    # Every game id must live in exactly one split, or a merged dataset would
    # leak positions of one game between train and test.
    seen = {}
    for split in ("train", "validation", "test"):
        for game_id in load_split(tmp_path / "merged", split)["game_id"]:
            seen.setdefault(int(game_id), set()).add(split)
    assert all(len(splits) == 1 for splits in seen.values())
    assert len(seen) == 8
    assert seen[9] == {"test"} and seen[10 ** 7 + 8] == {"validation"}


def test_combine_rejects_a_non_empty_output(tmp_path):
    write_v3(tmp_path / "v3", [1])
    destination = tmp_path / "merged"
    destination.mkdir()
    (destination / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        combine_datasets([tmp_path / "v3"], destination)
