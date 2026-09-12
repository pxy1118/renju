"""Shard storage: the current schema, and old shards normalised on the way in."""
from pathlib import Path
import json
import numpy as np
import pytest

from vk.datasets import (FORMAT_VERSION, LEGACY_ACTION, LEGACY_VERSION, ShardWriter,
                         combine_datasets, load_all, load_split, sentinel_topk,
                         split_for_game, topk_valid)
from vk.records import (ACTION_NONE, DTYPE, SOURCE_TEACHER, SOURCE_TEACHER_LEGACY,
                        VALUE_HEADS, blank, head_mask, set_head, stack)


def row(game_id, value=0.5, source=SOURCE_TEACHER, best=112, topk=True):
    """One schema-valid teacher row."""
    records = blank(1)
    records["state"][0] = np.zeros((3, 15, 15), np.uint8)
    records["policy"][0] = np.full(225, 1 / 225, np.float16)
    records["policy_valid"][0] = 1
    records["policy_weight"][0] = 1.0
    records["game_id"][0] = np.uint32(game_id)
    records["ply"][0] = np.uint16(1)
    records["winner"][0] = 1
    records["source"][0] = source
    records["simulations"][0] = 7
    records["teacher_best"][0] = np.uint16(best)
    records["teacher_nodes"][0] = np.uint64(7)
    set_head(records, "final", np.array([value], np.float32))
    actions, winrates = sentinel_topk(1)
    if topk:
        actions[0, 0], winrates[0, 0] = np.uint8(best), np.float16(0.9)
    records["teacher_topk_actions"][0] = actions[0]
    records["teacher_topk_winrates"][0] = winrates[0]
    return records[0]


def legacy_arrays(game_ids, version=LEGACY_VERSION):
    """A shard written by the old code: same columns, old field names."""
    rows = [dict(state=np.zeros((3, 15, 15), np.uint8),
                 policy=np.full(225, 1 / 225, np.float16),
                 value=np.float16(0.5), game_id=np.uint32(game_id),
                 ply=np.uint16(1), teacher_best=np.uint16(112),
                 teacher_nodes=np.uint64(7)) for game_id in game_ids]
    if version >= 3:
        for entry in rows:
            actions, winrates = sentinel_topk(1)
            actions[0, 0], winrates[0, 0] = np.uint8(112), np.float16(0.9)
            entry["teacher_topk_actions"], entry["teacher_topk_winrates"] = actions[0], winrates[0]
    return rows


def write_legacy(root, game_ids, version=LEGACY_VERSION, split=None):
    """A shard written by the old generator, which also split by game id."""
    rows = legacy_arrays(game_ids, version)
    fields = [name for name in ("state", "policy", "value", "game_id", "ply",
                               "teacher_best", "teacher_nodes")]
    if version >= 3:
        fields += ["teacher_topk_actions", "teacher_topk_winrates"]
    groups = {}
    for entry in rows:
        name = split or split_for_game(int(entry["game_id"]))
        groups.setdefault(name, []).append(entry)
    for name, group in groups.items():
        (Path(root) / name).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(root) / name / "shard-00000.npz",
                            **{key: np.stack([entry[key] for entry in group]) for key in fields})
    (Path(root) / "manifest.json").write_text(
        json.dumps({"format_version": version, "rule": "freestyle"}), encoding="utf-8")


def batch(game_ids, **overrides):
    """A schema-valid batch of teacher rows."""
    records = blank(len(game_ids))
    for index, game_id in enumerate(game_ids):
        records[index] = row(game_id, **overrides)
    return records


def write_current(root, game_ids, manifest=None):
    writer = ShardWriter(root, shard_size=2)
    for game_id in game_ids:
        writer.add("train" if game_id % 10 < 8 else "test", row(game_id))
    writer.close()
    data = {"format_version": FORMAT_VERSION, "rule": "freestyle", "seed": 1}
    data.update(manifest or {})
    (Path(root) / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def test_the_current_schema_round_trips(tmp_path):
    write_current(tmp_path / "data", range(6))
    loaded = load_split(tmp_path / "data", "train")
    assert loaded.dtype == DTYPE
    assert loaded["state"].shape == (6, 3, 15, 15)
    assert loaded["policy"].shape == (6, 225)
    assert loaded["value"].shape == (6, len(VALUE_HEADS))
    assert loaded["teacher_topk_actions"].dtype == np.uint8
    assert loaded["teacher_topk_winrates"].dtype == np.float16
    assert topk_valid(loaded["teacher_topk_actions"])[:, 0].all()
    assert loaded["teacher_topk_winrates"][0][0] == np.float16(0.9)
    assert head_mask(loaded, "final").all() and not head_mask(loaded, "mid").any()


def test_a_v3_shard_loads_as_the_current_schema(tmp_path):
    write_legacy(tmp_path / "legacy", [1, 2, 3])
    loaded = load_split(tmp_path / "legacy", "train")
    assert loaded.dtype == DTYPE and len(loaded) == 3
    assert (loaded["source"] == SOURCE_TEACHER_LEGACY).all()
    assert head_mask(loaded, "final").all(), "the old value column becomes the final target"
    assert not head_mask(loaded, "mid").any(), "the old files recorded no horizon"
    assert loaded["search_value"][0] == pytest.approx(0.5, abs=1e-3)
    assert (loaded["winner"] == 127).all(), "the old files recorded no result"


def test_a_v2_shard_has_no_topk_winrates(tmp_path):
    write_legacy(tmp_path / "legacy", [1, 2, 3], version=2)
    loaded = load_split(tmp_path / "legacy", "train")
    assert not topk_valid(loaded["teacher_topk_actions"]).any()
    assert (loaded["teacher_topk_actions"] == LEGACY_ACTION).all()
    assert np.isnan(loaded["teacher_topk_winrates"]).all()


def test_the_writer_refuses_a_row_missing_schema_fields(tmp_path):
    writer = ShardWriter(tmp_path / "data", shard_size=1)
    with pytest.raises(ValueError, match="require fields"):
        writer.add("train", {"state": np.zeros((3, 15, 15), np.uint8)})


def test_a_batch_is_split_by_game_id(tmp_path):
    writer = ShardWriter(tmp_path / "data", shard_size=1)
    writer.add_batch(batch((1, 2, 9, 18)))
    writer.close()
    assert writer.counts == {"train": 2, "validation": 1, "test": 1}
    assert len(load_split(tmp_path / "data", "validation")) == 1
    assert int(load_split(tmp_path / "data", "validation")["game_id"][0]) == 18


def test_an_unknown_field_set_is_rejected(tmp_path):
    (tmp_path / "train").mkdir()
    np.savez_compressed(tmp_path / "train" / "shard-00000.npz",
                        state=np.zeros((1, 3, 15, 15), np.uint8))
    with pytest.raises(ValueError, match="Invalid shard fields"):
        load_split(tmp_path, "train")


def test_combine_merges_old_and_new_sources_and_retags_games(tmp_path):
    write_current(tmp_path / "current", [1, 2, 3, 4])
    write_legacy(tmp_path / "legacy", [1, 2], version=2)
    manifest = combine_datasets([tmp_path / "current", tmp_path / "legacy"],
                                tmp_path / "merged")
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["positions_written"] == 6
    assert [source["rows"] for source in manifest["sources"]] == [4, 2]
    assert [source["game_id_offset"] for source in manifest["sources"]] == [0, 10 ** 7]

    merged = load_all(tmp_path / "merged")
    ids = sorted(int(value) for value in merged["game_id"])
    assert ids == [1, 2, 3, 4, 10 ** 7 + 1, 10 ** 7 + 2]
    valid = topk_valid(merged["teacher_topk_actions"])
    assert int(valid.sum()) == 4, "only the current source carries real winrates"
    assert (merged["source"][-2:] == SOURCE_TEACHER_LEGACY).all()


def test_combine_is_idempotent_and_keeps_games_in_one_split(tmp_path):
    write_current(tmp_path / "current", [1, 2, 3, 4, 9])
    write_legacy(tmp_path / "legacy", [1, 2, 8])
    first = combine_datasets([tmp_path / "current", tmp_path / "legacy"], tmp_path / "merged")
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
    write_current(tmp_path / "current", [1])
    destination = tmp_path / "merged"
    destination.mkdir()
    (destination / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        combine_datasets([tmp_path / "current"], destination)
