"""Old checkpoints stay readable, and the upgrade is explicit and tested.

The multi-head value schema changed the state dict. A checkpoint written before
that change has one value head and no horizon targets; these tests pin that it
still loads, that the horizon heads start from the final head, and that a
pre-refactor checkpoint is refused for resume rather than silently misread.
"""
import json
import numpy as np
import pytest
import torch

from vk.config import DEFAULTS, upgrade_format1 as upgrade_config
from vk.network import Network, value_heads
from vk.records import SCHEMA_VERSION, VALUE_HEADS
from vk.storage import (CHECKPOINT_FORMAT, atomic_save, load_checkpoint, load_model_state,
                        save, save_best)


def write_format1(path, arch="hybrid-8-1", rule="freestyle", channels=8, blocks=1):
    """A pre-refactor checkpoint: single value head, no schema field."""
    model = Network(arch)
    state = {key: value for key, value in model.state_dict().items()
             if not key.startswith(("value_mid", "value_short"))}
    atomic_save({"format": 1, "config": {"rule": rule, "arch": arch,
                                         "channels": channels, "blocks": blocks},
                 "model": state, "step": 3}, path)
    return model


def test_a_pre_refactor_checkpoint_gains_the_horizon_heads_from_the_final_head(tmp_path):
    original = write_format1(tmp_path / "old.pt")
    state = load_model_state(tmp_path / "old.pt", "freestyle")
    assert state["format"] == CHECKPOINT_FORMAT and state["schema"] == SCHEMA_VERSION
    assert state["config"]["arch"] == "hybrid-8-1"
    restored = Network("hybrid-8-1")
    restored.load_state_dict(state["model"])
    for head in VALUE_HEADS[1:]:
        assert torch.equal(restored.head_module(head)[0].weight,
                           restored.head_module("final")[0].weight)
    # The final head and the trunk are untouched by the upgrade.
    assert torch.equal(restored.trunk[0].weight, original.trunk[0].weight)


def test_a_pre_refactor_checkpoint_is_refused_for_resume(tmp_path):
    write_format1(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="not resumed"):
        load_checkpoint(tmp_path / "old.pt", "freestyle")


def test_the_legacy_architecture_keeps_its_single_head(tmp_path):
    write_format1(tmp_path / "legacy.pt", arch="legacy-64-6", channels=64, blocks=6)
    state = load_model_state(tmp_path / "legacy.pt", "freestyle")
    assert value_heads(state["config"]["arch"]) == ("final",)
    Network("legacy-64-6").load_state_dict(state["model"])
    assert not [key for key in state["model"] if key.startswith("value_mid")]


def test_a_rule_mismatch_is_still_a_mismatch(tmp_path):
    write_format1(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="mismatch"):
        load_model_state(tmp_path / "old.pt", "renju")


def test_a_current_checkpoint_round_trips_without_an_upgrade(tmp_path):
    cfg = dict(DEFAULTS, arch="hybrid-8-1", channels=8, blocks=1, rule="freestyle")
    model = Network("hybrid-8-1")
    optimizer = torch.optim.Adam(model.parameters())
    rng = np.random.default_rng(0)
    state = {"round": 1, "step": 2, "total_games": 3, "pending_steps": 0,
             "champion_model": None, "champion_optimizer": None, "champion_step": 0,
             "teacher": None}
    save(tmp_path, cfg, model, optimizer, None, rng, 1, 2, 3)
    loaded = load_checkpoint(sorted(tmp_path.glob("checkpoint-*.pt"))[-1], "freestyle")
    assert loaded["format"] == CHECKPOINT_FORMAT and loaded["schema"] == SCHEMA_VERSION
    assert loaded["value_heads"] == list(VALUE_HEADS)
    assert loaded["config"]["hard_rules"] == "forced"
    save_best(tmp_path, cfg, model.state_dict(), 2)
    best = load_checkpoint(tmp_path / "best.pt", "freestyle")
    assert "optimizer" not in best


def test_the_config_upgrade_maps_the_old_candidate_mode(tmp_path):
    upgraded = upgrade_config({"rule": "freestyle", "channels": 64, "blocks": 6,
                              "candidates": "legal", "workers": 4})
    assert upgraded["arch"] == "legacy-64-6"
    assert upgraded["hard_rules"] == "none" and upgraded["search_bias"] == "none"
    assert upgraded["workers"] == 4
    assert set(upgraded) == set(DEFAULTS)
    assert upgrade_config({"candidates": "tactical"})["search_bias"] == "tactical"
    assert upgrade_config({"candidates": "forced"})["search_bias"] == "none"
    with pytest.raises(ValueError, match="candidates"):
        upgrade_config({"candidates": "square3"})
