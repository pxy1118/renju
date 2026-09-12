"""Configuration is one table: parsing, validation and resume compatibility."""
import json
import numpy as np
import pytest

from vk.config import (DEFAULTS, RESUME_FREE, defaults, from_file, mix_vector,
                       resume_incompatible, translate_legacy, validate,
                       value_horizons, value_weights)


def write(path, values):
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_the_default_table_validates_and_covers_every_key():
    assert validate(dict(DEFAULTS)) == DEFAULTS
    assert set(RESUME_FREE) <= set(DEFAULTS)
    assert len(mix_vector(DEFAULTS)) == 3
    assert mix_vector(DEFAULTS).sum() == pytest.approx(1.0)
    assert value_weights(DEFAULTS)[0] == pytest.approx(1.0)
    horizons = value_horizons(DEFAULTS)
    assert horizons["final"] is None and horizons["short"] < horizons["mid"]


@pytest.mark.parametrize("key,value", [
    ("workers", 0), ("workers", 2.5), ("simulations", -1), ("batch_size", 0),
    ("cheap_search_prob", 1.5), ("hard_rules", "tactical"), ("arch", "no-such-arch"),
    ("search_bias", "aggressive"), ("value_horizon_mid", 2),
    ("policy_soft_temperature", 0.5), ("policy_noise_correction", "yes"),
    ("search_value_mix_final", -1.0), ("unknown_key", 1),
])
def test_an_invalid_configuration_is_rejected_with_its_name(key, value):
    with pytest.raises(ValueError) as error:
        validate(dict(DEFAULTS, **{key: value}))
    assert key in str(error.value)


def test_channels_and_blocks_must_match_the_architecture():
    with pytest.raises(ValueError, match="disagree"):
        validate(dict(DEFAULTS, arch="hybrid-8-1", channels=64, blocks=1))


def test_a_file_overrides_the_defaults_and_keeps_the_rule_argument(tmp_path):
    path = write(tmp_path / "cfg.json", {"workers": 3, "simulations": 64})
    cfg = from_file(path, "renju")
    assert cfg["workers"] == 3 and cfg["simulations"] == 64 and cfg["rule"] == "renju"
    assert cfg["cheap_search_simulations"] == DEFAULTS["cheap_search_simulations"]
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        from_file(write(tmp_path / "bad.json", {"nonsense": 1}), "freestyle")


def test_the_old_candidate_modes_map_onto_the_decoupled_switches(tmp_path):
    path = write(tmp_path / "old.json", {"candidates": "legal"})
    cfg = from_file(path, "freestyle")
    assert cfg["hard_rules"] == "none" and cfg["search_bias"] == "none"
    assert "candidates" not in cfg
    with pytest.raises(ValueError, match="candidates"):
        from_file(write(tmp_path / "worse.json", {"candidates": "square3"}), "freestyle")
    assert translate_legacy({"candidates": "tactical"})["search_bias"] == "tactical"


def test_resume_only_allows_the_fields_that_describe_data_gathering():
    stored = dict(DEFAULTS)
    assert resume_incompatible(dict(DEFAULTS, workers=1), stored) == set()
    assert resume_incompatible(dict(DEFAULTS, opening_plies=0), stored) == set()
    assert resume_incompatible(dict(DEFAULTS, selfplay_dump=True), stored) == set()
    assert resume_incompatible(dict(DEFAULTS, simulations=9), stored) == {"simulations"}
    # A key the checkpoint predates is compared at its current default.
    older = {key: value for key, value in stored.items() if key != "cheap_search_prob"}
    assert resume_incompatible(dict(DEFAULTS), older) == set()
    assert resume_incompatible(dict(DEFAULTS, cheap_search_prob=0.0), older) == {"cheap_search_prob"}


def test_defaults_builds_a_validated_configuration_for_direct_callers():
    cfg = defaults("renju", arch="hybrid-8-1", simulations=4)
    assert cfg["rule"] == "renju" and cfg["arch"] == "hybrid-8-1"
    assert (cfg["channels"], cfg["blocks"]) == (8, 1) and cfg["simulations"] == 4
    with pytest.raises(ValueError):
        defaults("renju", simulations=0)
