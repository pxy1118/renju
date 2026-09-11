"""Small CPU integration runs, separate from real CUDA acceptance."""
import json
import time
import numpy as np
import pytest
import torch
from vk.game import Game
from vk.network import Network, Evaluator
from vk.training import DEFAULTS, train, load_checkpoint, checkpoint_path
from vk.selfplay import collect


@pytest.mark.parametrize("rule", ["freestyle","renju"])
def test_train_and_resume(tmp_path,rule):
    torch.set_num_threads(2)
    cfg = dict(DEFAULTS,rule=rule,arch="hybrid-8-1",channels=8,blocks=1,simulations=1,
               workers=1,games_per_round=1,train_steps=1,batch_size=2,replay_capacity=300)
    first = train(cfg,tmp_path,"cpu",90,max_rounds=1)
    assert first["total_games"] == 1 and first["step"] == 1
    before = load_checkpoint(first["checkpoint"],rule)
    second = train(cfg,tmp_path,"cpu",90,resume="latest",max_rounds=1)
    after = load_checkpoint(second["checkpoint"],rule)
    assert after["round"] == 2 and after["step"] == 2 and after["total_games"] == 2
    assert not torch.equal(before["model"]["trunk.0.weight"],after["model"]["trunk.0.weight"])
    with pytest.raises(ValueError):
        train(cfg,tmp_path,"cpu",10)
    with pytest.raises(ValueError):
        train(dict(cfg, arch="legacy-64-6", channels=64, blocks=6),tmp_path,"cpu",10,resume="latest")


def test_resume_allows_worker_count_change(tmp_path,monkeypatch):
    import vk.training as module
    cfg = dict(DEFAULTS,arch="hybrid-8-1",channels=8,blocks=1,simulations=1,workers=1,
               games_per_round=1,train_steps=1,batch_size=2,replay_capacity=300)
    data = [(Game().encode(),np.full(225,1/225),1.0)]
    monkeypatch.setattr(module,"collect",lambda *args: (data,[{"winner":1,"moves":[],"simulations":1}],
                                                       {"seconds":1,"batches":1,"inference_positions":1,
                                                        "average_inference_batch_size":1,
                                                        "largest_inference_batch_size":1}))
    first = train(cfg,tmp_path,"cpu",30,max_rounds=1)
    changed = dict(cfg,workers=2)
    second = train(changed,tmp_path,"cpu",30,resume="latest",max_rounds=1)
    assert load_checkpoint(second["checkpoint"],"freestyle")["config"]["workers"] == 2


def test_resume_tolerates_a_checkpoint_predating_new_config_fields(tmp_path, monkeypatch):
    """Checkpoints written before ``opening_plies`` existed must still resume.

    The field only describes how this process gathers data, so it is compared at
    its current default instead of being reported as a configuration mismatch.
    """
    import vk.training as module
    cfg = dict(DEFAULTS, arch="hybrid-8-1", channels=8, blocks=1, simulations=1, workers=1,
               games_per_round=1, train_steps=1, batch_size=2, replay_capacity=300)
    data = [(Game().encode(), np.full(225, 1/225), 1.0)]
    stats = {"winner": 1, "moves": [], "simulations": 1, "opening": []}
    perf = {"seconds": 1, "batches": 1, "inference_positions": 1,
            "average_inference_batch_size": 1, "largest_inference_batch_size": 1}
    monkeypatch.setattr(module, "collect", lambda *args: (data, [stats], perf))

    first = train(cfg, tmp_path, "cpu", 30, max_rounds=1)
    state = load_checkpoint(first["checkpoint"], "freestyle")
    # Reproduce an old checkpoint: drop the field the new code knows about.
    older = {key: value for key, value in state["config"].items() if key != "opening_plies"}
    assert "opening_plies" not in older
    module.atomic_save({**state, "config": older}, tmp_path / "older.pt")

    resumed = train(dict(cfg, opening_plies=6), tmp_path, "cpu", 30,
                    resume=str(tmp_path / "older.pt"), max_rounds=1)
    stored = load_checkpoint(resumed["checkpoint"], "freestyle")["config"]
    assert stored["opening_plies"] == 6
    # A genuinely incompatible field is still rejected.
    with pytest.raises(ValueError, match="incompatible"):
        train(dict(cfg, simulations=9), tmp_path, "cpu", 30,
              resume=str(tmp_path / "older.pt"), max_rounds=1)


def test_short_run_actually_trains_and_varies_its_openings(tmp_path):
    """Regression for the run that collected 954 games without one update.

    The replay-unlock threshold must be reachable inside the time budget, or
    the loop samples forever and the optimizer never takes a step.
    """
    torch.set_num_threads(2)
    cfg = dict(DEFAULTS, arch="hybrid-8-1", channels=8, blocks=1, simulations=1, workers=1,
               games_per_round=4, train_steps=1, batch_size=2, replay_capacity=5000,
               min_replay_size=512, opening_plies=8)
    train(cfg, tmp_path, "cpu", 600, max_rounds=12)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 12
    unlocked = [index for index, record in enumerate(records) if record["trainable"]]
    assert unlocked, "training never unlocked"
    # Everything from the unlock onwards trains, and nothing before it does.
    assert unlocked == list(range(unlocked[0], len(records)))
    assert records[-1]["step"] == len(unlocked)
    assert records[-1]["opening_distinct"] > 1, "every game used the same opening"
    assert records[-1]["opening_sample_size"] == 8
    winners = {json.loads(line)["winner"]
               for line in (tmp_path / "games.jsonl").read_text().splitlines()}
    assert winners and winners <= {1, -1, 0}


def test_stop_saves_no_fake_outcome(tmp_path):
    cfg = dict(DEFAULTS,arch="hybrid-8-1",channels=8,blocks=1)
    result = train(cfg,tmp_path,"cpu",10,stop=lambda:True)
    state = load_checkpoint(result["checkpoint"],"freestyle")
    assert state["total_games"] == 0 and state["replay"] == [] and state["step"] == 0


def test_worker_error_is_propagated():
    cfg = dict(DEFAULTS,rule="invalid",workers=1)
    model = Network("hybrid-8-1")
    with pytest.raises(RuntimeError,match="Unknown rule"):
        collect(cfg,Evaluator(model,"cpu"),[1],time.monotonic()+20)


def test_resume_finishes_pending_updates_before_new_games(tmp_path,monkeypatch):
    import vk.training as module
    cfg = dict(DEFAULTS,arch="hybrid-8-1",channels=8,blocks=1,train_steps=3,batch_size=2)
    data = [(Game().encode(),np.full(225,1/225),1.0)]
    monkeypatch.setattr(module,"collect",lambda *args: (data,[{"winner":1,"moves":[],"simulations":1}],
                                                       {"seconds":1,"batches":1,"inference_positions":1}))
    original = module.update
    calls = [0]
    def counted(*args):
        result = original(*args)
        calls[0] += 1
        return result
    monkeypatch.setattr(module,"update",counted)
    first = train(cfg,tmp_path,"cpu",30,stop=lambda:calls[0] >= 1)
    assert load_checkpoint(first["checkpoint"],"freestyle")["pending_steps"] == 2
    def unexpected(*args):
        raise AssertionError("Must optimize pending replay first")
    monkeypatch.setattr(module,"collect",unexpected)
    resumed = train(cfg,tmp_path,"cpu",30,resume="latest",max_rounds=1)
    state = load_checkpoint(resumed["checkpoint"],"freestyle")
    assert state["step"] == 3 and state["pending_steps"] == 0 and state["round"] == 1
