"""Small CPU integration runs, separate from real CUDA acceptance."""
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
    cfg = dict(DEFAULTS,rule=rule,channels=4,blocks=1,simulations=1,
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
        train(dict(cfg,channels=8),tmp_path,"cpu",10,resume="latest")


def test_resume_allows_worker_count_change(tmp_path,monkeypatch):
    import vk.training as module
    cfg = dict(DEFAULTS,channels=4,blocks=1,simulations=1,workers=1,
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


def test_stop_saves_no_fake_outcome(tmp_path):
    cfg = dict(DEFAULTS,channels=4,blocks=1)
    result = train(cfg,tmp_path,"cpu",10,stop=lambda:True)
    state = load_checkpoint(result["checkpoint"],"freestyle")
    assert state["total_games"] == 0 and state["replay"] == [] and state["step"] == 0


def test_worker_error_is_propagated():
    cfg = dict(DEFAULTS,rule="invalid",workers=1)
    model = Network(4,1)
    with pytest.raises(RuntimeError,match="Unknown rule"):
        collect(cfg,Evaluator(model,"cpu"),[1],time.monotonic()+20)


def test_resume_finishes_pending_updates_before_new_games(tmp_path,monkeypatch):
    import vk.training as module
    cfg = dict(DEFAULTS,channels=4,blocks=1,train_steps=3,batch_size=2)
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
