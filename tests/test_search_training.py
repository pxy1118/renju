from collections import deque
import json
import numpy as np
import pytest
import torch
from az.game import Game
from az.search import MCTS, SearchStopped
from az.network import Network, Evaluator
from az.training import DEFAULTS, augment, update, save, load_checkpoint, train
from az.selfplay import play_game, collect
from az.evaluation import summary
from az.cli import read_config


def uniform(state):
    return np.zeros(225),0.0


def test_immediate_win_sign_and_tree_reuse():
    g = Game()
    g.board[105:109] = 1
    tree = MCTS(uniform,600)
    pi = tree.policy(g)
    assert np.argmax(pi) == 109
    assert tree.root.w[109]/tree.root.n[109] == 1
    child = tree.root.children[109]
    g.move(109)
    tree.advance(109,g)
    assert tree.root is child
    assert tree.key == tree.state_key(g)


def test_must_defend():
    g = Game()
    g.board[105:109] = -1
    def focused(state):
        logits = np.full(225,-30.0)
        logits[109] = 1
        logits[110] = 2
        return logits,0.0
    tree = MCTS(focused,100)
    pi = tree.policy(g)
    assert np.argmax(pi) == 109
    assert tree.root.w[110] < 0


def test_legality_noise_and_stop():
    g = Game("renju")
    tree = MCTS(uniform,4)
    pi = tree.policy(g,noise=True)
    assert pi[112] == 1 and pi.sum() == 1
    with pytest.raises(SearchStopped):
        tree.policy(g,stop=lambda: True)


def test_invalid_inference_fails_explicitly():
    with pytest.raises(RuntimeError,match="Invalid network"):
        MCTS(lambda state:(np.full(225,np.nan),0),4).policy(Game())


@pytest.mark.parametrize("rule", ["freestyle","renju"])
def test_full_game_labels(rule):
    data,stats = play_game(rule,uniform,1,441)
    g = Game(rule)
    for (x,pi,z),a in zip(data,stats["moves"]):
        assert np.array_equal(x,g.encode())
        assert pi.sum() == 1
        assert not pi[~g.legal()].any()
        assert z == g.player*stats["winner"]
        g.move(a)
    assert g.winner == stats["winner"]


def test_augment_alignment_all_eight():
    x = np.zeros((3,15,15),np.uint8)
    x[0,2,4] = 1
    p = np.zeros(225)
    p[34] = 1
    seen = set()
    for k in range(4):
        for m in (False,True):
            a,b = augment(x,p,k,m)
            assert np.array_equal(a[0].ravel(),b)
            seen.add(b.tobytes())
    assert len(seen) == 8


def test_update_checkpoint_roundtrip_and_isolation(tmp_path):
    torch.set_num_threads(2)
    cfg = dict(DEFAULTS,channels=8,blocks=1)
    model = Network(8,1)
    opt = torch.optim.Adam(model.parameters())
    rng = np.random.default_rng(8)
    replay = deque([(Game().encode(), np.full(225,1/225),1.0)],maxlen=1000)
    before = next(model.parameters()).detach().clone()
    metrics = update(model,opt,replay,2,"cpu",rng)
    assert metrics["loss"] > 0
    assert not torch.equal(before,next(model.parameters()))
    for i in range(5):
        path = save(tmp_path,cfg,model,opt,replay,rng,i,i,2)
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 3
    state = load_checkpoint(path,"freestyle")
    assert state["step"] == 4 and len(state["replay"]) == 1
    restored = Network(8,1)
    restored.load_state_dict(state["model"])
    assert np.allclose(Evaluator(restored,"cpu")(Game().encode())[0],Evaluator(model,"cpu")(Game().encode())[0])
    with pytest.raises(ValueError):
        load_checkpoint(path,"renju")
    next_rng = np.random.default_rng()
    next_rng.bit_generator.state = state["rng"]
    assert next_rng.random() == rng.random()


def test_timeout_does_not_label_partial_games():
    import time
    class Batch:
        def batch(self,states):
            return np.zeros((len(states),225)),np.zeros(len(states))
    cfg = dict(DEFAULTS,workers=1)
    data,games,_ = collect(cfg,Batch(),[1],time.monotonic()-1)
    assert data == [] and games == []


def test_paired_statistics_and_config(tmp_path):
    stats = summary([{"result":1,"color":1},{"result":-1,"color":-1}],2)
    assert stats["score"] == 0.5 and stats["complete"]
    assert stats["by_color"]["1"]["wins"] == 1
    file = tmp_path/"bad.json"
    file.write_text(json.dumps({"workers":0}))
    with pytest.raises(ValueError):
        read_config(file,"freestyle")


def test_project_gpu_lock_and_release(tmp_path,monkeypatch):
    import az.cli as cli
    monkeypatch.setattr(cli,"ROOT",tmp_path)
    with cli.gpu_lock("cuda"):
        with pytest.raises(RuntimeError,match="Another project process"):
            with cli.gpu_lock("cuda"):
                pass
    with cli.gpu_lock("cuda"):
        pass
