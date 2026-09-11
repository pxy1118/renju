from collections import deque
import json
import numpy as np
import pytest
import torch
from vk.game import Game
from vk.openings import balanced_opening
from vk.search import MCTS, SearchStopped, policy_move
from vk.network import Network, Evaluator
from vk.training import DEFAULTS, augment, update, save, load_checkpoint, train, reconcile_jsonl
from vk.selfplay import play_game, collect
from vk.evaluation import summary, match
from vk.cli import read_config


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
    assert pi[110] == 0
    assert tree.last_stats["candidate_mode"] == "forced_defense"


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


def test_focused_inference_must_not_leak_outside_the_candidate_mask():
    """A move the candidate set forbids must stay unreachable, whatever the prior."""
    g = Game()
    g.board[105:109] = -1                     # White threatens five on the row
    def biased(state):
        logits = np.full(225, -30.0)
        logits[7] = 20.0                      # far away, outside the mask
        logits[105] = 1.0
        return logits, 0.0
    action, candidates = policy_move(g, biased, "forced")
    assert candidates.mode == "forced_defense"
    assert action in (104, 109), "only the two blocking points exist"
    assert action != 7


def test_policy_move_uses_the_raw_ranking_when_nothing_is_forced():
    g = Game()
    g.move(112)
    def ranked(state):
        logits = np.zeros(225)
        logits[113] = 5.0
        return logits, 0.9
    action, candidates = policy_move(g, ranked, "legal")
    assert candidates.mode == "all_legal"
    assert action == 113


def test_legal_candidates_give_every_point_a_prior_and_expose_search_shape():
    g = Game()
    g.move(112)
    tree = MCTS(uniform, 40, 2.0, candidates="legal")
    tree.policy(g)
    stats = tree.last_stats
    assert stats["candidate_mode"] == "all_legal"
    assert stats["candidate_count"] == 224
    assert (tree.root.p > 0).sum() == 224, "no legal point may start at zero prior"
    # The point of these numbers is to show how far the search actually spread,
    # which visits-per-move cannot: every simulation adds exactly one root visit.
    assert stats["root_visited_moves"] <= 40
    assert 0 < stats["root_max_visit_share"] <= 1
    if stats["root_visited_moves"] == 1:
        assert stats["root_max_visit_share"] == 1 and stats["root_visit_entropy"] == pytest.approx(0)


def test_root_visit_shape_is_degenerate_when_all_visits_land_on_one_move():
    g = Game()
    g.board[100:104] = 1
    def single(state):
        logits = np.full(225, -30.0)
        logits[104] = 10.0
        return logits, 0.5
    tree = MCTS(single, 30, 2.0, candidates="forced")
    tree.policy(g)
    stats = tree.last_stats
    assert stats["root_visited_moves"] == 1
    assert stats["root_max_visit_share"] == pytest.approx(1.0)
    assert stats["root_visit_entropy"] == pytest.approx(0.0)
    assert stats["mean_visits_per_visited_move"] == pytest.approx(30.0)


def test_zero_simulations_report_no_visit_distribution():
    tree = MCTS(uniform, 0)
    pi = tree.policy(Game())
    assert not pi.any()
    assert tree.last_stats["root_visited_moves"] == 0
    assert tree.last_stats["root_max_visit_share"] == 0.0


def test_tactical_candidates_remain_the_default():
    assert MCTS(uniform, 4).candidates == "tactical"


@pytest.mark.parametrize("rule", ["freestyle","renju"])
def test_full_game_labels(rule):
    data,stats = play_game(rule,uniform,1,441)
    # Samples cover the searched moves only, so replay starts from the opening
    # the game actually began in.
    g = balanced_opening(rule,441)
    assert stats["opening"] == [int(a) for a in g.history]
    for (x,pi,z),a in zip(data,stats["moves"][len(stats["opening"]):]):
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
    cfg = dict(DEFAULTS,arch="hybrid-8-1",channels=8,blocks=1)
    model = Network("hybrid-8-1")
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
    restored = Network("hybrid-8-1")
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


def test_collect_does_not_drop_tasks_during_worker_startup():
    import time
    class Batch:
        def batch(self, states):
            return np.zeros((len(states), 225)), np.zeros(len(states))
    cfg = dict(DEFAULTS, workers=8, simulations=1)
    _, games, _ = collect(cfg, Batch(), range(8), time.monotonic() + 30)
    assert len(games) == 8


def test_resume_reconciles_uncommitted_jsonl(tmp_path):
    path = tmp_path / "games.jsonl"
    path.write_text('\n'.join(json.dumps({"round": value}) for value in (1, 2, 3)) + '\n',
                    encoding="utf-8")
    assert reconcile_jsonl(path, 2) == 1
    assert [json.loads(line)["round"] for line in path.read_text().splitlines()] == [1, 2]


def test_neural_arena_uses_batched_workers():
    import time
    model = Network("hybrid-8-1")
    cfg = dict(DEFAULTS, arch="hybrid-8-1", channels=8, blocks=1, workers=2, simulations=1)
    report = match(model, cfg, "cpu", Network("hybrid-8-1"), pairs=1,
                   deadline=time.monotonic() + 30)
    assert report["games"] == 2 and report["complete"]


def test_collect_starts_every_game_from_its_balanced_opening():
    import time
    class Batch:
        def batch(self, states):
            return np.zeros((len(states), 225)), np.zeros(len(states))
    cfg = dict(DEFAULTS, workers=1, simulations=1, opening_plies=8)
    _, games, _ = collect(cfg, Batch(), [17], time.monotonic() + 30)
    game = games[0]
    assert len(game["opening"]) == 8
    assert game["moves"][:8] == game["opening"]
    assert game["opening"] == [int(a) for a in balanced_opening("freestyle", 17, 8).history]


def test_collect_without_an_opening_starts_from_the_empty_board():
    import time
    class Batch:
        def batch(self, states):
            return np.zeros((len(states), 225)), np.zeros(len(states))
    cfg = dict(DEFAULTS, workers=1, simulations=1, opening_plies=0)
    _, games, _ = collect(cfg, Batch(), [5], time.monotonic() + 30)
    assert games[0]["opening"] == []


def test_metrics_record_expose_whether_training_can_start(tmp_path, monkeypatch):
    """The starvation bug was invisible; these fields make it visible."""
    import vk.training as module
    data = [(Game().encode(), np.full(225, 1 / 225), 1.0)]
    stats = {"winner": 1, "moves": [112], "simulations": 1, "opening": []}
    perf = {"seconds": 1, "batches": 1, "inference_positions": 1,
            "average_inference_batch_size": 1, "largest_inference_batch_size": 1}
    monkeypatch.setattr(module, "collect", lambda *args: (data, [stats], perf))

    locked = dict(DEFAULTS, arch="hybrid-8-1", channels=8, blocks=1, train_steps=1, batch_size=1,
                  min_replay_size=10_000)
    module.train(locked, tmp_path / "locked", "cpu", 30, max_rounds=1)
    record = json.loads((tmp_path / "locked" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["trainable"] is False and record["updates"] == 0
    assert record["min_replay_size"] == 10_000

    open_ = dict(locked, min_replay_size=1)
    module.train(open_, tmp_path / "open", "cpu", 30, max_rounds=1)
    record = json.loads((tmp_path / "open" / "metrics.jsonl").read_text().splitlines()[0])
    assert record["trainable"] is True and record["updates"] == 1


def test_paired_statistics_and_config(tmp_path):
    stats = summary([{"result":1,"color":1},{"result":-1,"color":-1}],2)
    assert stats["score"] == 0.5 and stats["complete"]
    assert stats["by_color"]["1"]["wins"] == 1
    collapsed = summary([{"result": 1 if i % 2 == 0 else -1,
                          "color": 1 if i % 2 == 0 else -1, "winner": 1}
                         for i in range(128)], 128)
    assert collapsed["color_collapse_detected"] is True
    file = tmp_path/"bad.json"
    file.write_text(json.dumps({"workers":0}))
    with pytest.raises(ValueError):
        read_config(file,"freestyle")


def test_project_gpu_lock_and_release(tmp_path,monkeypatch):
    import vk.cli as cli
    monkeypatch.setattr(cli,"ROOT",tmp_path)
    with cli.gpu_lock("cuda"):
        with pytest.raises(RuntimeError,match="Another project process"):
            with cli.gpu_lock("cuda"):
                pass
    with cli.gpu_lock("cuda"):
        pass
