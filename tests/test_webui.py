import json
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest
import torch

from az.game import lengths
from az.network import Network
from az.training import DEFAULTS, atomic_save
from az.webui import Table, make_server


def wait(table):
    deadline = time.monotonic() + 20
    while table.snapshot()["busy"] and time.monotonic() < deadline:
        time.sleep(.01)
    state = table.snapshot()
    assert not state["busy"]
    assert state["error"] is None
    return state


@pytest.fixture
def models(tmp_path):
    torch.set_num_threads(1)
    for rule in ("freestyle", "renju"):
        cfg = dict(DEFAULTS, rule=rule, channels=4, blocks=1)
        atomic_save(dict(format=1, config=cfg, model=Network(4, 1).state_dict(), step=7),
                    tmp_path / rule / "checkpoint-00000001.pt")
    return tmp_path


@pytest.mark.parametrize("rule,color", [("freestyle", "black"), ("renju", "white"), ("renju", "black")])
def test_real_inference_move_and_undo(models, rule, color):
    table = Table(models)
    try:
        table.new(dict(rule=rule, color=color, checkpoint="latest", simulations=32))
        before = wait(table)
        if rule == "renju":
            if color == "white":
                assert before["history"] == [112]
            else:
                assert before["legal"] == [112]
                with pytest.raises(ValueError):
                    table.act(dict(id=before["id"], action=0))
        action = before["legal"][0]
        table.act(dict(id=before["id"], action=action))
        after = wait(table)
        assert len(after["history"]) == len(before["history"]) + 2
        assert after["player"] == after["human"]
        with pytest.raises(ValueError):
            table.act(dict(id=before["id"], action=action))
        undone = table.act(dict(id=before["id"]), undo=True)
        assert undone["board"] == before["board"]
        assert undone["history"] == before["history"]
        table.new(dict(rule=rule, color=color, checkpoint="latest", simulations=32))
        wait(table)
        with pytest.raises(ValueError):
            table.act(dict(id=before["id"], action=action))
    finally:
        table.pool.shutdown()


def test_explicit_checkpoint_selection(models):
    """A named historical checkpoint can be replayed, not only latest/best."""
    import time
    for round_id in (2, 3):
        atomic_save(dict(format=1, config=dict(DEFAULTS, rule="freestyle", channels=4, blocks=1),
                         model=Network(4, 1).state_dict(), step=round_id * 100),
                    models / "freestyle" / f"checkpoint-{round_id:08d}.pt")
        time.sleep(0.01)
    table = Table(models)
    try:
        listing = table.models()["freestyle"]
        assert listing["latest"] == "checkpoint-00000003.pt"
        recent = [item["name"] for item in table.recent("freestyle")]
        assert recent[0] == "checkpoint-00000003.pt" and "checkpoint-00000001.pt" in recent
        assert all(item["mb"] > 0 and item["mtime"] for item in table.recent("freestyle"))
        oldest = recent[-1]
        assert oldest == "checkpoint-00000001.pt"  # fixture checkpoint, written first
        table.new(dict(rule="freestyle", color="black", checkpoint=oldest, simulations=32))
        state = wait(table)
        assert state["model"]["file"] == oldest
        assert state["model"]["step"] == 7
        assert state["model"]["latest"] == "checkpoint-00000003.pt"
        # Names that cannot be a file inside the rule folder are refused up front.
        for name in ("../secrets.pt", "artifacts/x.pt", "sub\\x.pt"):
            with pytest.raises(ValueError, match="无效"):
                table.new(dict(rule="freestyle", color="black", checkpoint=name))
    finally:
        table.pool.shutdown()


def test_missing_file_reports_through_error_channel(models):
    """A pruned or stale name is reported to the user once the load fails."""
    table = Table(models)
    try:
        table.new(dict(rule="freestyle", color="black",
                       checkpoint="checkpoint-00009999.pt", simulations=32))
        deadline = time.monotonic() + 20
        while table.snapshot()["busy"] and time.monotonic() < deadline:
            time.sleep(.01)
        state = table.snapshot()
        assert not state["busy"]
        assert state["error"] and "失败" in state["error"]
        with pytest.raises(ValueError):
            table.act(dict(id=state["id"], action=0))
        # Starting a valid game afterwards clears the failed state.
        table.new(dict(rule="freestyle", color="black", checkpoint="latest", simulations=32))
        assert wait(table)["error"] is None
    finally:
        table.pool.shutdown()


def winning_move(state):
    """First legal action that completes five in a row for the side to move."""
    board = np.array(state["board"], np.int8)
    for action in state["legal"]:
        probe = board.copy()
        probe[action] = state["player"]
        if max(lengths(probe, action, state["player"])) >= 5:
            return action
    return None


def test_human_win_is_reported_immediately(models):
    """The winning move must not leave the table 'thinking' or blame the wrong side."""
    table = Table(models)
    try:
        state = table.new(dict(rule="freestyle", color="black", checkpoint="latest", simulations=32))
        state = wait(table)
        assert state["history"] == []
        for _ in range(8):
            action = winning_move(state)
            if action is None:
                action = state["legal"][0]
            state = table.act(dict(id=state["id"], action=action))
            assert state["winner"] is None or state["busy"] is False, \
                "a finished game must not stay busy"
            state = wait(table)
            if state["winner"] is not None:
                break
            assert state["player"] == state["human"] == 1
        assert state["winner"] == 1 == state["human"], state
        assert state["legal"] == [] and state["error"] is None
        # The page paints its side indicator from these two fields, so a finished
        # game must not look like "white to move" next to "you won".
        assert state["player"] == -state["winner"], "winner's opponent is 'to move'"
        assert state["human"] == state["winner"]
        black = np.array(state["board"], np.int8)
        assert max(max(lengths(black, a, 1)) for a in range(225)
                   if state["board"][a] == 1) == 5
    finally:
        table.pool.shutdown()


def test_missing_model_and_busy(tmp_path):
    table = Table(tmp_path)
    try:
        with pytest.raises(ValueError, match="检查点"):
            table.new(dict(rule="renju", color="black", checkpoint="latest"))
        table.busy = True
        with pytest.raises(ValueError, match="思考"):
            table.new(dict(rule="freestyle", color="black", checkpoint="latest"))
    finally:
        table.pool.shutdown()


def test_http_local_protection_and_validation(models):
    server = make_server(0, models)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/config") as response:
            config = json.load(response)
        assert config["models"]["renju"]["latest"].startswith("checkpoint-")
        assert config["models"]["renju"]["best"] is None
        assert [item["name"] for item in config["recent"]["renju"]] == [config["models"]["renju"]["latest"]]
        assert config["recent"]["freestyle"][0]["mb"] > 0
        with urlopen(base) as response:
            assert "棋间" in response.read().decode()
        for headers, payload, status in [({}, {}, 403), ({"X-Renju-Token": config["token"]}, [], 400),
                                         ({"X-Renju-Token": config["token"], "Host": "evil.example"}, {}, 403)]:
            with pytest.raises(HTTPError) as err:
                urlopen(Request(base + "/api/new", data=json.dumps(payload).encode(), headers=headers))
            assert err.value.code == status
    finally:
        server.shutdown()
        server.server_close()
        server.table.pool.shutdown()
        thread.join()
