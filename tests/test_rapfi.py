from pathlib import Path
import textwrap
import os
import pytest

from az.game import Game
from az.rapfi import RapfiClient, RapfiError, board_command


def fake_engine(path, mode="ok"):
    path.write_text(textwrap.dedent(f'''\
        import pathlib, sys, time
        mode = {mode!r}
        marker = pathlib.Path("attempt.marker")
        for raw in sys.stdin:
            command = raw.strip()
            if command.startswith("START"):
                print("OK", flush=True)
            elif command == "YXSHOWINFO":
                print("MESSAGE Rapfi fake-1.0", flush=True)
            elif command.startswith("YXNBEST"):
                if mode in ("timeout", "crash") and not marker.exists():
                    marker.write_text("1")
                    if mode == "timeout": time.sleep(0.3)
                    else: sys.exit(3)
                if mode == "incomplete":
                    print("INFO PV 0", flush=True); print("INFO NUMPV 5", flush=True)
                    print("INFO DEPTH 2", flush=True); print("INFO WINRATE 0.7", flush=True)
                    print("INFO BESTLINE 0,1", flush=True); print("INFO PV DONE", flush=True)
                    print("0,1", flush=True); continue
                if mode == "illegal":
                    print("INFO PV 0", flush=True); print("INFO NUMPV 1", flush=True)
                    print("INFO DEPTH 2", flush=True); print("INFO WINRATE 0.7", flush=True)
                    print("INFO BESTLINE 7,7", flush=True); print("INFO PV DONE", flush=True)
                    print("7,7", flush=True); continue
                for i in range(5):
                    print(f"INFO PV {{i}}", flush=True); print("INFO NUMPV 5", flush=True)
                    print("INFO DEPTH 1", flush=True); print(f"INFO NODES {{100+i}}", flush=True)
                    print(f"INFO WINRATE {{0.8-i*0.1}}", flush=True)
                    print(f"INFO BESTLINE {{i}},1 {{i}},2", flush=True); print("INFO PV DONE", flush=True)
                print("INFO PV 0", flush=True); print("INFO NUMPV 5", flush=True)
                print("INFO DEPTH 2", flush=True); print("INFO WINRATE 0.9", flush=True)
                print("INFO BESTLINE 2,2", flush=True); print("INFO PV DONE", flush=True)
                print("0,1", flush=True)
            elif command == "END":
                break
        '''), encoding="utf-8")
    return path


def position():
    game = Game()
    game.move(112)
    return game


def test_yxboard_roles_follow_current_player():
    game = Game()
    game.move(112)
    assert board_command(game) == "YXBOARD 7,7,2 DONE"
    game.move(111)
    assert board_command(game) == "YXBOARD 7,7,1 6,7,2 DONE"


def test_last_complete_multipv_and_coordinate_conversion(tmp_path):
    engine = fake_engine(tmp_path / "fake.py")
    with RapfiClient(engine, tmp_path, timeout=1, retries=0) as client:
        result = client.analyze(position(), 5)
    assert result.version == "fake-1.0"
    assert [move.action for move in result.moves] == [15, 16, 17, 18, 19]
    assert all(move.depth == 1 for move in result.moves)
    assert result.moves[0].nodes == 100 and result.moves[0].pv == (15, 30)


@pytest.mark.parametrize("mode", ["timeout", "crash"])
def test_restart_and_retry(tmp_path, mode):
    engine = fake_engine(tmp_path / "fake.py", mode)
    with RapfiClient(engine, tmp_path, timeout=0.1, retries=1) as client:
        assert len(client.analyze(position(), 5).moves) == 5


def test_incomplete_output_is_rejected(tmp_path):
    engine = fake_engine(tmp_path / "fake.py", "incomplete")
    with RapfiClient(engine, tmp_path, timeout=1, retries=1) as client:
        with pytest.raises(RapfiError, match="complete MultiPV"):
            client.analyze(position(), 5)


def test_illegal_teacher_move_is_rejected(tmp_path):
    engine = fake_engine(tmp_path / "fake.py", "illegal")
    with RapfiClient(engine, tmp_path, timeout=1, retries=0) as client:
        with pytest.raises(RapfiError, match="illegal"):
            client.analyze(position(), 1)


@pytest.mark.integration
@pytest.mark.skipif(not (os.environ.get("RAPFI_ENGINE") and os.environ.get("RAPFI_ENGINE_DIR")),
                    reason="set RAPFI_ENGINE and RAPFI_ENGINE_DIR for the real integration test")
def test_real_user_supplied_rapfi():
    with RapfiClient(os.environ["RAPFI_ENGINE"], os.environ["RAPFI_ENGINE_DIR"],
                     threads=1, hash_mb=64, max_nodes=5000, timeout=5, retries=0) as client:
        result = client.analyze(position(), 5)
    assert len(result.moves) == 5
    assert all(position().legal()[move.action] for move in result.moves)
