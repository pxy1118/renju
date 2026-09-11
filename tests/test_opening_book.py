"""The opening book is a measurement instrument, so its contract is exact.

A colour-split evaluation is only meaningful if both colours start from a
position that was *verified* balanced. These tests pin the loading contract and
that evaluation and self-play really do start from the book.
"""
from pathlib import Path
import json
import textwrap

import numpy as np
import pytest

from vk.evaluation import evaluate_rapfi, opening
from vk.network import Network
from vk.openings import balanced_opening, book_opening, load_opening_book
from vk.selfplay import play_game
from vk.training import DEFAULTS


def make_book(path, rule="freestyle", seeds=(1, 2), gap=0.15):
    """A valid book built from deterministic sampled openings."""
    entries = []
    for seed in seeds:
        moves = [int(action) for action in balanced_opening(rule, seed, 8).history]
        entries.append({"moves": moves, "ply": len(moves), "value_black": 0.5,
                        "value_white": 0.5, "value_next_mover": 0.5, "next_player": 1,
                        "canonical": str(seed)})
    book = {"format": "renju-opening-book", "format_version": 1, "rule": rule,
            "count": len(entries), "acceptance": {"gap": gap, "ply_range": [8, 20]},
            "openings": entries}
    Path(path).write_text(json.dumps(book), encoding="utf-8")
    return book


def fake_engine(path):
    path.write_text(textwrap.dedent('''\
        import sys
        occupied = set()
        for raw in sys.stdin:
            command = raw.strip()
            if command.startswith("START"):
                print("OK", flush=True)
            elif command == "YXSHOWINFO":
                print("MESSAGE Rapfi book-fake", flush=True)
            elif command.startswith("YXBOARD"):
                occupied = {int(item.split(",")[1])*15 + int(item.split(",")[0])
                            for item in command.split()[1:-1]}
            elif command.startswith("YXNBEST"):
                legal = [action for action in range(225) if action not in occupied][:2]
                for index, action in enumerate(legal):
                    x, y = action % 15, action // 15
                    print(f"INFO PV {index}", flush=True)
                    print(f"INFO NUMPV {len(legal)}", flush=True)
                    print("INFO DEPTH 3", flush=True)
                    print("INFO WINRATE 0.5", flush=True)
                    print(f"INFO BESTLINE {x},{y}", flush=True)
                    print("INFO PV DONE", flush=True)
                action = legal[0]
                print(f"{action%15},{action//15}", flush=True)
            elif command == "END":
                break
        '''), encoding="utf-8")
    return path


def test_a_book_replays_its_recorded_moves(tmp_path):
    book = make_book(tmp_path / "book.json")
    loaded = load_opening_book(tmp_path / "book.json", rule="freestyle")
    game = book_opening(loaded, 0)
    assert [int(a) for a in game.history] == book["openings"][0]["moves"]
    # Index wraps, so a caller can walk the book with its whole seed stream.
    assert book_opening(loaded, len(book["openings"])).history == game.history


def test_loading_rejects_anything_that_is_not_a_usable_book(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_opening_book(tmp_path / "missing.json")

    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"format": "something-else", "openings": [{"moves": [1]}]}))
    with pytest.raises(ValueError, match="Not an opening book"):
        load_opening_book(wrong)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"format": "renju-opening-book", "openings": []}))
    with pytest.raises(ValueError, match="empty"):
        load_opening_book(empty)

    make_book(tmp_path / "renju.json", rule="renju")
    with pytest.raises(ValueError, match="for 'renju'"):
        load_opening_book(tmp_path / "renju.json", rule="freestyle")

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"format": "renju-opening-book",
                                  "openings": [{"moves": [0, 999]}]}))
    with pytest.raises(ValueError, match="invalid move list"):
        load_opening_book(broken)


def test_the_book_mode_requires_a_path():
    with pytest.raises(ValueError, match="needs a loaded opening book"):
        opening("freestyle", 1, "book")
    with pytest.raises(ValueError, match="Unknown opening mode"):
        opening("freestyle", 1, "magic")


def test_evaluation_starts_from_the_book_opening(tmp_path):
    make_book(tmp_path / "book.json", seeds=(5,))
    engine = fake_engine(tmp_path / "engine.py")
    cfg = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
               simulations=2, workers=1, opening_mode="book",
               opening_book=str(tmp_path / "book.json"))
    report = evaluate_rapfi(Network("hybrid-8-1"), cfg, "cpu", engine, tmp_path,
                            pairs=1, threads=1, hash_mb=8, max_nodes=10, timeout=2,
                            with_records=True)
    assert report["opening_mode"] == "book" and report["complete"]
    assert report["opening_book"] == str(tmp_path / "book.json")
    assert sorted(record["pair"] for record in report["records"]) == [0, 0]
    assert sorted(record["color"] for record in report["records"]) == [-1, 1]
    # The games really began after the book's moves: the report's games are two
    # colours of one opening pair, which is what makes the colour split usable.
    assert report["by_color"]["1"]["wins"] + report["by_color"]["1"]["losses"] == 1
    assert report["by_color"]["-1"]["wins"] + report["by_color"]["-1"]["losses"] == 1


def test_missing_book_fails_before_any_game(tmp_path):
    engine = fake_engine(tmp_path / "engine.py")
    cfg = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
               simulations=2, workers=1, opening_mode="book",
               opening_book=str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="Opening book not found"):
        evaluate_rapfi(Network("hybrid-8-1"), cfg, "cpu", engine, tmp_path,
                       pairs=1, threads=1, hash_mb=8, max_nodes=10, timeout=2)


def test_self_play_can_start_from_the_book(tmp_path):
    make_book(tmp_path / "book.json", seeds=(3,))
    loaded = load_opening_book(tmp_path / "book.json")
    recorded = [int(action) for action in loaded["openings"][0]["moves"]]
    data, stats = play_game("freestyle", lambda state: (np.zeros(225), 0.0), 2, 0,
                            book=loaded)
    assert stats["opening"] == recorded
    assert stats["moves"][:len(recorded)] == recorded
    assert len(data) >= 1
