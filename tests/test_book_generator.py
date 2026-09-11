"""The generator's balance arithmetic, which a one-sided check silently broke.

``balance()`` is the acceptance rule. An earlier version used
``min(V_black, V_white) <= gap``, which with two complementary values is at most
0.5 and therefore accepted ``V_black = 0.03`` -- the most lopsided opening
possible. These tests pin the corrected two-sided rule and the replay helpers.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from artifacts import opening_book_generate as generator
from vk.openings import balanced_opening


def entry(value_black):
    return {"value_black": value_black, "value_white": 1.0 - value_black}


def test_balance_is_two_sided():
    # 0.03 is the lopsided case the one-sided rule used to accept.
    assert generator.balance(entry(0.03)) == pytest.approx(0.47)
    assert generator.balance(entry(0.97)) == pytest.approx(0.47)
    assert generator.balance(entry(0.5)) == pytest.approx(0.0)
    assert not generator.accepted_by(entry(0.03), 0.15)
    assert not generator.accepted_by(entry(0.97), 0.15)
    assert generator.accepted_by(entry(0.5), 0.15)
    # Just inside the band is kept, just outside is dropped. The boundary itself
    # is not asserted: 0.35 - 0.5 is not exactly -0.15 in binary floating point.
    assert generator.accepted_by(entry(0.36), 0.15)
    assert generator.accepted_by(entry(0.64), 0.15)
    assert not generator.accepted_by(entry(0.34), 0.15)
    assert not generator.accepted_by(entry(0.66), 0.15)


def test_opening_replay_round_trips_the_move_list():
    moves = [int(action) for action in balanced_opening("freestyle", 11, 8).history]
    game = generator.opening_from_moves("freestyle", moves)
    assert [int(action) for action in game.history] == moves
    assert np.count_nonzero(game.board) == len(moves)
    assert game.player == 1 if len(moves) % 2 == 0 else -1


def test_deduplicate_keeps_one_member_of_each_class_and_prefers_balance():
    rows = [{"canonical": "a", "value_black": 0.9}, {"canonical": "a", "value_black": 0.5},
            {"canonical": "b", "value_black": 0.2}]
    kept = generator.deduplicate(rows, lambda item: item["canonical"])
    assert sorted(item["canonical"] for item in kept) == ["a", "b"]
    assert next(item for item in kept if item["canonical"] == "a")["value_black"] == 0.5
