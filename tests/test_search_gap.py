"""The search-versus-policy gap: the measurement the search stack exists for."""
import time

import numpy as np
import pytest
import torch

from vk.config import DEFAULTS
from vk.evaluation import evaluate_search_gap
from vk.network import Network


def cfg(**overrides):
    base = dict(DEFAULTS, rule="freestyle", arch="hybrid-8-1", channels=8, blocks=1,
                simulations=2, workers=1, opening_plies=2, temperature_moves=0)
    base.update(overrides)
    return base


def test_the_search_gap_report_states_both_arms_and_a_paired_interval():
    torch.set_num_threads(2)
    model = Network("hybrid-8-1")
    report = evaluate_search_gap(model, cfg(), "cpu", pairs=2, deadline=time.monotonic() + 60)
    assert report["games"] == 4 and report["complete"]
    assert report["search_mode"] == "mcts" and report["baseline"] == "policy"
    assert -100 <= report["search_advantage_pp"] <= 100
    low, high = report["search_advantage_ci95_pp"]
    assert low <= report["search_advantage_pp"] <= high
    # Both arms face the same openings, so the summary has one game per arm per
    # (pair, colour) and the per-colour split sums to the total.
    assert report["wins"] + report["losses"] + report["draws"] == 4
    assert set(report["by_color"]) == {"1", "-1"}
    assert "search added nothing measurable" in report["reading"]


def test_every_searched_move_reports_its_shape():
    torch.set_num_threads(2)
    model = Network("hybrid-8-1")
    report = evaluate_search_gap(model, cfg(simulations=3), "cpu", pairs=1,
                                 deadline=time.monotonic() + 60)
    statistics = report.get("search_statistics", {})
    assert statistics, "the searched arm must publish its search shape"
    assert statistics["moves"] > 0
    assert "policy_only" in statistics["hard_mode_hist"], \
        "the baseline arm plays without search and must say so"
    assert 0 <= statistics["root_max_visit_share_mean"] <= 1


def test_a_stopped_run_returns_what_it_has_instead_of_hanging():
    torch.set_num_threads(2)
    model = Network("hybrid-8-1")
    report = evaluate_search_gap(model, cfg(), "cpu", pairs=3, deadline=time.monotonic() - 1)
    assert report["games"] == 0 and report["complete"] is False
    assert report["search_advantage_pp"] is None and report["search_advantage_ci95_pp"] is None


def test_the_interval_transforms_with_the_paired_score():
    from vk.evaluation import _search_gap_report
    results = [{"result": 1, "color": 1, "winner": 1, "pair": 0, "moves": [], "moves_detail": []},
               {"result": -1, "color": -1, "winner": 1, "pair": 0, "moves": [], "moves_detail": []}]
    report = _search_gap_report(results, [], 1)
    assert report["score"] == pytest.approx(0.5)
    assert report["search_advantage_pp"] == pytest.approx(0.0)
    low, high = report["search_advantage_ci95_pp"]
    assert low <= 0 <= high
