"""The round report is what makes a training run readable."""
import numpy as np
import pytest

from vk.config import DEFAULTS
from vk.diagnostics import (entropy, kl_divergence, record_round, saturation,
                            target_report)
from vk.network import Network
from vk.records import VALUE_HEADS, blank, set_head


def test_saturation_reports_the_shape_of_the_labels():
    report = saturation(np.array([0.99, -0.98, 0.1, 0.95]))
    assert report["rows"] == 4
    assert report["abs_gt_0.9_share"] == pytest.approx(0.75)
    assert report["abs_lt_0.5_share"] == pytest.approx(0.25)
    assert report["mean_abs"] == pytest.approx((0.99 + 0.98 + 0.1 + 0.95) / 4)
    masked = saturation(np.array([0.99, 0.0]), np.array([True, False]))
    assert masked["rows"] == 1 and masked["abs_gt_0.9_share"] == 1.0
    assert saturation(np.array([np.nan]))["rows"] == 0


def test_entropy_and_divergence_agree_with_their_definitions():
    uniform = np.full((1, 4), 0.25)
    assert entropy(uniform) == pytest.approx(np.log(4))
    certain = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert entropy(certain) == pytest.approx(0.0)
    assert kl_divergence(certain, uniform) == pytest.approx(np.log(4))
    assert kl_divergence(uniform, uniform) == pytest.approx(0.0, abs=1e-9)
    # A row with no support is skipped rather than counted as zero divergence.
    empty = np.zeros((1, 4))
    assert kl_divergence(np.vstack([empty, certain]), np.vstack([uniform, uniform])) == \
        pytest.approx(np.log(4))


def synthetic_records(rows=6):
    batch = blank(rows)
    batch["state"][:, 2] = 1
    batch["policy"][:, 0] = 1.0
    batch["policy_valid"] = 1
    batch["policy_weight"] = 1.0
    batch["full_search"][0] = 1
    batch["q_spread"] = 0.2
    batch["policy_surprise"] = 0.5
    batch["value_surprise"] = 0.1
    batch["weight"] = 2.0
    targets = np.array([1.0, -1.0, 0.0, 0.5, -0.5, 1.0], np.float32)[:rows]
    set_head(batch, "final", targets)
    set_head(batch, "mid", np.full(rows, np.nan, np.float32))
    return batch


def test_the_target_report_describes_every_head_and_the_pipeline():
    report = target_report(synthetic_records())
    assert report["rows"] == 6
    for head in VALUE_HEADS:
        assert f"target_saturation_{head}" in report
    assert report["target_saturation_mid"]["rows"] == 0
    assert report["policy_valid_share"] == 1.0
    assert report["full_search_share"] == pytest.approx(1 / 6)
    assert report["policy_target_entropy"] == pytest.approx(0.0)
    assert report["q_spread_mean"] == pytest.approx(0.2, abs=1e-3)
    # Weights are reported normalised to mean one: that is the distribution the
    # sampler draws from, so a round with small surprises cannot look broken.
    assert report["sample_weight_mean"] == pytest.approx(1.0)


def test_the_round_report_runs_the_network_over_the_batch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    model = Network("hybrid-8-1")
    cfg = dict(DEFAULTS, arch="hybrid-8-1", channels=8, blocks=1)
    report = record_round(model, synthetic_records(4), cfg, "cpu", limit=2,
                          rng=np.random.default_rng(0))
    assert report["sampled_rows"] == 2
    assert report["policy_entropy_network"] > 0
    assert report["kl_target_network"] > 0
    for head in VALUE_HEADS:
        assert f"prediction_saturation_{head}" in report
    assert record_round(model, blank(0), cfg, "cpu") == {}
