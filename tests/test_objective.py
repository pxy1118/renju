"""The objective is where the training signal is actually defined."""
import numpy as np
import pytest
import torch

from vk.config import DEFAULTS
from vk.objective import Objective, compute_loss, soft_policy_target, tensor_batch
from vk.records import VALUE_HEADS, blank, set_head


def make_batch(rows=4, heads=("final",), policy_valid=True, policy_weight=1.0,
               target_policy=None, value=0.5):
    records = blank(rows)
    for index in range(rows):
        records["policy"][index] = (np.full(225, 1 / 225, np.float16)
                                    if target_policy is None else target_policy)
        records["policy_valid"][index] = 1 if policy_valid else 0
        records["policy_weight"][index] = policy_weight
    for head in heads:
        set_head(records, head, np.full(rows, float(value), np.float32))
    states = np.zeros((rows, 3, 15, 15), np.uint8)
    tensors = tensor_batch(records, "cpu", states, records["policy"].astype(np.float32))
    return records, tensors


def outputs(rows=4, heads=3, logits=0.0, values=None):
    policy = torch.full((rows, 225), float(logits))
    if values is None:
        values = torch.full((rows, heads), 0.5)
    else:
        values = torch.tensor(values, dtype=torch.float32)
    return policy, values


def test_the_soft_target_is_the_target_to_a_power():
    policy = torch.tensor([[0.8, 0.15, 0.05]], dtype=torch.float32)
    softened = soft_policy_target(policy, 2.0)
    assert float(softened.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(softened[0, 0]) < 0.8 and float(softened[0, 2]) > 0.05
    assert float(soft_policy_target(policy, 1.0)[0, 0]) == pytest.approx(0.8, abs=1e-6)


def test_only_valid_heads_contribute_to_the_value_loss():
    cfg = dict(DEFAULTS)
    _, tensors = make_batch(heads=("final",), value=1.0)
    policy, values = outputs(rows=4, values=[[0.5, 0.0, 0.0]] * 4)
    report = compute_loss(policy, values, tensors, cfg)
    assert report["value_loss_final"] == pytest.approx(0.25, abs=1e-5)
    # The horizon heads are invalid in this batch, so they train nothing at all
    # rather than being pulled towards zero.
    assert report["value_loss_mid"] is None and report["value_loss_short"] is None
    assert report["value_valid_share_mid"] == 0.0
    assert float(report["value_loss"]) == pytest.approx(0.25, abs=1e-5)


def test_a_head_with_no_valid_rows_does_not_dilute_the_others():
    cfg = dict(DEFAULTS, value_weight_mid=1000.0)
    _, tensors = make_batch(heads=("final",))
    policy, values = outputs(rows=4, values=[[0.0, 0.0, 0.0]] * 4)
    report = compute_loss(policy, values, tensors, cfg)
    assert float(report["value_loss"]) == pytest.approx(0.25, abs=1e-5)


def test_head_weights_scale_their_own_term():
    cfg = dict(DEFAULTS, value_weight_mid=2.0)
    _, tensors = make_batch(heads=("final", "mid"), value=0.0)
    policy, values = outputs(rows=4, values=[[0.0, 1.0, 0.0]] * 4)
    report = compute_loss(policy, values, tensors, cfg)
    assert report["value_loss_final"] == pytest.approx(0.0, abs=1e-6)
    assert float(report["value_loss"]) == pytest.approx(2.0, abs=1e-5)


def test_unsupervised_rows_do_not_touch_the_policy_gradient():
    cfg = dict(DEFAULTS, policy_soft_weight=0.0, value_weight_final=0.0,
               value_weight_mid=0.0, value_weight_short=0.0)
    _, tensors = make_batch(policy_valid=False)
    policy, values = outputs()
    policy.requires_grad_(True)
    report = compute_loss(policy, values, tensors, cfg)
    assert report["policy_valid_share"] == 0.0
    assert float(report["policy_loss"].detach()) == 0.0
    report["loss"].backward()
    assert float(policy.grad.abs().sum()) == 0.0, "no valid row means no gradient"


def test_the_policy_term_is_a_weighted_mean_over_valid_rows():
    cfg = dict(DEFAULTS, policy_soft_weight=0.0)
    target = np.zeros(225, np.float16)
    target[0] = 1.0
    _, tensors = make_batch(rows=2, target_policy=target, policy_weight=0.5)
    policy, values = outputs(rows=2, logits=0.0)
    report = compute_loss(policy, values, tensors, cfg)
    assert float(report["policy_loss"]) == pytest.approx(np.log(225), abs=1e-4)


def test_a_non_finite_loss_fails_loudly():
    cfg = dict(DEFAULTS)
    _, tensors = make_batch()
    policy = torch.full((4, 225), float("nan"))
    with pytest.raises(RuntimeError, match="Non-finite"):
        compute_loss(policy, torch.zeros(4, 3), tensors, cfg)


def test_a_single_head_family_is_refused_by_name_not_by_index_error():
    from vk.network import Network
    from vk.objective import require_multi_head
    assert require_multi_head(Network("hybrid-8-1")).value_heads == VALUE_HEADS
    with pytest.raises(ValueError, match="needs"):
        require_multi_head(Network("legacy-64-6"))


def test_the_objective_reads_every_weight_from_the_configuration():
    objective = Objective.from_config(dict(DEFAULTS, policy_weight=2.0,
                                           policy_soft_weight=0.0,
                                           value_weight_final=3.0,
                                           value_weight_mid=0.25,
                                           value_weight_short=0.0))
    assert len(objective.value_weights) == len(VALUE_HEADS)
    assert objective.policy_weight == 2.0 and objective.value_weights[0] == 3.0
