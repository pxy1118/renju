"""The supervision objective, assembled in one place.

Every loss term, weight and mask the configuration asks for is computed here.
Training and pretraining both call it, so a weight cannot live in two modules
and drift apart, and a head with no valid target in a batch contributes nothing
instead of quietly pulling its parameters towards zero.

The policy target gets two terms. The hard one is the pruned search target, the
soft one is that target raised to 1/T: it keeps probability mass on the moves
the search also considered, which stops a policy head from collapsing onto one
move per position.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .config import value_weights
from .records import VALUE_HEADS, head_index, value_valid_mask

EPSILON = 1e-8


@dataclass(frozen=True)
class Objective:
    policy_weight: float
    policy_soft_weight: float
    policy_soft_temperature: float
    value_weights: tuple

    @classmethod
    def from_config(cls, cfg):
        return cls(policy_weight=float(cfg["policy_weight"]),
                   policy_soft_weight=float(cfg["policy_soft_weight"]),
                   policy_soft_temperature=float(cfg["policy_soft_temperature"]),
                   value_weights=tuple(float(weight) for weight in value_weights(cfg)))


def require_multi_head(model):
    """Fail loudly if a network cannot carry the schema's value heads.

    The objective supervises every head the schema declares. The legacy family
    exists only so pre-hybrid checkpoints keep loading, so training it is a
    mistake worth reporting instead of an index error inside the loss.
    """
    heads = tuple(getattr(model, "value_heads", ()))
    if heads != VALUE_HEADS:
        raise ValueError(f"Architecture {getattr(model, 'arch', '?')!r} exposes value heads "
                         f"{heads}, but the training objective needs {VALUE_HEADS}; "
                         f"use one of the hybrid architectures")
    return model


def soft_policy_target(policy, temperature):
    """The target raised to 1/T and renormalised, for a batch of rows."""
    temperature = max(float(temperature), 1e-6)
    scaled = policy.clamp(min=0.0) ** (1.0 / temperature)
    return scaled / scaled.sum(dim=1, keepdim=True).clamp(min=EPSILON)


def policy_entropy(probabilities):
    row = -(probabilities.clamp(min=EPSILON) * probabilities.clamp(min=EPSILON).log()).sum(1)
    return float(row.mean())


def compute_loss(logits, values, batch, cfg):
    """Total loss plus every per-term number a training record should carry.

    batch holds tensors: policy, policy_valid, policy_weight, value, value_valid.
    The value tensor is per head in schema order and value_valid is a boolean
    mask of the same shape.
    """
    objective = Objective.from_config(cfg)
    batch["policy_valid"] = batch["policy_valid"].to(torch.bool)
    batch["value_valid"] = batch["value_valid"].to(torch.bool)
    valid = batch["policy_valid"]
    probability = torch.softmax(logits.detach(), dim=1)
    report = {"policy_entropy": policy_entropy(probability),
              "policy_valid_share": float(valid.float().mean())}
    if valid.any():
        target = batch["policy"][valid]
        log_probability = F.log_softmax(logits[valid], dim=1)
        hard = (-(target * log_probability).sum(1))
        soft = (-(soft_policy_target(target, objective.policy_soft_temperature)
                  * log_probability).sum(1))
        row_weight = batch["policy_weight"][valid]
        weighted = row_weight * (hard + objective.policy_soft_weight * soft)
        policy_loss = weighted.sum() / row_weight.sum().clamp(min=EPSILON)
        report["policy_target_entropy"] = policy_entropy(target)
        report["policy_ce_hard"] = float(hard.detach().mean())
        report["policy_ce_soft"] = float(soft.detach().mean())
    else:
        policy_loss = logits.sum() * 0.0
    value_loss = logits.sum() * 0.0
    for index, head in enumerate(VALUE_HEADS):
        weight = objective.value_weights[index]
        mask = batch["value_valid"][:, index]
        prediction = values[:, index]
        report[f"value_abs_mean_{head}"] = float(prediction.detach().abs().mean())
        report[f"value_abs_gt_0.9_share_{head}"] = float(
            (prediction.detach().abs() > 0.9).float().mean())
        report[f"value_valid_share_{head}"] = float(mask.float().mean())
        if weight <= 0 or not mask.any():
            report[f"value_loss_{head}"] = None
            continue
        delta = prediction[mask] - batch["value"][mask, index]
        term = delta.square().mean()
        report[f"value_loss_{head}"] = float(term.detach())
        report[f"value_mae_{head}"] = float(delta.detach().abs().mean())
        value_loss = value_loss + weight * term
    total = objective.policy_weight * policy_loss + value_loss
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite training loss")
    report.update({"loss": total, "policy_loss": policy_loss, "value_loss": value_loss,
                   "policy_weight": objective.policy_weight,
                   "value_weight": objective.value_weights[head_index("final")]})
    return report


def tensor_batch(batch, device, states, policies):
    """Tensors for one augmented batch of position records."""
    value_valid = value_valid_mask(batch)
    return {
        "state": torch.tensor(states, dtype=torch.float32, device=device),
        "policy": torch.tensor(policies, dtype=torch.float32, device=device),
        "policy_valid": torch.tensor(batch["policy_valid"] != 0, device=device),
        "policy_weight": torch.tensor(batch["policy_weight"].astype("float32"), device=device),
        "value": torch.tensor(batch["value"].astype("float32"), device=device),
        "value_valid": torch.tensor(value_valid, dtype=torch.bool, device=device),
    }
