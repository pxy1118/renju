import numpy as np
import pytest
import torch

from vk.game import Game
from vk.network import (ARCHITECTURES, DEFAULT_ARCH, Evaluator, HybridNetwork,
                        LegacyNetwork, Network, Residual, TransformerBlock,
                        architecture, architecture_of, build, parameter_count)
from vk.search import MCTS


def sample_networks():
    return [Network(DEFAULT_ARCH), LegacyNetwork(), build("hybrid-64-3"), build("hybrid-8-1")]


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_architecture_builds_and_runs(arch):
    width, pattern, hybrid = architecture(arch)
    model = Network(arch)
    assert (model.width, model.pattern) == (width, pattern)
    assert model.blocks == len(pattern)
    assert model.transformer_blocks == pattern.count("T")
    assert isinstance(model, HybridNetwork if hybrid else LegacyNetwork)
    policy, value = model.eval()(torch.randn(2, 3, 15, 15))
    assert policy.shape == (2, 225)
    assert value.shape == (2,)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()
    assert (value.abs() <= 1).all()


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        Network("no-such-arch")
    with pytest.raises(ValueError, match="Unknown architecture"):
        architecture("no-such-arch")


def test_hybrid_layout_assembles_residuals_and_transformers_in_pattern_order():
    model = Network("hybrid-128-10")
    assert model.pattern == "RRTRRTRRTR"
    assert model.blocks == 10 and model.transformer_blocks == 3
    kinds = [type(block) for block in model.trunk[3:]]
    expected = [TransformerBlock if kind == "T" else Residual for kind in model.pattern]
    assert kinds == expected
    assert model.heads == 8
    assert [block.dim for block in model.trunk[3:] if isinstance(block, TransformerBlock)] == [16] * 3


def test_parameter_budget_is_stable():
    """Guard the capacity: an accidental change to the layout must be visible."""
    count = parameter_count(Network("hybrid-128-10"))
    assert 2_700_000 < count < 2_900_000, count
    assert count == 2_796_734


def test_single_position_inference_keeps_its_shape():
    """MCTS expands one leaf at a time, so batch size 1 must not collapse."""
    logits, value = Evaluator(Network(DEFAULT_ARCH).eval(), "cpu")(Game().encode())
    assert logits.shape == (225,)
    assert isinstance(value, float) and -1 <= value <= 1


def test_relative_position_bias_changes_the_output():
    """A position bias that is ignored would be a silent no-op."""
    torch.manual_seed(0)
    model = Network("hybrid-128-10").eval()
    block = next(item for item in model.trunk if isinstance(item, TransformerBlock))
    board = torch.randn(1, 3, 15, 15)
    with torch.no_grad():
        before = model(board)[0].clone()
        block.bias.copy_(torch.randn_like(block.bias))
        after = model(board)[0]
    assert not torch.allclose(before, after)
    assert block.bias.shape == (8, 841)


def test_attention_bias_uses_signed_2d_offsets():
    """A query at one corner must index a different slot than its opposite."""
    block = TransformerBlock(32, 2)
    assert block.attention_bias().shape == (1, 2, 225, 225)
    index = block.index
    assert index.shape == (225, 225)
    # Point p = (r, c) is the flat index r*15 + c, so the four corners are 0,
    # 14, 210 and 224. Slot = (dr + 14) * 29 + (dc + 14).
    assert index[0, 0].item() == 14 * 29 + 14          # offset (0, 0)
    assert index[0, 14].item() == 14 * 29 + 0          # offset (0, -14)
    assert index[0, 224].item() == 0                   # offset (-14, -14)
    assert index[224, 0].item() == 28 * 29 + 28        # offset (+14, +14)
    # Opposite offsets are separate parameters, so the bias is directional
    # rather than a symmetric distance embedding.
    assert index[0, 224].item() != index[224, 0].item()
    assert index[0, 14].item() != index[210, 0].item()


def test_no_cls_token_so_tokens_are_exactly_the_board():
    seen = {}

    class Recorder(TransformerBlock):
        def forward(self, x):
            seen["tokens"] = x.shape[-2] * x.shape[-1]
            return super().forward(x)

    block = Recorder(16, 1)
    block.eval()(torch.randn(1, 16, 15, 15))
    assert seen["tokens"] == 225


def test_narrow_architectures_keep_a_usable_head_count():
    assert Network("hybrid-8-1").heads == 1
    assert Network("hybrid-64-3").heads == 4
    model = Network("hybrid-8-1").eval()
    policy, value = model(torch.randn(3, 3, 15, 15))
    assert policy.shape == (3, 225) and value.shape == (3,)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()


def test_odd_width_is_rejected_with_a_clear_error():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build("hybrid-7-1")
    import vk.network as module
    module.ARCHITECTURES["broken-7-1"] = dict(width=7, pattern="R", hybrid=True)
    try:
        with pytest.raises(ValueError, match="must be even"):
            build("broken-7-1")
    finally:
        del module.ARCHITECTURES["broken-7-1"]


def test_mismatched_board_size_is_rejected():
    model = Network("hybrid-8-1").eval()
    with pytest.raises(ValueError, match="15x15 board"):
        model(torch.randn(1, 3, 19, 19))


def test_config_without_arch_resolves_to_the_legacy_network():
    """Checkpoints written before `arch` existed must still name an architecture."""
    assert architecture_of({"channels": 64, "blocks": 6}) == "legacy-64-6"
    assert architecture_of({"arch": "hybrid-128-10", "channels": 128, "blocks": 10}) == "hybrid-128-10"
    with pytest.raises(ValueError, match="No architecture matches"):
        architecture_of({"channels": 999, "blocks": 3})


def test_legacy_network_keeps_the_old_parameter_contract():
    """The legacy family exists so pre-hybrid checkpoints keep loading.

    These keys are the compatibility contract; changing them silently breaks
    every stored checkpoint, so assert them explicitly rather than trusting that
    nobody edits the class. ``runs/freestyle-hybrid/best.pt`` was verified
    against this exact key set.
    """
    model = LegacyNetwork()
    keys = set(model.state_dict())
    expected = {"trunk.0.weight", "trunk.1.weight", "trunk.1.bias",
                "policy.0.weight", "policy.0.bias", "policy.3.weight", "policy.3.bias",
                "value.0.weight", "value.0.bias", "value.3.weight", "value.3.bias",
                "value.5.weight", "value.5.bias"}
    assert expected <= keys
    # No hybrid-only parameters may leak into the legacy family.
    assert not [key for key in keys if "qkv" in key or "ln1" in key or "ln2" in key]
    # Six residual blocks sit at trunk positions 3..8 (0=conv, 1=bn, 2=relu).
    assert sorted({int(key.split(".")[1]) for key in keys if key.startswith("trunk.")}) == \
        [0, 1] + list(range(3, 9))
    assert sorted(key for key in keys if key.endswith("num_batches_tracked")) == \
        ["trunk.1.num_batches_tracked"] + \
        sorted(f"trunk.{index}.body.{position}.num_batches_tracked"
               for index in range(3, 9) for position in (1, 4))
    # The value head reads the flattened board, which is what the old head did.
    assert model.value[3].in_features == 225
    assert model.transformer_blocks == 0
    policy, value = model.eval()(torch.randn(1, 3, 15, 15))
    assert policy.shape == (1, 225) and value.shape == (1,)


def test_only_the_legacy_family_is_non_hybrid():
    hybrid = {name for name, entry in ARCHITECTURES.items() if entry["hybrid"]}
    assert "legacy-64-6" not in hybrid
    assert DEFAULT_ARCH in hybrid
    assert isinstance(Network(), HybridNetwork)


def test_forward_and_backward_reach_every_parameter():
    torch.set_num_threads(2)
    model = Network("hybrid-8-1")
    policy, value = model(torch.randn(2, 3, 15, 15))
    (policy.square().mean() + value.square().mean()).backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing == []


def test_search_and_training_accept_the_new_network():
    """The network must be usable through the real call paths, not just directly."""
    torch.set_num_threads(2)
    model = Network("hybrid-8-1").eval()
    game = Game()
    game.board[105:109] = 1
    policy = MCTS(Evaluator(model, "cpu"), 8).policy(game)
    assert policy.shape == (225,)
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)

    from collections import deque
    from vk.training import update
    replay = deque([(Game().encode(), np.full(225, 1 / 225), 1.0)], maxlen=32)
    metrics = update(Network("hybrid-8-1"), torch.optim.Adam(model.parameters(), lr=1e-3),
                     replay, 2, "cpu", np.random.default_rng(0))
    assert np.isfinite(metrics["loss"])
