"""Residual + Transformer hybrid network, selected by architecture name.

ARCHITECTURES is the single source of truth for every layout: a width, a block
pattern string of R (residual) and T (transformer) characters, and the value
heads that architecture exposes. The block count and the number of transformer
blocks are derived from the pattern, so a name and the layers actually built
can never disagree.

The value head is multi-scale. final targets the game result, mid and short
target the outcome over a horizon of play with a search-value bootstrap, which
is what keeps sibling positions distinguishable instead of every leaf reading
plus or minus one. The final head keeps the historical parameter names, so a
checkpoint written before the horizon heads existed loads with them
initialised from it.

legacy-64-6 reproduces the original all-convolutional network exactly, down to
its parameter names and its single value head, so checkpoints written before
the hybrid existed still load. It is versioned by that contract: do not
"improve" it, add a new entry.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SIZE = 15
POINTS = SIZE * SIZE
PLANES = 3
HEADS = 8
HEAD_DIM = 16
MLP_RATIO = 4
DROPOUT = 0.0
SPAN = 2 * SIZE - 1          # relative offsets run from -(SIZE-1) to +(SIZE-1)
LEGACY_ARCH = "legacy-64-6"
VALUE_HEADS = ("final", "mid", "short")
VALUE_HEAD_WIDTH = 64

ARCHITECTURES = {
    # 7 residual blocks and 3 transformer blocks, in the order given.
    "hybrid-128-10": dict(width=128, pattern="RRTRRTRRTR", hybrid=True,
                          value_heads=VALUE_HEADS),
    # The original network: plain residual stack with a single value head, kept
    # bit-compatible with the checkpoints it produced.
    LEGACY_ARCH: dict(width=64, pattern="RRRRRR", hybrid=False, value_heads=("final",)),
    # Small hybrid layouts for tests and for a cheap end-to-end smoke run.
    "hybrid-64-3": dict(width=64, pattern="RTR", hybrid=True, value_heads=VALUE_HEADS),
    "hybrid-8-1": dict(width=8, pattern="R", hybrid=True, value_heads=VALUE_HEADS),
}
DEFAULT_ARCH = "hybrid-128-10"


def architecture(name):
    """Resolve an architecture name to its width, block pattern and family."""
    try:
        entry = ARCHITECTURES[name]
    except KeyError:
        raise ValueError(f"Unknown architecture: {name!r} "
                         f"(known: {sorted(ARCHITECTURES)})") from None
    pattern = entry["pattern"]
    if not pattern or set(pattern) - {"R", "T"}:
        raise ValueError(f"Invalid block pattern for {name!r}: {pattern!r}")
    return entry["width"], pattern, entry["hybrid"]


def value_heads(arch=DEFAULT_ARCH):
    """The value heads an architecture exposes, in schema order.

    A pre-refactor checkpoint has one value head, so its state dict can be
    upgraded by copying that head into the horizon heads of the same network.
    """
    try:
        return tuple(ARCHITECTURES[arch]["value_heads"])
    except KeyError:
        raise ValueError(f"Unknown architecture: {arch!r} "
                         f"(known: {sorted(ARCHITECTURES)})") from None


def final_head_index(model_or_heads):
    heads = getattr(model_or_heads, "value_heads", model_or_heads)
    return tuple(heads).index("final")


def heads_for(width):
    """As many 16-wide heads as the width allows, at least one."""
    return max(1, min(HEADS, width // HEAD_DIM))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def value_head(width):
    """One tanh-bounded scalar head; the pooled trunk is its input."""
    return nn.Sequential(nn.Linear(width, VALUE_HEAD_WIDTH), nn.ReLU(),
                         nn.Linear(VALUE_HEAD_WIDTH, 1), nn.Tanh())


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.body(x))


class TransformerBlock(nn.Module):
    """Pre-LN transformer over the 225 board points, with a 2D relative bias.

    There is no CLS token: the tokens *are* the board points, so the policy head
    can keep reading per-point features from the same grid shape.
    """
    def __init__(self, channels, heads, mlp_ratio=MLP_RATIO, dropout=DROPOUT):
        super().__init__()
        self.channels, self.heads = channels, heads
        self.dim = channels // heads
        self.ln1, self.ln2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, channels),
                                 nn.Dropout(dropout))
        # One table per head, indexed by (drow + SIZE-1) * SPAN + (dcol + SIZE-1).
        self.bias = nn.Parameter(torch.zeros(heads, SPAN * SPAN))
        rows, cols = np.divmod(np.arange(POINTS), SIZE)
        index = ((rows[:, None] - rows[None, :] + SIZE - 1) * SPAN
                 + (cols[:, None] - cols[None, :] + SIZE - 1))
        self.register_buffer("index", torch.from_numpy(index), persistent=False)

    def attention_bias(self):
        """[1, heads, 225, 225] additive bias, broadcast over query batch."""
        return self.bias[:, self.index.reshape(-1)].reshape(1, self.heads, POINTS, POINTS)

    def forward(self, x):
        if x.shape[-2:] != (SIZE, SIZE):
            raise ValueError(f"Expected a {SIZE}x{SIZE} board, got {tuple(x.shape[-2:])}")
        batch = x.shape[0]
        tokens = x.flatten(2).transpose(1, 2)                       # [B, 225, C]

        normed = self.ln1(tokens)
        q, k, v = self.qkv(normed).chunk(3, dim=-1)
        shape = (batch, POINTS, self.heads, self.dim)
        q, k, v = (part.reshape(shape).transpose(1, 2) for part in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attention_bias())
        attended = attended.transpose(1, 2).reshape(batch, POINTS, self.channels)
        tokens = tokens + self.proj(attended)

        tokens = tokens + self.mlp(self.ln2(tokens))
        return tokens.transpose(1, 2).reshape(batch, self.channels, SIZE, SIZE)


class HybridNetwork(nn.Module):
    """Shared trunk assembled from the architecture block pattern.

    The final value head keeps the parameter names of the single-head network
    (value.0.*, value.2.*), which is what lets a pre-refactor checkpoint load
    with the horizon heads seeded from it.
    """
    def __init__(self, arch):
        super().__init__()
        width, pattern, hybrid = architecture(arch)
        if not hybrid:
            raise ValueError(f"{arch!r} is not a hybrid architecture")
        if width % 2:
            raise ValueError(f"Architecture width must be even, got {width}")
        blocks = [Residual(width) if kind == "R" else TransformerBlock(width, heads_for(width))
                  for kind in pattern]
        self.arch, self.pattern, self.width = arch, pattern, width
        self.blocks = len(blocks)
        self.transformer_blocks = pattern.count("T")
        self.heads = heads_for(width)
        self.value_heads = value_heads(arch)
        self.trunk = nn.Sequential(nn.Conv2d(PLANES, width, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(width), nn.ReLU(), *blocks)
        self.policy = nn.Sequential(nn.Conv2d(width, 2, 1), nn.ReLU(), nn.Flatten(),
                                    nn.Linear(2 * POINTS, POINTS))
        # Pool the trunk before reducing it: a 1x1 conv first would collapse the
        # channels and leave nothing to pool over the board.
        self.value = value_head(width)
        for head in self.value_heads:
            if head != "final":
                setattr(self, f"value_{head}", value_head(width))

    def head_module(self, head):
        if head not in self.value_heads:
            raise ValueError(f"Architecture {self.arch!r} has no value head {head!r} "
                             f"(heads: {list(self.value_heads)})")
        return getattr(self, "value" if head == "final" else f"value_{head}")

    def forward(self, x):
        if x.shape[-2:] != (SIZE, SIZE):
            # Checked here rather than inside the transformer so any board size
            # mismatch fails with one clear message instead of a shape error
            # buried in whichever head happens to run first.
            raise ValueError(f"Expected a {SIZE}x{SIZE} board, got {tuple(x.shape[-2:])}")
        x = self.trunk(x)
        pooled = x.mean(dim=(2, 3))
        values = torch.cat([self.head_module(head)(pooled) for head in self.value_heads], dim=1)
        return self.policy(x), values


class LegacyNetwork(nn.Module):
    """The original 64x6 convolutional network, retained byte-for-byte.

    Every parameter name and every head matches the checkpoints produced before
    the hybrid existed, which is what lets a stored legacy checkpoint keep
    loading. Keep the shape of this class frozen: it has one value head, so its
    output is a single column.
    """
    def __init__(self, arch=LEGACY_ARCH):
        super().__init__()
        width, pattern, hybrid = architecture(arch)
        if hybrid:
            raise ValueError(f"{arch!r} is not a legacy architecture")
        blocks = [Residual(width) for _ in pattern]
        self.arch, self.pattern, self.width = arch, pattern, width
        self.blocks = len(blocks)
        self.transformer_blocks = 0
        self.heads = heads_for(width)
        self.value_heads = value_heads(arch)
        self.trunk = nn.Sequential(nn.Conv2d(PLANES, width, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(width), nn.ReLU(), *blocks)
        self.policy = nn.Sequential(nn.Conv2d(width, 2, 1), nn.ReLU(), nn.Flatten(),
                                    nn.Linear(2 * POINTS, POINTS))
        self.value = nn.Sequential(nn.Conv2d(width, 1, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(POINTS, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        x = self.trunk(x)
        return self.policy(x), self.value(x)


def build(arch=DEFAULT_ARCH):
    """Instantiate the network family the architecture name selects."""
    _, _, hybrid = architecture(arch)
    return (HybridNetwork if hybrid else LegacyNetwork)(arch)


def Network(arch=DEFAULT_ARCH):
    """Default factory; build is the same thing and reads better at call sites."""
    return build(arch)


def architecture_of(config):
    """The architecture a stored config describes.

    Checkpoints written before arch existed only carry channels and blocks;
    resolve those back to a name so resuming an old run reports a real
    configuration difference instead of a wall of missing state-dict keys.
    """
    if config.get("arch"):
        return config["arch"]
    for name, entry in ARCHITECTURES.items():
        if entry["width"] == config.get("channels") and len(entry["pattern"]) == config.get("blocks"):
            return name
    raise ValueError(f"No architecture matches channels={config.get('channels')} "
                     f"blocks={config.get('blocks')}")


def device_check(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run 'uv sync --locked' and 'python main.py doctor'; CPU fallback is disabled.")
    x = torch.ones(16, 16, device=device, requires_grad=True)
    (x @ x).sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}


@dataclass
class Inference:
    """One model call: every policy logit and every value head.

    The search needs all heads to form its leaf value, while most callers only
    want the win value. Keeping both in one object means no caller has to guess
    what a bare second return value means.
    """
    policy: np.ndarray
    values: np.ndarray
    heads: tuple
    mix: np.ndarray

    @classmethod
    def leaf_value(cls, policy, value):
        """An inference whose leaf value was mixed elsewhere.

        Self-play actors receive one already-mixed value per state over the
        pipe, so they build this instead of a full multi-head result.
        """
        return cls(np.asarray(policy), np.array([float(value)]), ("leaf",), np.array([1.0]))

    def head(self, name="final"):
        return self.values[..., tuple(self.heads).index(name)]

    def leaf(self):
        return self.values @ self.mix


class Evaluator:
    """Batch inference, plus the leaf value the search should use."""

    def __init__(self, model, device, mix=None):
        self.model, self.device = model, device
        self.heads = tuple(model.value_heads)
        if mix is None:
            mix = np.eye(len(self.heads))[self.heads.index("final")]
        self.mix = np.asarray(mix, np.float64)
        if self.mix.shape != (len(self.heads),):
            raise ValueError(f"Leaf value mix must have {len(self.heads)} entries "
                             f"for heads {list(self.heads)}")

    def batch(self, states):
        self.model.eval()
        with torch.inference_mode():
            policy, values = self.model(torch.from_numpy(np.stack(states)).to(self.device))
            return Inference(policy.cpu().numpy(), values.cpu().numpy(), self.heads, self.mix)

    def __call__(self, state):
        result = self.batch([state])
        return Inference(result.policy[0], result.values[0], self.heads, self.mix)

    def leaf(self, states):
        """Just the mixed leaf value, batched, for callers that need nothing else."""
        return self.batch(states).leaf()
