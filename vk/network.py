import numpy as np
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.body(x))


class Network(nn.Module):
    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(channels), nn.ReLU(),
                                   *[Residual(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(450, 225))
        self.value = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(225, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        x = self.trunk(x)
        return self.policy(x), self.value(x).squeeze(-1)


def device_check(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run 'uv sync --locked' and 'python main.py doctor'; CPU fallback is disabled.")
    x = torch.ones(16, 16, device=device, requires_grad=True)
    (x @ x).sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}


class Evaluator:
    def __init__(self, model, device):
        self.model, self.device = model, device

    def batch(self, states):
        self.model.eval()
        with torch.inference_mode():
            p, v = self.model(torch.from_numpy(np.stack(states)).to(self.device))
            return p.cpu().numpy(), v.cpu().numpy()

    def __call__(self, state):
        p, v = self.batch([state])
        return p[0], float(v[0])
