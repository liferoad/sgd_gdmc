"""Synthetic 1-D regression dataset.

A small nonlinear function ``y = sin(2*pi*x) + 0.3 * sin(4*pi*x)``
evaluated on a uniform grid in [0, 1] with Gaussian noise added.
``N`` is small (default 200) so the loss landscape is dominated by the
function and the optimization problem is well-posed.

The dataset lives entirely in memory and is fast to iterate, so it's
ideal as a sanity check that GDMC, projected-GD, and continuous
baselines all reach similar minima.
"""

from __future__ import annotations
import math
import torch
from torch.utils.data import Dataset


def make_toy(N: int = 200, noise: float = 0.05, seed: int = 0,
             in_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x, y)`` with x of shape (N, in_dim) and y of shape (N,)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(N, in_dim, generator=g)
    y = torch.sin(2 * math.pi * x.sum(dim=1)) + 0.3 * torch.sin(4 * math.pi * x.sum(dim=1))
    y = y + noise * torch.randn(N, generator=g)
    return x, y


class ToyRegression(Dataset):
    def __init__(self, N: int = 200, noise: float = 0.05, seed: int = 0,
                 in_dim: int = 1) -> None:
        self.x, self.y = make_toy(N, noise, seed, in_dim)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
