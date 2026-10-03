"""Momentum SGD baseline."""

from __future__ import annotations
from typing import Iterable
import torch


def make_momentum(params: Iterable[torch.nn.Parameter], lr: float = 0.01,
                  momentum: float = 0.9, **kwargs) -> torch.optim.SGD:
    """SGD with heavy-ball momentum (PyTorch's SGD)."""
    return torch.optim.SGD(params, lr=lr, momentum=momentum, **kwargs)
