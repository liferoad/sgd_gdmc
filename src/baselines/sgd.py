"""Plain SGD baseline."""

from __future__ import annotations
from typing import Iterable
import torch


def make_sgd(params: Iterable[torch.nn.Parameter], lr: float = 0.01, **kwargs) -> torch.optim.SGD:
    """Vanilla SGD, no momentum."""
    return torch.optim.SGD(params, lr=lr, momentum=0.0, **kwargs)
