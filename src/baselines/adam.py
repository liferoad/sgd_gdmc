"""Adam baseline."""

from __future__ import annotations
from typing import Iterable
import torch


def make_adam(params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
              betas=(0.9, 0.999), eps: float = 1e-8, **kwargs) -> torch.optim.Adam:
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, **kwargs)
