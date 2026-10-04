"""Momentum signSGD baseline.

This is the continuous-space version of the GDMC proposal: step every
coordinate by a fixed amount in -sign(momentum). It isolates how much of
GDMC's behaviour comes from the *discrete move* versus the sign proposal
itself, which is the mechanism control the GDMC study needs.
"""

from __future__ import annotations
from typing import Iterable, Optional
import torch


class SignSGD(torch.optim.Optimizer):
    """Momentum signSGD: p <- p - lr * sign(momentum)."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-2,
                 momentum: float = 0.9) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.requires_closure = False

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.data)
                buf = state["buf"]
                buf.mul_(mu).add_(p.grad)
                p.data.add_(torch.sign(buf), alpha=-lr)
        return loss


def make_signsgd(params: Iterable[torch.nn.Parameter], lr: float = 1e-2,
                 momentum: float = 0.9) -> SignSGD:
    return SignSGD(params, lr=lr, momentum=momentum)
