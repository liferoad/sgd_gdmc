"""Stochastic Gradient Langevin Dynamics (SGLD) baseline.

Implements the standard SGLD update from Welling & Teh (2011):

    theta_{t+1} = theta_t - lr * grad + sqrt(2 * lr / beta) * eps,  eps ~ N(0, I)

The temperature ``beta`` controls the noise scale. We use the same
Metropolis acceptance idea from the paper for a "Metropolis-adjusted
SGLD" (i.e. SGLD + accept/reject) so this baseline is the natural
continuous-space cousin of GDMC.

For simplicity we expose a plain SGLD (no accept/reject) — that is what
is normally meant by SGLD and is what the Bayesian-deep-learning
literature uses.
"""

from __future__ import annotations
from typing import Iterable, Optional
import math
import torch


class SGLD(torch.optim.Optimizer):
    """SGLD as a torch.optim.Optimizer.

    Parameters
    ----------
    params : iterable
    lr : float, default 1e-3
    beta : float, default 1.0
        Inverse temperature controlling the noise scale; smaller beta
        -> larger noise.
    weight_decay : float, default 0.0
    rng : torch.Generator, optional
    """

    def __init__(self, params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3, beta: float = 1.0,
                 weight_decay: float = 0.0,
                 rng: Optional[torch.Generator] = None) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if beta <= 0.0:
            raise ValueError("beta must be > 0")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rng = rng
        # SGLD does not strictly *need* a closure, but it can re-evaluate
        # the loss on the proposed point if you give it one. We support
        # both styles.
        self.requires_closure = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            noise_scale = math.sqrt(2.0 * lr / beta)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p.data, alpha=wd)
                # noise: standard Gaussian on the parameter device.
                if self.rng is None:
                    noise = torch.randn_like(p.data)
                else:
                    noise = torch.randn(p.data.shape, generator=self.rng, dtype=p.data.dtype, device=p.data.device)
                p.data.add_(g, alpha=-lr)
                p.data.add_(noise, alpha=noise_scale)
        return loss


def make_sgld(params: Iterable[torch.nn.Parameter], lr: float = 1e-3,
              beta: float = 1.0, weight_decay: float = 0.0) -> SGLD:
    return SGLD(params, lr=lr, beta=beta, weight_decay=weight_decay)
