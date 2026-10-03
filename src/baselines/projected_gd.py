"""Projected-GD baseline: any torch.optim.Optimizer + project to grid after each step.

This is the QAT-style "GD-then-project" baseline. We wrap a base
optimizer (typically SGD) and snap every parameter to the supplied
grid after each ``.step()``.
"""

from __future__ import annotations
from typing import Iterable, Optional
import torch

from ..gdmc.grid import GridLike  # type: ignore


class ProjectedGD:
    """Wraps a base optimizer and projects to a grid after each step.

    Parameters
    ----------
    params : iterable
    base : str or torch.optim.Optimizer class
        Which base optimizer to use. "sgd" / "momentum" / "adam".
    grid : GridLike
        Quantization grid (UniformGrid or AdaptiveGrid).
    lr, momentum, betas : float
        Hyperparameters forwarded to the base optimizer.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base: str = "sgd",
        grid: Optional[GridLike] = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        betas=(0.9, 0.999),
    ) -> None:
        if grid is None:
            from ..gdmc.grid import make_grid  # type: ignore
            grid = make_grid("uniform", bits=4)
        self.grid = grid
        self.requires_closure = False
        base = base.lower()
        if base == "sgd":
            self.inner = torch.optim.SGD(params, lr=lr, momentum=0.0)
        elif base in ("momentum", "momentum-sgd"):
            self.inner = torch.optim.SGD(params, lr=lr, momentum=momentum)
        elif base == "adam":
            self.inner = torch.optim.Adam(params, lr=lr, betas=betas)
        else:
            raise ValueError(f"unknown base optimizer: {base!r}")

    @torch.no_grad()
    def project(self) -> None:
        for group in self.inner.param_groups:
            for p in group["params"]:
                if p.data is None:
                    continue
                p.data.copy_(self.grid.snap(p.data))

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None:
            loss = closure()
        else:
            loss = None
        self.inner.step()
        self.project()
        return loss

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, sd):
        self.inner.load_state_dict(sd)

    @property
    def param_groups(self):
        return self.inner.param_groups


def make_projected_gd(params: Iterable[torch.nn.Parameter], grid: GridLike,
                      base: str = "sgd", lr: float = 0.01) -> ProjectedGD:
    return ProjectedGD(params, base=base, grid=grid, lr=lr)
