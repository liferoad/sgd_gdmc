"""Projected-GD baseline: any torch.optim.Optimizer + project to grid after each step.

This is the QAT-style "GD-then-project" baseline. We wrap a base
optimizer (typically Adam) and snap every parameter to the supplied
grid after each step().

Step-size pitfall
-----------------
If the base step is much smaller than the grid spacing delta, the snap
undoes the step and the weights never move.  At 4 bits delta = 2/15 =
0.133, so the historical lr = 1e-2 froze this baseline.  Pass
lr_scale to size the learning rate from the grid instead:

    ProjectedGD(params, grid=UniformGrid(4), lr_scale=0.5)
    -> lr = 0.5 * delta

Use grid_delta() to see the spacing for a grid.
"""

from __future__ import annotations
from typing import Iterable, Optional
import torch

from ..gdmc.grid import GridLike  # type: ignore


def grid_delta(grid: GridLike) -> Optional[float]:
    """Grid spacing for a fixed-range uniform grid, else None.

    Grids that derive their range per tensor (AdaptiveGrid, uniform-pt)
    have no single spacing, so callers must pass an explicit lr.
    """
    levels = getattr(grid, "levels", None)
    if levels is None or levels <= 1:
        return None
    if getattr(grid, "per_tensor", False):
        return None
    vmin = getattr(grid, "vmin", None)
    vmax = getattr(grid, "vmax", None)
    if vmin is None or vmax is None:
        return None
    return (float(vmax) - float(vmin)) / (levels - 1)


class ProjectedGD:
    """Wraps a base optimizer and projects to a grid after each step.

    Parameters
    ----------
    params : iterable
    base : {"sgd", "momentum", "adam"}
    grid : GridLike, optional
    lr : float, default 0.01
        Explicit learning rate.  Ignored when lr_scale is given.
    lr_scale : float, optional
        If given, lr = lr_scale * grid_delta(grid).  Required for grids
        with no fixed spacing unless an explicit lr is supplied.
    momentum, betas : forwarded to the base optimizer.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base: str = "sgd",
        grid: Optional[GridLike] = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        betas=(0.9, 0.999),
        lr_scale: Optional[float] = None,
    ) -> None:
        if grid is None:
            from ..gdmc.grid import make_grid  # type: ignore
            grid = make_grid("uniform", bits=4)
        self.grid = grid
        self.requires_closure = False
        if lr_scale is not None:
            delta = grid_delta(grid)
            if delta is None:
                raise ValueError(
                    "lr_scale was given but the grid has no fixed spacing; "
                    "pass an explicit lr instead"
                )
            lr = float(lr_scale) * delta
        if lr <= 0.0:
            raise ValueError("lr must be > 0 (or pass lr_scale)")
        self.lr = float(lr)
        self.lr_scale = lr_scale
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
                      base: str = "sgd", lr: float = 0.01,
                      lr_scale: Optional[float] = None) -> ProjectedGD:
    return ProjectedGD(params, base=base, grid=grid, lr=lr, lr_scale=lr_scale)
