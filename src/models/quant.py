"""Latent-weight quantization-aware training (QAT) with a straight-through
estimator.

This is the "practical" low-bit baseline the GDMC comparison needs: weights
are stored in full precision (latent weights) and quantized in the forward
pass, so gradients flow through the quantizer by the straight-through
estimator. At evaluation time the latent weights are snapped to the real grid.

It is deliberately a different scheme from GDMC: GDMC moves the *deployed*
weights on the grid and keeps no latent copy.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gdmc.grid import UniformGrid


class QuantLinearSTE(nn.Module):
    """Linear layer whose forward pass uses grid-quantized weights (STE)."""

    def __init__(self, in_features: int, out_features: int, grid: UniformGrid,
                 bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight
        vmin, vmax = self.grid.range_for(w)
        wq = self.grid.snap_with_range(w, vmin, vmax)
        # Straight-through: forward uses wq, backward sees d/dw.
        w_ste = w + (wq - w).detach()
        return F.linear(x, w_ste, self.linear.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


class QuantMLP(nn.Module):
    """MLP whose Linear weights are quantized in the forward pass."""

    def __init__(self, in_dim: int = 784, hidden=(256, 256), out_dim: int = 10,
                 bits: int = 4, vmin: float = -1.0, vmax: float = 1.0) -> None:
        super().__init__()
        grid = UniformGrid(bits=bits, vmin=vmin, vmax=vmax)
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(QuantLinearSTE(prev, h, grid))
            layers.append(nn.ReLU())
            prev = h
        layers.append(QuantLinearSTE(prev, out_dim, grid))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
