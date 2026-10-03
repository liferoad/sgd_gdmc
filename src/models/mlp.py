"""Small MLP used for toy regression and MNIST."""

from __future__ import annotations
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple fully-connected network with ReLU activations."""

    def __init__(self, in_dim: int, hidden=(128, 128),
                 out_dim: int = 10, activation: str = "relu") -> None:
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self._keep_last_dim = False

    def forward(self, x):
        out = self.net(x)
        # For out_dim=1 we squeeze the trailing axis to make shape (B,)
        # match a (B,) target (so MSELoss/BCELoss work without broadcast
        # warnings). The default for out_dim > 1 is to leave (B, K).
        if out.shape[-1] == 1 and not self._keep_last_dim:
            out = out.squeeze(-1)
        return out

    def set_keep_last_dim(self, keep: bool) -> None:
        """Set whether to keep the trailing dim even when out_dim==1."""
        self._keep_last_dim = bool(keep)
