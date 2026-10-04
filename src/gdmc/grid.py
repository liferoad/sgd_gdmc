"""Grid quantization utilities for GDMC.

Two grid variants are supported:

* ``UniformGrid`` — K = 2**bits equally-spaced levels in [vmin, vmax].
  Symmetric around 0 by default (so the levels are
  {-L, ..., -delta, 0, +delta, ..., +L} for bits >= 2 and {-c, +c} for
  bits == 1, but in this project we never use bits == 1).

* ``AdaptiveGrid`` — per-tensor grid centred on the current value with
  spacing that shrinks over time. Implemented as a uniform grid whose
  range is rescaled to the current weight magnitude so resolution is
  preserved even as the weights shrink.

Both grids expose three methods needed by the optimizer:

* ``snap(w)`` — project a continuous tensor onto the grid (round-to-nearest).
* ``neighbour(w, direction)`` — return the grid point one step from ``w``
  in the sign of ``direction`` (+1 or -1). Used by the move-set.
* ``levels()`` — return the actual discrete value vector (for diagnostics).

The grids are designed to be cheap to call many times per step: they
hold no state per-tensor beyond the per-tensor range, so the per-call
cost is just a ``torch.round`` and a few clamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


def bits_to_levels(bits: int) -> int:
    """Number of grid points for ``bits`` quantization levels."""
    if bits < 1:
        raise ValueError("bits must be >= 1")
    return 2 ** bits


# Type alias used by the optimizer and baselines.
GridLike = "UniformGrid | AdaptiveGrid"


class UniformGrid:
    """Uniform K-level grid symmetric around 0.

    Parameters
    ----------
    bits : int
        Number of bits. K = 2**bits levels.
    vmin, vmax : float, optional
        Range of the grid. Defaults to ``(-1, 1)``. For binary weights
        the common convention is ``(-1, 1)``; for ``bits >= 2`` it can
        also be data-driven per-tensor (see ``per_tensor``).
    per_tensor : bool, default False
        If True, ``vmin`` / ``vmax`` are interpreted as multiples of the
        per-tensor max absolute value (e.g. ``vmin=-1, vmax=1`` means
        ``[-max|w|, +max|w|]`` for the current tensor). Useful when you
        want the grid to track the magnitude of the weights.
    """

    def __init__(self, bits: int, vmin: float = -1.0, vmax: float = 1.0,
                 per_tensor: bool = False) -> None:
        if bits < 1:
            raise ValueError("bits must be >= 1")
        if vmax <= vmin:
            raise ValueError("vmax must be > vmin")
        self.bits = bits
        self.levels = bits_to_levels(bits)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.per_tensor = bool(per_tensor)

    # --- core API ---

    def _range(self, w: torch.Tensor) -> tuple[float, float]:
        if self.per_tensor:
            m = float(w.detach().abs().max().item()) if w.numel() else 0.0
            # Guard against zero-tensor.
            if m == 0.0:
                m = 1.0
            return (self.vmin * m, self.vmax * m)
        return (self.vmin, self.vmax)

    def snap(self, w: torch.Tensor) -> torch.Tensor:
        """Round-to-nearest grid point, clamped to the grid range."""
        vmin, vmax = self._range(w)
        if self.levels == 1:
            return torch.full_like(w, 0.5 * (vmin + vmax))
        delta = (vmax - vmin) / (self.levels - 1)
        idx = torch.round((w - vmin) / delta)
        idx = idx.clamp(0, self.levels - 1)
        return vmin + idx * delta

    def neighbour(self, w: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """Return the grid point one step from ``w`` in direction ``sign``.

        ``sign`` is +1 / -1 (or 0, meaning no move). The result is the
        adjacent grid point, clipped to the grid range. We start from
        the snapped value so we always stay on the grid.

        Equivalent to ``step(w, sign, k=1)``; kept for backwards
        compatibility.
        """
        return self.step(w, sign, k=1)

    def step(self, w: torch.Tensor, sign: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Return the grid point ``k`` steps from ``w`` in direction ``sign``.

        ``sign`` is +1 / -1 (or 0, meaning no move). The result is the
        grid point that is ``k`` grid levels away in the sign direction,
        clipped to the grid range. ``k`` must be a non-negative
        integer; ``k=0`` returns the snapped value. ``k=1`` reproduces
        the original ``neighbour`` behaviour.

        The snap is done first, so the result is always on the grid
        regardless of whether ``w`` is.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        vmin, vmax = self._range(w)
        snapped = self.snap(w)
        if self.levels == 1 or k == 0:
            return snapped
        delta = (vmax - vmin) / (self.levels - 1)
        idx = torch.round((snapped - vmin) / delta)
        idx = (idx + k * sign.to(idx.dtype)).clamp(0, self.levels - 1)
        return vmin + idx * delta

    def step_multi(self, w: torch.Tensor, sign: torch.Tensor,
                   k_vec: torch.Tensor) -> torch.Tensor:
        """Vectorized multi-step: per-element step counts.

        k_vec has the same shape as w and holds a non-negative number of
        grid levels to move in the sign direction. Elements with k_vec=0
        do not move. Used by the magnitude-scaled / auto step modes.
        """
        vmin, vmax = self._range(w)
        snapped = self.snap(w)
        if self.levels == 1:
            return snapped
        delta = (vmax - vmin) / (self.levels - 1)
        idx = torch.round((snapped - vmin) / delta)
        idx = (idx + k_vec.to(idx.dtype) * sign.to(idx.dtype)).clamp(0, self.levels - 1)
        return vmin + idx * delta

    def values(self, w: torch.Tensor) -> torch.Tensor:
        """Return the discrete grid values for the given tensor shape."""
        vmin, vmax = self._range(w)
        if self.levels == 1:
            return torch.tensor([0.5 * (vmin + vmax)])
        delta = (vmax - vmin) / (self.levels - 1)
        return vmin + delta * torch.arange(self.levels, dtype=torch.float32)

    def __repr__(self) -> str:  # pragma: no cover
        return f"UniformGrid(bits={self.bits}, levels={self.levels}, vmin={self.vmin}, vmax={self.vmax}, per_tensor={self.per_tensor})"


class AdaptiveGrid:
    """Per-tensor uniform grid whose range tracks the weight magnitude.

    Concretely, on each call the grid is *rebuilt* with vmin = -alpha *
    max|w|, vmax = +alpha * max|w| where ``alpha`` is a small safety
    factor (>1) so weights that grow a bit are still representable.

    This is closer to a "k-means" style adaptive quantizer — the spacing
    shrinks as |w| shrinks — but it remains a *uniform* grid per call
    so that ``snap`` and ``neighbour`` stay cheap.

    NOTE: ``snap`` and ``neighbour`` derive the grid range from the
    *input* tensor. If you call them on a subset of the weights, the
    range may be wrong. The optimizer uses an explicit ``range_for``
    method to obtain the range from the *full* tensor first, then calls
    ``snap_range`` / ``neighbour_range`` with that range to operate on
    the subset. This keeps the grid consistent within a single step.
    """

    def __init__(self, bits: int, alpha: float = 1.05) -> None:
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.bits = bits
        self.levels = bits_to_levels(bits)
        self.alpha = float(alpha)

    def _range(self, w: torch.Tensor) -> tuple[float, float]:
        m = float(w.detach().abs().max().item()) if w.numel() else 0.0
        if m == 0.0:
            m = 1.0
        m *= self.alpha
        return (-m, m)

    def range_for(self, w: torch.Tensor) -> tuple[float, float]:
        """Public alias used by the optimizer to capture a range
        from the full tensor before operating on a subset."""
        return self._range(w)

    def snap_with_range(self, w: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
        if self.levels == 1:
            return torch.zeros_like(w)
        delta = (vmax - vmin) / (self.levels - 1)
        idx = torch.round((w - vmin) / delta)
        idx = idx.clamp(0, self.levels - 1)
        return vmin + idx * delta

    def neighbour_with_range(self, w: torch.Tensor, sign: torch.Tensor,
                             vmin: float, vmax: float) -> torch.Tensor:
        return self.step_with_range(w, sign, k=1, vmin=vmin, vmax=vmax)

    def step(self, w: torch.Tensor, sign: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Auto-range scalar multi-step (mirrors UniformGrid.step)."""
        vmin, vmax = self._range(w)
        return self.step_with_range(w, sign, k=k, vmin=vmin, vmax=vmax)

    def step_with_range(self, w: torch.Tensor, sign: torch.Tensor,
                        k: int = 1, vmin: float | None = None,
                        vmax: float | None = None) -> torch.Tensor:
        """Like :meth:`step` but with an explicit grid range.

        This is the version used by the optimizer (which captures the
        range from the *full* tensor up front).  ``k`` must be a
        non-negative integer; ``k=1`` reproduces the original
        ``neighbour_with_range`` behaviour.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        if vmin is None or vmax is None:
            vmin, vmax = self._range(w)
        if self.levels == 1 or k == 0:
            return self.snap_with_range(w, vmin, vmax)
        snapped = self.snap_with_range(w, vmin, vmax)
        delta = (vmax - vmin) / (self.levels - 1)
        idx = torch.round((snapped - vmin) / delta)
        idx = (idx + k * sign.to(idx.dtype)).clamp(0, self.levels - 1)
        return vmin + idx * delta

    def step_multi_with_range(self, w: torch.Tensor, sign: torch.Tensor,
                              k_vec: torch.Tensor, vmin: float,
                              vmax: float) -> torch.Tensor:
        """Vectorized multi-step with an explicit range (per-element k)."""
        if self.levels == 1:
            return self.snap_with_range(w, vmin, vmax)
        snapped = self.snap_with_range(w, vmin, vmax)
        delta = (vmax - vmin) / (self.levels - 1)
        idx = torch.round((snapped - vmin) / delta)
        idx = (idx + k_vec.to(idx.dtype) * sign.to(idx.dtype)).clamp(0, self.levels - 1)
        return vmin + idx * delta

    def step_multi(self, w: torch.Tensor, sign: torch.Tensor,
                   k_vec: torch.Tensor) -> torch.Tensor:
        """Auto-range vectorized multi-step."""
        vmin, vmax = self._range(w)
        return self.step_multi_with_range(w, sign, k_vec, vmin, vmax)

    # Backwards-compatible (auto-range) methods, used when the caller
    # doesn't need to keep the range consistent across a single step.
    def snap(self, w: torch.Tensor) -> torch.Tensor:
        vmin, vmax = self._range(w)
        return self.snap_with_range(w, vmin, vmax)

    def neighbour(self, w: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        vmin, vmax = self._range(w)
        return self.neighbour_with_range(w, sign, vmin, vmax)

    def values(self, w: torch.Tensor) -> torch.Tensor:
        vmin, vmax = self._range(w)
        if self.levels == 1:
            return torch.tensor([0.0])
        delta = (vmax - vmin) / (self.levels - 1)
        return vmin + delta * torch.arange(self.levels, dtype=torch.float32)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AdaptiveGrid(bits={self.bits}, levels={self.levels}, alpha={self.alpha})"


def make_grid(spec: str, bits: int, vmin: float = -1.0, vmax: float = 1.0) -> "UniformGrid | AdaptiveGrid":
    """Factory: ``spec`` in {"uniform", "adaptive"}."""
    spec = spec.lower()
    if spec == "uniform":
        return UniformGrid(bits=bits, vmin=vmin, vmax=vmax, per_tensor=False)
    if spec == "uniform-pt":
        return UniformGrid(bits=bits, vmin=vmin, vmax=vmax, per_tensor=True)
    if spec == "adaptive":
        return AdaptiveGrid(bits=bits, alpha=1.05)
    raise ValueError(f"unknown grid spec: {spec!r}")
