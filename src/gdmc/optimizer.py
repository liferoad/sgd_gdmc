"""GDMC optimizer for discretized deep-learning weight optimization.

The optimizer implements the algorithm from

    Hu, Beratan, Yang. "A gradient-directed Monte Carlo method for global
    optimization in a discrete space: Application to protein sequence
    design and folding." J. Chem. Phys. 131, 154117 (2009).

adapted to neural-network weight optimization. We follow the paper's
two-step recipe:

1.  **Compute the gradient** of the loss w.r.t. the (continuous)
    weights via ordinary backprop. This is the gradient of the
    "virtual continuous surface" introduced by the discretization.

2.  **Propose a discrete move**. We pick a fraction ``move_frac`` of
    weights (the *move set*) and for each one, step one grid point in
    the direction of ``-sign(grad)`` using
    ``Grid.neighbour(w, sign)``. This is the gradient-directed
    proposal — equivalent to the per-site / per-direction move in the
    paper.

3.  **Accept or reject the move** with the Metropolis criterion
    ``p = min{1, exp(-beta * (L_new - L_old))}`` evaluated on the
    **discretized** weights. ``beta`` is a temperature (larger = more
    greedy). The Metropolis acceptance lets the search climb out of
    local minima that pure gradient descent would get stuck in.

The optimizer follows the ``torch.optim.Optimizer`` API so it can be
dropped into a normal training loop. We do, however, require a custom
``step`` that also accepts a *closure* returning the loss — exactly
like ``torch.optim.LBFGS`` — because the acceptance test needs to
evaluate the loss again on the proposed weights.

Usage
-----

>>> optimizer = GDMCOptimizer(model.parameters(), grid=grid, beta=2.0, ...)
>>> for x, y in loader:
...     def closure():
...         optimizer.zero_grad()
...         loss = criterion(model(x), y)
...         loss.backward()
...         return loss
...     optimizer.step(closure)
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional
import math
import torch

from .grid import make_grid, GridLike  # type: ignore


class GDMCOptimizer(torch.optim.Optimizer):
    """Gradient-Directed Monte Carlo optimizer on a discretized weight grid.

    Parameters
    ----------
    params : iterable
        Model parameters (will be quantized in-place each step).
    grid : GridLike, optional
        Quantization grid. If None, a uniform 4-bit grid in [-1, 1] is
        used. All parameters share the same grid by default; pass a
        list of grids for per-parameter behaviour.
    beta : float, default 2.0
        Metropolis inverse-temperature. Larger = greedier (closer to
        gradient descent). Paper used 1.2e-3 (protein design) and 2.4
        (protein folding); the magnitude of the loss differs, so we
        expose ``beta`` for tuning.
    move_frac : float, default 0.01
        Fraction of weights to mutate per step. ``1.0`` proposes a move
        for every weight at once (one giant step); ``0.001`` proposes a
        move for a small random subset (slower but more like the paper's
        per-site proposals).
    accept_on : {"minibatch", "full"}, default "minibatch"
        Which loss to use for the Metropolis acceptance test. "minibatch"
        uses the *current* minibatch loss (cheap but noisy). "full"
        uses the *full* training set (expensive but unbiased). For
        DL we default to "minibatch" — the noise acts as an implicit
        regularizer and matches SGLD's behaviour.
    beta1 : float, default 0.0
        First-moment momentum on the gradient, à la Adam. ``0.0``
        (the default) means no momentum — the proposal direction is
        ``-sign(g)`` and the optimizer is bit-exactly equivalent to
        v1. Set to e.g. ``0.9`` to use ``m_t = beta1 * m_{t-1} + (1 - beta1) * g_t``
        and propose moves in ``-sign(m_t)``.
    k : int, default 1
        Number of *grid points* to step per move (the original
        behaviour). ``k > 1`` takes larger discrete steps in the
        descent direction, which is needed at fine grids where one
        step is too small to make progress in a short training run.
    rng : torch.Generator, optional
        Random source for the move-set selection. Default uses the
        global generator.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        grid: Optional[GridLike] = None,
        bits: int = 4,
        grid_spec: str = "uniform",
        beta: float = 2.0,
        move_frac: float = 0.01,
        accept_on: str = "minibatch",
        beta1: float = 0.0,
        k: int = 1,
        k_mode: str = "fixed",
        step_scale: float = 1e-3,
        k_max: int = 1 << 22,
        grad_ema: float = 0.99,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        if grid is None:
            grid = make_grid(grid_spec, bits=bits)
        if k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        # Normalize to a list of grids, one per param group.
        defaults = dict(
            grid=grid,
            beta=beta,
            move_frac=move_frac,
            accept_on=accept_on,
            beta1=beta1,
            k=k,
            k_mode=k_mode,
            step_scale=step_scale,
            k_max=k_max,
            grad_ema=grad_ema,
        )
        super().__init__(params, defaults)
        self.rng = rng
        # Flag for the training loop: closure-style step.
        self.requires_closure = True
        # Diagnostic accumulators (read by the training loop).
        self.last_loss_old: Optional[float] = None
        self.last_loss_new: Optional[float] = None
        self.last_accepted: Optional[bool] = None
        self.last_acceptance_rate: float = 1.0  # smoothed
        self.last_mean_k: Optional[float] = None  # mean grid steps (auto mode)
        self.step_count = 0

    # ---- helpers ----

    def _select_indices(self, numel: int, frac: float) -> torch.Tensor:
        if frac >= 1.0:
            return torch.arange(numel, device="cpu")
        n = max(1, int(round(numel * frac)))
        # Sample without replacement (better than with-replacement for
        # covering the weights).
        if self.rng is None:
            return torch.randperm(numel, device="cpu")[:n]
        return torch.randperm(numel, generator=self.rng, device="cpu")[:n]

    def _make_step(self, p: torch.Tensor, g: torch.Tensor, grid: GridLike,
                   move_frac: float, beta1: float = 0.0, k: int = 1,
                   k_mode: str = "fixed", step_scale: float = 1e-3,
                   k_max: int = 1 << 22, grad_ema: float = 0.99) -> torch.Tensor:
        """Return a candidate update to ``p`` of the same shape.

        ``beta1`` and ``k`` default to the v1 behaviour (no momentum,
        one grid step). When ``beta1 > 0`` the first-moment buffer is
        read and updated *in place*; when ``k > 1`` the proposal is
        ``k`` grid points in the descent direction.
        """
        if move_frac <= 0.0 or p.numel() == 0:
            return torch.zeros_like(p)
        # Capture the grid range from the FULL tensor. This is critical
        # for AdaptiveGrid: calling ``snap``/``neighbour`` on a subset
        # would re-derive the range from the subset and produce a
        # different (finer) grid.
        if hasattr(grid, "range_for"):
            vmin, vmax = grid.range_for(p.data)
            snap_fn = grid.snap_with_range
            neigh_fn = grid.neighbour_with_range
            step_fn = grid.step_with_range
        else:
            vmin = vmax = None
            snap_fn = grid.snap
            neigh_fn = grid.neighbour
            step_fn = grid.step
        # Snap current weights so the move set is defined on the grid.
        if vmin is not None:
            snapped = snap_fn(p.data, vmin, vmax)
        else:
            snapped = snap_fn(p.data)
        # First-moment momentum (Adam-style).  When beta1 == 0 we skip
        # the state lookup entirely to keep the v1 code path free.
        if beta1 > 0.0:
            state = self.state[p]
            m = state.get("m")
            if m is None:
                m = torch.zeros_like(p.data)
                state["m"] = m
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            descent_dir = m
        else:
            descent_dir = g
        # Sign of the descent direction: -1 for descent.
        sign = -torch.sign(descent_dir).to(p.dtype)

        # Per-coordinate number of grid steps to take.
        #   "fixed" -> every moved weight takes exactly `k` steps.
        #   "auto"  -> choose k_i so the *continuous* displacement is
        #              step_scale * |g_i| / g_ref, i.e. the grid step is
        #              sized to match a target update magnitude.  This is
        #              what escapes the fine-grid plateau: at 16/32 bits a
        #              single grid step is far below float32 precision, but
        #              a few hundred (or a few million) steps are not.
        if k_mode == "auto":
            state = self.state[p]
            g_abs = descent_dir.detach().abs()
            g_now = float(g_abs.mean().item())
            gref = state.get("g_ref")
            if gref is None:
                gref = g_now if g_now > 0.0 else 1.0
            else:
                gref = grad_ema * float(gref) + (1.0 - grad_ema) * g_now
            state["g_ref"] = gref
            gref = max(gref, 1e-12)
            if vmin is not None:
                delta = (vmax - vmin) / max(grid.levels - 1, 1)
            else:
                vv_min, vv_max = grid._range(p.data)
                delta = (vv_max - vv_min) / max(grid.levels - 1, 1)
            k_vec = (step_scale * g_abs / gref / max(delta, 1e-30))
            k_vec = k_vec.clamp(1.0, float(k_max)).round()
            self.last_mean_k = float(k_vec.mean().item())
        else:
            k_vec = None

        # Pick the move-set indices (in this tensor's flat space).
        idx = self._select_indices(p.numel(), move_frac)
        flat_p = snapped.view(-1)
        flat_sign = sign.view(-1)
        # Compute candidate on the move-set only.
        candidate = snapped.clone()
        # Apply neighbour only at selected indices.
        mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
        mask[idx] = True
        flat_cand = candidate.view(-1)
        sub_w = flat_p[mask]
        sub_s = flat_sign[mask]
        if k_vec is not None:
            sub_k = k_vec.view(-1)[mask]
            if vmin is not None:
                sub_new = grid.step_multi_with_range(sub_w, sub_s, sub_k, vmin, vmax)
            else:
                sub_new = grid.step_multi(sub_w, sub_s, sub_k)
        elif vmin is not None:
            sub_new = step_fn(sub_w, sub_s, k=k, vmin=vmin, vmax=vmax)
        else:
            sub_new = step_fn(sub_w, sub_s, k=k)
        flat_cand[mask] = sub_new
        return (candidate - snapped).detach()

    def _snap(self, p: torch.Tensor, grid: GridLike) -> torch.Tensor:
        """Snap using the same range as the candidate build, for consistency."""
        if hasattr(grid, "range_for"):
            vmin, vmax = grid.range_for(p.data)
            return grid.snap_with_range(p.data, vmin, vmax)
        return grid.snap(p.data)

    @torch.no_grad()
    def _accept(self, delta: torch.Tensor, p: torch.Tensor,
                loss_old: float, loss_new: float, beta: float) -> bool:
        """Metropolis acceptance: p = min(1, exp(-beta * (L_new - L_old)))."""
        dL = float(loss_new) - float(loss_old)
        if dL <= 0.0:
            return True
        # Numeric-safe exp; if exp underflows, treat as 0 (reject).
        log_p = -beta * dL
        if log_p < -50.0:
            return False
        u = float(torch.rand(1, generator=self.rng).item()) if self.rng is not None else float(torch.rand(1).item())
        return u < math.exp(log_p)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        """Perform one GDMC step.

        ``closure`` must compute and return the loss *and* populate
        ``.grad`` on every parameter. We re-evaluate the loss on the
        proposed weights for the Metropolis test.
        """
        if closure is None:
            raise ValueError("GDMCOptimizer.step requires a closure that returns the loss")

        # 1. Snap current weights to the grid and compute loss/grad.
        #    The closure the user passed already calls zero_grad and
        #    backward, so .grad is now populated.
        for group in self.param_groups:
            grid = group["grid"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self._snap(p, grid))

        with torch.enable_grad():
            loss_tensor = closure()
        loss_old = float(loss_tensor.item())

        # 2. For each parameter, build a candidate delta and remember
        #    the original snapped value so we can roll back if the move
        #    is rejected.
        snapshots = []
        deltas = []
        params = []
        for group in self.param_groups:
            grid = group["grid"]
            beta = group["beta"]
            move_frac = group["move_frac"]
            beta1 = group["beta1"]
            k = group["k"]
            k_mode = group["k_mode"]
            step_scale = group["step_scale"]
            k_max = group["k_max"]
            grad_ema = group["grad_ema"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                snap = self._snap(p, grid).clone()
                delta = self._make_step(p, p.grad, grid, move_frac,
                                        beta1=beta1, k=k, k_mode=k_mode,
                                        step_scale=step_scale, k_max=k_max,
                                        grad_ema=grad_ema)
                snapshots.append(snap)
                deltas.append(delta)
                params.append(p)

        # 3. Apply the candidate update to the live weights, then re-evaluate.
        for p, d in zip(params, deltas):
            p.data.add_(d)

        with torch.enable_grad():
            loss_new_tensor = closure()
        loss_new = float(loss_new_tensor.item())

        # 4. Metropolis test on the FIRST param group's beta (we have one
        #    group in this project; if you use multiple groups you'd want
        #    to extend this to per-group acceptance).
        beta = self.param_groups[0]["beta"]
        accepted = self._accept(deltas[0] if deltas else torch.zeros(1),
                                params[0] if params else torch.zeros(1),
                                loss_old, loss_new, beta)

        # 5. Roll back if rejected, otherwise keep the new weights (which
        #    are on the grid by construction).
        if not accepted:
            for p, snap in zip(params, snapshots):
                p.data.copy_(snap)

        # 6. Diagnostics.
        self.last_loss_old = loss_old
        self.last_loss_new = loss_new
        self.last_accepted = bool(accepted)
        self.last_acceptance_rate = 0.9 * self.last_acceptance_rate + 0.1 * (1.0 if accepted else 0.0)
        self.step_count += 1

        return loss_tensor
