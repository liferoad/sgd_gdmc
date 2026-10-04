"""GDMC optimizer for discretized deep-learning weight optimization.

The optimizer implements the algorithm from

    Hu, Beratan, Yang. "A gradient-directed Monte Carlo method for global
    optimization in a discrete space: Application to protein sequence
    design and folding." J. Chem. Phys. 131, 154117 (2009).

adapted to neural-network weight optimization. We follow the paper's
two-step recipe:

1.  Compute the gradient of the loss w.r.t. the (continuous) weights via
    ordinary backprop. This is the gradient of the "virtual continuous
    surface" introduced by the discretization.

2.  Propose a discrete move. We pick a fraction move_frac of weights (the
    move set) and for each one, step one grid point in the direction of
    -sign(grad) using Grid.neighbour(w, sign). This is the gradient-directed
    proposal, equivalent to the per-site / per-direction move in the paper.

3.  Accept or reject the move with the Metropolis criterion
    p = min{1, exp(-beta * (L_new - L_old))} evaluated on the discretized
    weights. beta is a temperature (larger = more greedy).

Memory
------
The proposal and its rollback are stored sparsely: only the moved indices
and their old/new values are materialised (O(move_frac * numel)), instead
of full-size snap/candidate/delta tensors. With the default
select_mode="bernoulli" the move-set selection also avoids the full int64
randperm that the legacy path allocated. Weights are still float32 here:
this is a storage optimisation, not a packed-weight implementation.

Acceptance
----------
The Metropolis test compares two loss evaluations. accept_on selects which:

* "minibatch" (default, legacy behaviour): re-evaluate the loss on the
  same minibatch whose gradient defined the move. Cheap, but biased toward
  acceptance, because the move was built to reduce that loss.
* "separate": evaluate on a separate batch supplied by the training loop
  through the second argument of step(). This removes the optimism bias at
  the cost of an extra forward pass.

accept_on="full" (documented in earlier versions but never implemented) is
gone and now raises.

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

from typing import Callable, Iterable, List, Optional
import math
from dataclasses import dataclass
import torch

from .grid import make_grid, GridLike  # type: ignore


@dataclass
class _Proposal:
    """Sparse description of one proposed move."""

    param: torch.Tensor
    index: torch.Tensor          # int64 flat indices that move
    old_values: torch.Tensor     # grid values at those indices before the move
    new_values: torch.Tensor     # grid values after the move
    step_sizes: torch.Tensor     # continuous displacement per moved entry


def _mean_abs(t: torch.Tensor, chunk: int = 1 << 20) -> float:
    """Mean of |t| without materialising a full-size abs() tensor."""
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if n == 0:
        return 0.0
    total = 0.0
    for start in range(0, n, chunk):
        total += float(flat[start:start + chunk].abs().sum().item())
    return total / n


class GDMCOptimizer(torch.optim.Optimizer):
    """Gradient-Directed Monte Carlo optimizer on a discretized weight grid.

    Parameters
    ----------
    params : iterable
        Model parameters (quantized in place each step).
    grid : GridLike, optional
        Quantization grid. If None, a uniform grid of ``bits`` levels in
        [-1, 1] is used.
    beta : float, default 2.0
        Metropolis inverse-temperature (larger = greedier).
    move_frac : float, default 0.01
        Fraction of weights to mutate per step.
    accept_on : {"minibatch", "separate"}, default "minibatch"
        Which loss to use for the Metropolis test. "separate" requires the
        training loop to pass an accept_closure to step().
    beta1 : float, default 0.0
        First-moment momentum on the gradient, Adam-style.
    k : int, default 1
        Number of grid points to step per move (fixed mode).
    k_mode : {"fixed", "auto"}, default "fixed"
        "auto" sizes the per-coordinate step so the continuous
        displacement matches step_scale * |g| / g_ref.
    step_scale : float, default 1e-3
        Target displacement for an average-gradient coordinate (auto mode).
    k_max : int, default 1 << 22
        Cap on per-coordinate grid steps in auto mode.
    grad_ema : float, default 0.99
        EMA rate for the gradient reference in auto mode.
    select_mode : {"bernoulli", "randperm"}, default "bernoulli"
        How to draw the move set. "randperm" reproduces the original
        implementation (full int64 permutation, more memory); "bernoulli"
        uses rand < move_frac and is the memory-lean default.
    noise_rho : float, default 0.0
        If > 0, add noise_rho * RMS(g) * N(0, 1) to the proposal gradient of
        every tensor (norm-scaled gradient noise). The acceptance loss is
        unaffected. This is the feedback study's central independent
        variable, built in rather than monkey-patched.
    noise_generator : torch.Generator, optional
        Random source for noise_rho (defaults to rng).
    rng : torch.Generator, optional
        Random source for the move set, Metropolis and (by default) the
        gradient noise. Passing one keeps the optimizer from perturbing the
        global RNG stream, so data order is reproducible across optimizers
        run with the same seed.
    """

    requires_accept_closure = True

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
        select_mode: str = "bernoulli",
        noise_rho: float = 0.0,
        noise_generator: Optional[torch.Generator] = None,
    ) -> None:
        if grid is None:
            grid = make_grid(grid_spec, bits=bits)
        if k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if accept_on not in ("minibatch", "separate"):
            raise ValueError(
                f"accept_on must be 'minibatch' or 'separate', got {accept_on!r} "
                "(the old 'full' option was never implemented)"
            )
        if select_mode not in ("bernoulli", "randperm"):
            raise ValueError(f"unknown select_mode: {select_mode!r}")
        if k_mode not in ("fixed", "auto"):
            raise ValueError(f"unknown k_mode: {k_mode!r}")
        if noise_rho < 0.0:
            raise ValueError("noise_rho must be >= 0")
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
            select_mode=select_mode,
            noise_rho=noise_rho,
        )
        super().__init__(params, defaults)
        self.rng = rng
        self.noise_generator = noise_generator if noise_generator is not None else rng
        self.requires_closure = True
        self.last_loss_old: Optional[float] = None
        self.last_loss_new: Optional[float] = None
        self.last_grad_loss: Optional[float] = None
        self.last_delta_loss: Optional[float] = None
        self.last_accepted: Optional[bool] = None
        self.last_acceptance_rate: float = 1.0
        self.last_mean_k: Optional[float] = None
        self.last_num_moved: int = 0
        self.last_accepted_step_size: float = 0.0
        self.step_count = 0

    # ---- helpers ----

    def _device_gen(self, device, base=None):
        """A generator on the given device, derived deterministically from base.

        torch requires the generator's device to match the tensor's, so a CPU
        generator cannot drive randomness for CUDA/MPS tensors. We derive one
        device generator per device from the caller's generator and cache it.
        """
        base = self.rng if base is None else base
        if base is None:
            return None
        device = torch.device(device)
        if base.device == device:
            return base
        cache = getattr(self, "_gen_cache", None)
        if cache is None:
            cache = {}
            self._gen_cache = cache
        key = (str(device), id(base))
        if key not in cache:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,),
                                     generator=base).item())
            cache[key] = torch.Generator(device=device).manual_seed(seed)
        return cache[key]

    def _group_for(self, p: torch.Tensor):
        for g in self.param_groups:
            if any(q is p for q in g["params"]):
                return g
        return self.param_groups[0]

    def _add_gradient_noise(self, g: torch.Tensor, rho: float) -> None:
        if rho <= 0.0:
            return
        s = rho * float(g.detach().pow(2).mean().sqrt().item())
        if s <= 0.0:
            return
        gen = self._device_gen(g.device, self.noise_generator)
        if gen is None:
            g.add_(torch.randn_like(g) * s)
        else:
            g.add_(torch.randn(g.shape, generator=gen,
                               dtype=g.dtype, device=g.device) * s)

    def _grid_range(self, p: torch.Tensor, grid: GridLike):
        if hasattr(grid, "range_for"):
            vmin, vmax = grid.range_for(p.data)
            return vmin, vmax, True
        return None, None, False

    def _snap_inplace(self, p: torch.Tensor, grid: GridLike) -> None:
        """Snap p.data onto the grid **in place**, with no full-size temporary.

        The range is captured once from the full tensor (range_for), so this is
        correct for fixed, per-tensor and adaptive grids alike, and the chained
        in-place ops avoid allocating a copy of the parameter.
        """
        levels = getattr(grid, "levels", 1)
        if levels == 1:
            p.data.zero_()
            return
        vmin, vmax, ranged = self._grid_range(p, grid)
        if not ranged:
            vmin, vmax = float(grid.vmin), float(grid.vmax)
        delta = (vmax - vmin) / (levels - 1)
        p.data.sub_(vmin).div_(delta).round_().clamp_(0, levels - 1)
        p.data.mul_(delta).add_(vmin)

    def _select_mask(self, numel: int, frac: float, device,
                     mode: str) -> torch.Tensor:
        """Boolean move-set mask with at least one element selected."""
        if frac >= 1.0:
            return torch.ones(numel, dtype=torch.bool, device=device)
        n = max(1, int(round(numel * frac)))
        gen = self._device_gen(device)
        if mode == "randperm":
            idx = torch.randperm(numel, device=device, generator=gen)[:n]
            mask = torch.zeros(numel, dtype=torch.bool, device=device)
            mask[idx] = True
            return mask
        mask = torch.rand(numel, device=device, generator=gen) < frac
        if not bool(mask.any()):
            j = int(torch.randint(numel, (1,), device=device,
                                  generator=gen).item())
            mask[j] = True
        return mask

    def _build_proposal(self, p: torch.Tensor, grid: GridLike) -> Optional[_Proposal]:
        """Construct (but do not apply) the sparse proposal for p."""
        group = self._group_for(p)
        move_frac = group["move_frac"]
        if move_frac <= 0.0 or p.numel() == 0 or p.grad is None:
            return None

        grad = p.grad
        self._add_gradient_noise(grad, group["noise_rho"])

        beta1 = group["beta1"]
        if beta1 > 0.0:
            state = self.state[p]
            m = state.get("m")
            if m is None:
                m = torch.zeros_like(p.data)
                state["m"] = m
            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            descent_dir = m
        else:
            descent_dir = grad

        vmin, vmax, ranged = self._grid_range(p, grid)
        if ranged:
            delta = (vmax - vmin) / max(grid.levels - 1, 1)
        else:
            delta = (float(grid.vmax) - float(grid.vmin)) / max(grid.levels - 1, 1)

        flat_p = p.data.view(-1)
        mask = self._select_mask(p.numel(), move_frac, p.device,
                                 group["select_mode"])
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return None

        sub_w = flat_p[idx]
        if ranged:
            sub_snap = grid.snap_with_range(sub_w, vmin, vmax)
        else:
            sub_snap = grid.snap(sub_w)

        dir_sub = descent_dir.view(-1)[idx]
        sign = -torch.sign(dir_sub)

        if group["k_mode"] == "auto":
            state = self.state[p]
            g_now = _mean_abs(descent_dir)
            gref = state.get("g_ref")
            if gref is None:
                gref = g_now if g_now > 0.0 else 1.0
            else:
                gref = group["grad_ema"] * float(gref) + (1.0 - group["grad_ema"]) * g_now
            state["g_ref"] = gref
            gref = max(float(gref), 1e-12)
            g_abs = dir_sub.detach().abs()
            k_moved = (group["step_scale"] * g_abs / gref / max(delta, 1e-30))
            k_moved = k_moved.clamp(1.0, float(group["k_max"])).round()
            self.last_mean_k = float(k_moved.mean().item())
        else:
            k_moved = None

        if k_moved is not None:
            if ranged:
                new = grid.step_multi_with_range(sub_snap, sign, k_moved, vmin, vmax)
            else:
                new = grid.step_multi(sub_snap, sign, k_moved)
            steps = k_moved * delta
        else:
            if ranged:
                new = grid.step_with_range(sub_snap, sign, k=group["k"],
                                           vmin=vmin, vmax=vmax)
            else:
                new = grid.step(sub_snap, sign, k=group["k"])
            steps = torch.full_like(new, float(group["k"]) * delta)

        old = sub_snap
        return _Proposal(param=p, index=idx, old_values=old,
                         new_values=new, step_sizes=steps)

    def _apply(self, records: List[_Proposal]) -> None:
        for r in records:
            r.param.data.view(-1)[r.index] = r.new_values

    def _rollback(self, records: List[_Proposal]) -> None:
        for r in records:
            r.param.data.view(-1)[r.index] = r.old_values

    @torch.no_grad()
    def _accept(self, loss_old: float, loss_new: float, beta: float) -> bool:
        dL = float(loss_new) - float(loss_old)
        self.last_delta_loss = dL
        if dL <= 0.0:
            return True
        log_p = -beta * dL
        if log_p < -50.0:
            return False
        if self.rng is not None:
            u = float(torch.rand(1, device=self.rng.device,
                                 generator=self.rng).item())
        else:
            u = float(torch.rand(1).item())
        return u < math.exp(log_p)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None,
             accept_closure: Optional[Callable[[], torch.Tensor]] = None):
        """Perform one GDMC step.

        closure must compute and return the loss and populate .grad on every
        parameter.

        accept_closure, when supplied, is evaluated *before and after* the
        proposal and those two values are the Metropolis pair. It should be a
        deterministic, side-effect-free loss evaluation (src/train.py passes an
        eval-mode, BatchNorm-preserving closure). accept_on="separate" requires
        it; without it the legacy behaviour is kept (minibatch mode reuses the
        gradient closure; "separate" raises).
        """
        if closure is None:
            raise ValueError("GDMCOptimizer.step requires a closure that returns the loss")
        accept_on = self.param_groups[0]["accept_on"]
        if accept_on == "separate" and accept_closure is None:
            raise ValueError("accept_on='separate' requires an accept_closure argument")

        # 1. Snap current weights onto the grid, then compute loss/grad.
        for group in self.param_groups:
            grid = group["grid"]
            for p in group["params"]:
                self._snap_inplace(p, grid)

        with torch.enable_grad():
            loss_tensor = closure()
        grad_loss = float(loss_tensor.item())

        # 2. Baseline acceptance loss. When the caller supplies an
        #    accept_closure it is evaluated BEFORE the proposal, so both
        #    acceptance losses come from the same evaluation. Mixing the
        #    gradient-batch loss with an acceptance-batch loss would fold the
        #    difference between the two batches into the Metropolis decision.
        if accept_closure is not None:
            with torch.enable_grad():
                loss_old = float(accept_closure().item())
        else:
            loss_old = grad_loss

        # 3. Build the sparse proposals for every parameter.
        records: List[_Proposal] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                r = self._build_proposal(p, group["grid"])
                if r is not None:
                    records.append(r)

        # 4. Apply the candidate moves, then re-evaluate the acceptance loss.
        self._apply(records)
        if accept_closure is not None:
            with torch.enable_grad():
                loss_new = float(accept_closure().item())
        else:
            with torch.enable_grad():
                loss_new = float(closure().item())

        # 5. Metropolis test (first param group's beta).
        accepted = self._accept(loss_old, loss_new, self.param_groups[0]["beta"])

        # 5. Roll back if rejected.
        if not accepted:
            self._rollback(records)

        # 6. Diagnostics.
        num_moved = int(sum(int(r.index.numel()) for r in records))
        if records:
            mean_step = float(torch.cat([r.step_sizes.reshape(-1)
                                         for r in records]).mean().item())
        else:
            mean_step = 0.0
        self.last_grad_loss = grad_loss
        self.last_loss_old = loss_old
        self.last_loss_new = loss_new
        self.last_accepted = bool(accepted)
        self.last_num_moved = num_moved
        self.last_accepted_step_size = mean_step if accepted else 0.0
        self.last_acceptance_rate = (0.9 * self.last_acceptance_rate
                                     + 0.1 * (1.0 if accepted else 0.0))
        self.step_count += 1
        return loss_tensor

    def _snap(self, p: torch.Tensor, grid: GridLike) -> torch.Tensor:
        """Legacy helper: return a fresh snapped copy of p.data."""
        vmin, vmax, ranged = self._grid_range(p, grid)
        if ranged:
            return grid.snap_with_range(p.data, vmin, vmax)
        return grid.snap(p.data)
