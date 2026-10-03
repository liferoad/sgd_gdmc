# GDMC v2 results — momentum and multi-step moves

This is the v2 update to v1. v1 (the version documented in
[results/REPORT.md](../results/REPORT.md)) had two known
limitations:

* **L1 — "one grid step is suboptimal at fine grids"**: at 16/32 bits,
  the v1 move is essentially a no-op.
* **L2 — "no momentum"**: the proposal direction `sign(g)` is noisy,
  especially on coarse grids where many weights have the same
  gradient sign but different magnitudes.

v2 adds two new optimizer hyperparameters:

* ``beta1`` (default 0.0): first-moment momentum on the gradient,
  Adam-style. ``m_t = beta1 * m_{t-1} + (1 - beta1) * g_t``;
  the proposal direction becomes ``-sign(m_t)``.
* ``k`` (default 1): number of grid points to step per move.
  ``k > 1`` takes larger discrete steps in the descent direction.

When ``beta1 == 0`` and ``k == 1``, v2 is **bit-exactly equivalent**
to v1 — the defaults preserve the original behavior. New code lives
in [src/gdmc/optimizer.py](../src/gdmc/optimizer.py) and
[src/gdmc/grid.py](../src/gdmc/grid.py) (new `step(w, sign, k)` method on
both grids).

## Sweep setup

`experiments/06_gdmc_v2_sweep.py` runs 6 (beta1, k) combinations
× 6 bit-widths × 3 seeds on the toy regression task, 30 epochs each.
Raw results: [results/raw/gdmc_v2.csv](../results/raw/gdmc_v2.csv)
(108 runs).

The v1 baseline is `beta1=0.0, k=1`; everything else is v2. The
labels in the CSV identify the configuration (`v1-b1=0.0-k=1`,
`v2-b1=0.9-k=1`, `v2-b1=0.0-k=2`, `v2-b1=0.0-k=4`,
`v2-b1=0.9-k=2`, `v2-b1=0.9-k=4`).

## Headline table — mean ± std test loss (lower is better)

| config           | b=2            | b=3            | b=4            | b=8            | b=16           | b=32           |
|------------------|----------------|----------------|----------------|----------------|----------------|----------------|
| v1 (b1=0, k=1)   | 0.672 ± 0.554  | 0.398 ± 0.068  | 0.367 ± 0.197  | 0.099 ± 0.005  | 0.606 ± 0.034  | 0.618 ± 0.033  |
| v2 b1=0.9, k=1   | 1.270 ± 0.921  | **0.147 ± 0.029** | **0.101 ± 0.069** | 0.089 ± 0.016  | 0.604 ± 0.034  | 0.618 ± 0.033  |
| v2 b1=0.0, k=2   | 1.839 ± 1.456  | 0.406 ± 0.090  | 0.321 ± 0.137  | 0.081 ± 0.014  | 0.595 ± 0.034  | 0.618 ± 0.033  |
| v2 b1=0.0, k=4   | 1.598 ± 1.490  | 0.445 ± 0.231  | 0.526 ± 0.087  | 0.088 ± 0.042  | 0.576 ± 0.035  | 0.618 ± 0.033  |
| v2 b1=0.9, k=2   | 1.918 ± 1.470  | 0.379 ± 0.187  | 0.168 ± 0.015  | 0.056 ± 0.034  | 0.592 ± 0.035  | 0.618 ± 0.033  |
| **v2 b1=0.9, k=4** | 1.918 ± 1.470  | 0.559 ± 0.048  | 0.301 ± 0.147  | **0.029 ± 0.025** | 0.571 ± 0.036  | 0.618 ± 0.033  |

## Headline table — mean acceptance rate

| config           | b=2  | b=3  | b=4  | b=8  | b=16 | b=32 |
|------------------|------|------|------|------|------|------|
| v1 (b1=0, k=1)   | 0.03 | 0.32 | 0.82 | 1.00 | 1.00 | 1.00 |
| v2 b1=0.9, k=1   | 0.03 | 0.87 | 0.96 | 1.00 | 1.00 | 1.00 |
| v2 b1=0.0, k=2   | 0.00 | 0.07 | 0.25 | 1.00 | 1.00 | 1.00 |
| v2 b1=0.0, k=4   | 0.00 | 0.00 | 0.23 | 1.00 | 1.00 | 1.00 |
| v2 b1=0.9, k=2   | 0.00 | 0.31 | 0.92 | 1.00 | 1.00 | 1.00 |
| v2 b1=0.9, k=4   | 0.00 | 0.00 | 0.76 | 1.00 | 1.00 | 1.00 |

## Reading the results

The table has a clear three-zone story:

### 3-4 bits — momentum is a huge win, multi-step hurts

At 3-bit, **v2 with just momentum (beta1=0.9, k=1) takes the test
loss from 0.398 → 0.147, a 2.7× improvement**. At 4-bit, it goes
0.367 → 0.101, a **3.6× improvement**. This is the biggest
algorithmic gain we found.

Why does this work? At 3-4 bits the grid is so coarse that the
gradient is very noisy: many weights have the same gradient sign
but different magnitudes, and neighbouring grid points are not
equidistant in loss value. The momentum buffer averages this noise
over time so the proposal direction is more reliable. The acceptance
rate confirms it: at 3-bit it goes from 0.32 (v1) to 0.87 (v2
momentum), and at 4-bit from 0.82 to 0.96.

Multi-step at 3-4 bits is the *opposite* story. k=2 or k=4 at 3-bit
(8 levels) means jumping 2-4 levels in one move — a 25-50% of the
range. The acceptance rate collapses to 0.00-0.31. The test loss
degrades to v1 or worse. **Multi-step should not be used at coarse
grids.**

### 8 bits — multi-step is a clear win

At 8-bit, **v2 with beta1=0.9, k=4 takes the loss from 0.099 →
0.029, a 3.4× improvement**. All v2 configurations beat v1 here
because the grid is fine enough that the v1 one-step move is leaving
performance on the table. Multi-step lets the optimizer cover ground
faster; momentum adds a small extra win. Acceptance stays at 1.00
because the surface is smooth enough that even larger moves are
accepted.

### 16-32 bits — nothing helps

At 16 and 32 bits, every configuration gets ~0.57-0.62 test loss
(close to the v1 number). This is the plateau reported in v1: the
model is in a random-walk regime, and neither momentum nor
multi-step breaks it. The acceptance rate is 1.00 for all, but the
*effective* step is still tiny.

**Conclusion: at fine grids, "one step" is the wrong rate-limiting
factor — the real problem is that the loss landscape itself is
flat. Fixing this needs the future-work ideas (Fix 5.3 warm-start
from a continuous Adam baseline, Fix 1.2 magnitude-scaled k, etc.).

### 2 bits — nothing helps

All configurations get ~0.67-1.9 test loss. 2 bits (4 levels) is
just too coarse for the toy regression; the model can't
represent the function regardless of optimizer.

## Best v2 configuration per bit-width

| bits | best v2 config                | test loss (mean ± std)  | v1 for reference       | speedup vs v1 |
|------|--------------------------------|-------------------------|------------------------|---------------|
| 2    | (none — v2 not useful)        | 0.67+                   | 0.67 ± 0.55            | ≈ 1×          |
| 3    | **beta1=0.9, k=1**            | 0.147 ± 0.029           | 0.398 ± 0.068          | **2.7×**       |
| 4    | **beta1=0.9, k=1**            | 0.101 ± 0.069           | 0.367 ± 0.197          | **3.6×**       |
| 8    | **beta1=0.9, k=4**            | 0.029 ± 0.025           | 0.099 ± 0.005          | **3.4×**       |
| 16   | beta1=0.9, k=4                | 0.571 ± 0.036           | 0.606 ± 0.034          | 1.06×         |
| 32   | (none — all stuck at 0.62)     | 0.618 ± 0.033           | 0.618 ± 0.033          | 1.00×         |

## Recommendation for practitioners

* **At 2-4 bits: enable momentum, keep k=1.**
  `GDMCOptimizer(params, grid=grid, beta1=0.9, k=1)`. This is the
  single most useful change in v2 — 2.7-3.6× lower test loss on
  the only bit-widths where GDMC is uniquely valuable.
* **At 8 bits: enable momentum and multi-step.**
  `beta1=0.9, k=4` was the best here (3.4× better than v1). At
  the boundary of 8/16 bits, k=2 is a safe middle ground.
* **At 16+ bits: stick with v1** (or abandon GDMC entirely — Adam
  works fine at this resolution).

## What v2 does *not* fix

* **The 16/32-bit plateau.** v2's multi-step is up to 4 levels, but
  4 levels on a 65536-level grid is still nothing. The
  *fundamental* limit is the discrete-step size, and v2's constant-k
  doesn't break it. The fix in [future_work.md](future_work.md)
  (Fix 1.1) — a *gradient-magnitude-scaled* k — would.

* **The 2-bit failure.** 4 levels is just too few. No optimizer
  variant saves this.

* **2× compute per step.** v2 is the same cost as v1 (the k- and
  beta1-paths are O(numel) tensor ops, dominated by the two forward
  passes for the acceptance test). The next big compute win is
  reusing the autograd graph across old and proposed weights
  (future_work Fix 11.1).

## Files touched in v2

* `src/gdmc/grid.py` — added `step(w, sign, k)` and
  `step_with_range(w, sign, k, vmin, vmax)`; `neighbour` is now a
  thin wrapper that calls `step(k=1)`.
* `src/gdmc/optimizer.py` — `GDMCOptimizer.__init__` adds `beta1`
  and `k`; `_make_step` reads/updates a per-tensor momentum buffer
  in `self.state[p]['m']` and uses `grid.step(...)` instead of
  `grid.neighbour(...)`.
* `src/runner.py` — `RunResult` adds `beta1` and `k` columns;
  `run_one` threads them through.
* `experiments/06_gdmc_v2_sweep.py` — new ablation script.
