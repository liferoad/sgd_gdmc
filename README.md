# sgd_gdmc — Applying Gradient-Directed Monte Carlo to Deep Learning Weight Optimization

This project adapts the **GDMC** (Gradient-Directed Monte Carlo) method
of Hu, Beratan, and Yang ([J. Chem. Phys. 131, 154117, 2009](https://doi.org/10.1063/1.3236834))
from discrete protein sequence design / folding to **discretized deep
learning weight optimization**.

## Idea

In the paper, GDMC optimizes over a discrete space by

1. treating discrete variables as continuous to obtain gradients,
2. using those gradients to propose the next discrete state, and
3. accepting / rejecting with the Metropolis criterion so the search
   can escape local minima.

We translate this to neural-network training by **quantizing the model
weights to a grid** (uniform or adaptive, with 2 / 3 / 4 / 8 / 16 / 32
bits), computing the usual backprop gradient, using it to propose a
neighbouring grid point, and accepting / rejecting with
`p = min{1, exp(−β(L_new − L_old))}`.

## Layout

```
src/
  gdmc/         GDMC optimizer + grid quantization utilities
  baselines/    SGD, momentum, Adam, Projected-GD, SGLD
  models/       small MLP, small CNN
  data/         toy regression, MNIST, CIFAR-10 loaders
experiments/   one script per task sweep
analysis/      plots and tables
results/       outputs (CSVs, PNGs, REPORT.md)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

Each experiment writes CSV results into `results/raw/` and PNG plots into
`results/`.

```bash
# Sanity check (fast, ~1 min)
.venv/bin/python experiments/01_toy_regression.py

# MNIST MLP sweep (a few minutes)
.venv/bin/python experiments/02_mnist_mlp.py

# MNIST CNN sweep (a few minutes)
.venv/bin/python experiments/03_mnist_cnn.py

# CIFAR-10 CNN sweep (longer; ~hour-scale on CPU)
.venv/bin/python experiments/04_cifar10_cnn.py

# Beta ablation on MNIST CNN
.venv/bin/python experiments/05_beta_sweep.py

# Aggregate results and write REPORT.md
.venv/bin/python analysis/aggregate.py
```

See `plan.md` for the full experimental plan, `results/REPORT.md`
(written by `analysis/aggregate.py`) for the v1 final write-up,
and `docs/v2_results.md` for the GDMC v2 update (momentum +
multi-step moves).

## GDMC v2 (momentum + multi-step)

`GDMCOptimizer` accepts two new kwargs (both default to the v1
behaviour, so v1 callers are bit-exactly unchanged):

* `beta1` (default 0.0): first-moment momentum on the gradient,
  Adam-style. The proposal direction becomes `-sign(m_t)`. Set to
  `0.9` for v2.
* `k` (default 1): number of grid points to step per move. Set to
  `2`–`8` for v2.

Empirical results on the toy regression sweep (see
`docs/v2_results.md`):

| bits | best v2 config | test loss (mean ± std) | v1 for reference | speedup |
|------|----------------|------------------------|------------------|---------|
| 3    | beta1=0.9, k=1 | 0.147 ± 0.029          | 0.398 ± 0.068    | 2.7×    |
| 4    | beta1=0.9, k=1 | 0.101 ± 0.069          | 0.367 ± 0.197    | 3.6×    |
| 8    | beta1=0.9, k=4 | 0.029 ± 0.025          | 0.099 ± 0.005    | 3.4×    |

Recommended: use momentum (`beta1=0.9`) at 2-4 bits, momentum +
multi-step (`beta1=0.9, k=4`) at 8 bits, and stick with v1 at 16+
bits.

Run the v2 sweep:

```bash
.venv/bin/python experiments/06_gdmc_v2_sweep.py

# Corrected baseline comparison (tuned Adam, grid-scaled Projected-GD,
# 8-bit Adam, momentum-signSGD) - see docs/fixes_2026-10-04.md
.venv/bin/python experiments/12_corrected_baselines.py

# Measured peak-memory benchmark: FP32 Adam vs 8-bit Adam vs GDMC
.venv/bin/python experiments/13_memory_benchmark.py

# Gradient-noise study (validation quality vs proposal-gradient noise)
.venv/bin/python experiments/14_noise_study.py
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The suite covers the grid invariants, the optimizer (including the regression
that the proposal must actually change the weights at every bit-width, and that
rejected moves roll back exactly), the 8-bit Adam baseline, the training loop /
runner, and the aggregator.

## Optimizer and runner options added 2026-10-04

* GDMCOptimizer(accept_on="separate") plus step(closure, accept_closure):
  evaluate the Metropolis test on a different minibatch
  (TrainConfig.accept_split or accept_loader). The accept closure is
  evaluated before AND after the proposal, in eval() mode with BatchNorm
  buffers and the training mode restored, so the decision compares two
  same-batch losses and rejected moves leave no side effects.
* GDMCOptimizer(noise_rho=rho): norm-scaled Gaussian noise on the proposal
  gradient only (the noise study's independent variable).
* select_mode="bernoulli" (default) draws the move set with rand < move_frac,
  avoiding a full int64 permutation; proposals and rollback are stored
  sparsely (moved entries only).
* GDMCOptimizer takes an explicit rng and no longer consumes the global RNG,
  so runs with the same seed share a data order (common random numbers).
* ProjectedGD(lr_scale=s) sets lr = s * grid spacing, so the baseline can no
  longer be frozen by lr < delta.
* New baselines: adam8bit (block-wise 8-bit optimizer state) and signsgd
  (momentum signSGD).
* RunResult now records delta_loss, num_moved, mean_acceptance_rate,
  peak_rss_mb and rss_growth_mb; final_train_loss is populated.

Known artifacts in results produced before 2026-10-04, and the corrected
numbers, are documented in docs/fixes_2026-10-04.md.
