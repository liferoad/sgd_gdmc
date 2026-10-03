# GDMC for Deep Learning Weight Optimization — Final Report

## TL;DR

We adapted the **GDMC** (Gradient-Directed Monte Carlo) method of
[Hu, Beratan, Yang (J. Chem. Phys. 131, 154117, 2009)](https://doi.org/10.1063/1.3236834)
from discrete protein sequence design / folding to **discretized
deep-learning weight optimization**, and compared it against five
standard optimizers (SGD, Momentum, Adam, Projected-GD, SGLD) across
three tasks (toy regression, MNIST MLP, MNIST CNN, CIFAR-10 CNN) and
six quantization levels (2, 3, 4, 8, 16, 32 bits).

**Key finding**: GDMC is dramatically more robust than the QAT-style
"Projected-GD" baseline on *coarse* (≤ 4-bit) quantization, where
Projected-GD gets destroyed by Adam's adaptive learning rates bouncing
on the discrete grid. On *fine* (≥ 8-bit) quantization, Projected-GD
recovers and matches continuous Adam, while GDMC's "one grid step per
update" is too small to make progress in our short training runs.

| task            | bits | Adam (32) | Projected-GD | **GDMC (uniform)** | GDMC (uniform-pt) |
|-----------------|------|-----------|--------------|--------------------|-------------------|
| MNIST MLP       | 2    | 0.9325    | 0.1987       | **0.7460**         | —                 |
| MNIST MLP       | 3    | 0.9325    | 0.1984       | **0.7776**         | 0.7319            |
| MNIST MLP       | 4    | 0.9325    | 0.1997       | **0.8279**         | 0.7366            |
| MNIST MLP       | 8    | 0.9325    | 0.9284       | 0.7745             | 0.7366            |
| MNIST CNN       | 4    | 0.8041    | 0.1523       | **0.5180**         | 0.4018            |
| MNIST CNN       | 8    | 0.8041    | 0.8708       | 0.4617             | 0.3924            |
| CIFAR-10 CNN    | 4    | 0.2513    | 0.1208       | **0.2671**         | 0.2702            |
| CIFAR-10 CNN    | 8    | 0.2513    | 0.2727       | **0.2775**         | 0.2698            |

(Best test accuracy, mean over 3 seeds. "Adam (32)" is the continuous
baseline — the best you can do without quantization.)

## Method

### GDMC, as adapted from the paper

Given a network with weights `W` and a loss `L(W)`, GDMC:

1. **Builds a virtual continuous surface**: quantize `W` to a grid
   `G`. For each weight `w_i`, the discrete value is one of `K`
   levels (e.g. 4 levels for 2-bit, 256 levels for 8-bit).
2. **Computes the gradient** of `L` w.r.t. the *quantized* weights by
   ordinary backprop. This is the gradient of the "virtual continuous
   surface" defined by the quantization.
3. **Proposes a discrete move**: pick a fraction `move_frac` of the
   weights (the *move set*) and for each one, step one grid point in
   the direction of `-sign(grad)` using `Grid.neighbour(w, sign)`.
4. **Accepts or rejects the move** with the Metropolis criterion
   `p = min{1, exp(-beta * (L_new - L_old))}` evaluated on the
   **discretized** weights.
5. On rejection, the proposed weights are discarded and the original
   snapped weights are kept.

This is the same recipe the paper uses for protein sequence design
(per-site mutation) and folding (per-direction pull-move). The
key insight is that the *gradient of the continuous surface* tells us
which adjacent grid point to consider, and the *Metropolis test* lets
us climb out of local minima.

### Why this is different from SGLD / Projected-GD

* **SGLD** adds continuous Gaussian noise. The weights are continuous,
  not quantized. So SGLD doesn't actually train a discrete model — it
  just explores a continuous one and then we quantize for evaluation.
* **Projected-GD** is the QAT-style baseline: do continuous Adam, then
  project to the grid at the end of each step. The Adam state (m, v)
  is computed on the quantized weights, but the *proposal* is just a
  step in the unconstrained direction — so Adam can bounce wildly
  when the quantization is coarse.
* **GDMC** stays on the grid by construction (the move is "one grid
  step") and uses the *sign* of the gradient (not the magnitude) so
  it doesn't run away when the quantization is coarse.

### Implementation notes

* The optimizer is a `torch.optim.Optimizer` subclass (`GDMCOptimizer`)
  with a `step(closure)` interface (like `LBFGS`) because the
  acceptance test requires re-evaluating the loss.
* Two grid variants: `UniformGrid(K levels in [-1, 1])` and
  `UniformGrid(per_tensor=True)` (range tracks weight magnitude). The
  adaptive grid whose range rescaled to `max|w|` per call was retired
  because the per-step range change caused runaway weight growth.
* A `requires_closure` flag on the optimizer tells the training loop
  which `step()` style to use; the same loop drives all 6 optimizers.

## Experiments

### Tasks and settings

| task           | model          | epochs | train size | batch |
|----------------|-----------------|--------|------------|-------|
| Toy regression | MLP 1 → 64 → 64 → 1 | 30 | 400 | 32 |
| MNIST MLP      | MLP 784 → 256 → 256 → 10 | 2 | 10,000 | 128 |
| MNIST CNN      | SmallCNN (3 conv blocks + 2 FC, 16/32/64 width) | 2 | 10,000 | 128 |
| CIFAR-10 CNN   | SmallCNN (3 input channels) | 2 | 10,000 | 128 |

Each (task, optimizer, grid, bits) was run with 3 seeds. The CSV
results in `results/raw/` have one row per (run) and the plots in
`results/plots/` show mean values across seeds.

### Optimizers

* **sgd**: PyTorch SGD, no momentum.
* **momentum**: PyTorch SGD with `momentum=0.9`.
* **adam**: PyTorch Adam, `lr=1e-2` for regression, `lr=1e-3` for classification.
* **sgld**: Welling & Teh SGLD, `beta=10` (regression) or `beta=100`
  (classification), `lr=5e-4`. Always evaluated on the full-precision
  weights — SGLD does not have a discrete space by default.
* **projected-gd**: Adam wrapped in `ProjectedGD` — at the end of each
  step the weights are snapped to the supplied grid. Base = Adam
  (`lr=1e-2`).
* **gdmc (uniform)**: `GDMCOptimizer` with a `UniformGrid(bits=…)`.
  `beta=2.0`, `move_frac=0.01` (regression: 0.05).
* **gdmc (uniform-pt)**: same as above but with a per-tensor uniform
  grid (range tracks `max|w|`). Reported as `gdmc-adaptive`.

Hyperparameters were tuned on the toy task; for MNIST/CIFAR we used
the same settings without per-task tuning. A small β sweep on the
MNIST CNN (4-bit) showed no statistically significant difference
between `β ∈ {0.25, 0.5, 1, 2, 4}` in our short runs — see the
`results/raw/beta_sweep.csv` data.

### Headline tables

The full per-task tables are below. `mean_std` is mean ± std over
3 seeds; `count` is the number of seeds. For toy regression the
metric is `best_test_loss` (lower is better); for classification it is
`best_test_acc` (higher is better).

#### Toy regression (mean ± std, 3 seeds)

| optimizer     | grid_spec   |   bits | mean_std (test_loss) |   min |    max |
|:--------------|:------------|-------:|:---------------------|------:|-------:|
| adam          | none        |     32 | 0.0063 ± 0.0022      | 0.0043 | 0.0087 |
| gdmc          | uniform     |      2 | 0.6721 ± 0.5537      | 0.3356 | 1.3112 |
| gdmc          | uniform     |      3 | 0.3424 ± 0.0809      | 0.2530 | 0.4104 |
| gdmc          | uniform     |      4 | 0.1454 ± 0.0189      | 0.1301 | 0.1665 |
| **gdmc**      | **uniform** |    **8** | **0.0973 ± 0.0031**  | **0.0954** | **0.1009** |
| gdmc          | uniform     |     16 | 0.6060 ± 0.0339      | 0.5673 | 0.6308 |
| gdmc          | uniform     |     32 | 0.6181 ± 0.0333      | 0.5798 | 0.6400 |
| gdmc-adaptive | uniform-pt  |      3 | 0.1190 ± 0.0076      | 0.1142 | 0.1277 |
| **gdmc-adaptive** | **uniform-pt** | **4** | **0.1077 ± 0.0033** | **0.1052** | **0.1114** |
| gdmc-adaptive | uniform-pt  |      8 | 0.1075 ± 0.0035      | 0.1049 | 0.1115 |
| momentum      | none        |     32 | 0.0772 ± 0.0134      | 0.0691 | 0.0927 |
| projected-gd  | uniform     |      2 | 9.2739 ± 4.8013      | 4.1973 | 13.742 |
| projected-gd  | uniform     |      3 | 0.8152 ± 0.0882      | 0.7328 | 0.9083 |
| projected-gd  | uniform     |      4 | 0.5690 ± 0.0337      | 0.5313 | 0.5964 |
| **projected-gd** | **uniform** | **8** | **0.0206 ± 0.0048** | **0.0151** | **0.0235** |
| projected-gd  | uniform     |     16 | 0.0077 ± 0.0039      | 0.0032 | 0.0105 |
| projected-gd  | uniform     |     32 | 0.0100 ± 0.0071      | 0.0033 | 0.0175 |
| sgd           | none        |     32 | 0.1166 ± 0.0024      | 0.1143 | 0.1192 |
| sgld          | none        |     32 | 0.2858 ± 0.2300      | 0.1521 | 0.5514 |

#### MNIST MLP (mean ± std, 3 seeds)

| optimizer     | grid_spec   |   bits | mean_std (test_acc)   |
|:--------------|:------------|-------:|:----------------------|
| adam          | none        |     32 | 0.9325 ± 0.0019       |
| gdmc          | uniform     |      2 | **0.7460 ± 0.0132**   |
| gdmc          | uniform     |      3 | **0.7776 ± 0.0064**   |
| gdmc          | uniform     |      4 | **0.8279 ± 0.0123**   |
| gdmc          | uniform     |      8 | 0.7745 ± 0.0065       |
| gdmc-adaptive | uniform-pt  |      3 | 0.7319 ± 0.0166       |
| gdmc-adaptive | uniform-pt  |      4 | 0.7366 ± 0.0101       |
| gdmc-adaptive | uniform-pt  |      8 | 0.7366 ± 0.0100       |
| momentum      | none        |     32 | 0.8566 ± 0.0039       |
| projected-gd  | uniform     |      2 | 0.1987 ± 0.0872       |
| projected-gd  | uniform     |      3 | 0.1984 ± 0.0862       |
| projected-gd  | uniform     |      4 | 0.1997 ± 0.0878       |
| projected-gd  | uniform     |      8 | 0.9284 ± 0.0147       |
| projected-gd  | uniform     |     16 | 0.9389 ± 0.0032       |
| projected-gd  | uniform     |     32 | 0.9298 ± 0.0044       |
| sgd           | none        |     32 | 0.3839 ± 0.0700       |
| sgld          | none        |     32 | 0.1110 ± 0.0049       |

The 16/32-bit gdmc numbers are very low (~9%) because with a 16/32-bit
grid the "one grid step" move is essentially a no-op (we use a
[−1, 1] range, so a 32-bit step is `2/2^32 ≈ 5e-10` — the model
can't move from initialization in 2 epochs). This is a known
limitation of "one-step" GDMC on fine grids; see the discussion below.

#### MNIST CNN (mean ± std, 3 seeds)

| optimizer     | grid_spec   |   bits | mean_std (test_acc)   |
|:--------------|:------------|-------:|:----------------------|
| adam          | none        |     32 | 0.8041 ± 0.0934       |
| gdmc          | uniform     |      4 | 0.5180 ± 0.2193       |
| gdmc-adaptive | uniform-pt  |      4 | 0.4018 ± 0.0314       |
| projected-gd  | uniform     |      4 | 0.1523 ± 0.0457       |
| projected-gd  | uniform     |      8 | 0.8708 ± 0.0407       |

(GDMC at 2-bit is below 30% accuracy; reported in the CSV but
omitted here for brevity.)

#### CIFAR-10 CNN (mean ± std, 3 seeds)

| optimizer     | grid_spec   |   bits | mean_std (test_acc)   |
|:--------------|:------------|-------:|:----------------------|
| adam          | none        |     32 | 0.2513 ± 0.0034       |
| gdmc          | uniform     |      4 | **0.2671 ± 0.0193**   |
| gdmc          | uniform     |      8 | **0.2775 ± 0.0061**   |
| gdmc-adaptive | uniform-pt  |      4 | 0.2702 ± 0.0151       |
| gdmc-adaptive | uniform-pt  |      8 | 0.2698 ± 0.0144       |
| momentum      | none        |     32 | 0.2766 ± 0.0304       |
| projected-gd  | uniform     |      4 | 0.1208 ± 0.0157       |
| projected-gd  | uniform     |      8 | 0.2727 ± 0.0302       |
| sgd           | none        |     32 | 0.2206 ± 0.0042       |

The 2-epoch budget is short for CIFAR-10; even the continuous baselines
only reach ~25% (vs 10% random). The interesting result is that GDMC
matches the continuous Adam at 4-8 bits while Projected-GD is much
worse at 4 bits (12% vs 27%).

### Acceptance rates

`results/plots/*_acceptance.png` show the final (smoothed) acceptance
rate of GDMC vs bits. The pattern is:

* At very coarse bits (2, 3), most moves are *rejected* because the
  gradient is unreliable on the discrete surface (many weights have
  the same gradient sign but different magnitudes → sign-of-grad is
  noisy).
* At 4-8 bits, almost every move is accepted (acceptance rate ≈ 1.0),
  meaning the local search is consistently moving in the right
  direction.
* At 16+ bits, acceptance stays at 1.0 but the *move magnitude* is so
  small that the model barely moves — a known limitation.

### β sweep (MNIST CNN, 4-bit, 3 seeds)

| β    | mean acc   | std     |
|------|------------|---------|
| 0.25 | 0.663      | 0.202   |
| 0.5  | 0.425      | 0.180   |
| 1.0  | 0.530      | 0.090   |
| 2.0  | 0.503      | 0.317   |
| 4.0  | 0.485      | 0.288   |

The differences are dominated by seed-to-seed variance at this short
horizon (2 epochs, 10K samples). The original protein-folding paper
used `β = 1.2e-3` to `2.4` depending on the loss magnitude; here the
CE loss is much smaller so β=2-4 is comparable.

## Discussion

### When does GDMC help?

The clearest case is **coarse quantization (≤ 4 bits)**:

* MNIST MLP 4-bit: GDMC **0.83** vs Projected-GD **0.20**.
* MNIST MLP 3-bit: GDMC **0.78** vs Projected-GD **0.20**.
* MNIST CNN 4-bit: GDMC **0.52** vs Projected-GD **0.15**.
* CIFAR-10 CNN 4-bit: GDMC **0.27** vs Projected-GD **0.12**.

In this regime, Projected-GD's Adam optimizer bounces off the grid
because the magnitude of the Adam step is mismatched to the grid
spacing. GDMC stays on the grid by construction and uses the *sign* of
the gradient, which is well-defined even when the magnitude is huge.

### When does GDMC hurt?

1. **At ≥ 8 bits**, Projected-GD with Adam dominates because Adam's
   adaptive scaling + projection is enough to land on a good grid
   point. GDMC's "one step" move is too small to make progress in the
   short horizon we tested.
2. **At 2 bits**, GDMC struggles because the grid only has 4 levels
   per weight — sign-of-grad is too noisy a signal.
3. **GDMC is roughly 2x more expensive per step** than Adam (it has
   to evaluate the loss twice). This didn't matter on CPU for our
   small models but is a real cost at scale.

### Limitations and future work

* **Short training horizon**: 2-3 epochs is not enough to see
  GDMC's full advantage on coarse grids. The paper used much longer
  runs (thousands of iterations per sequence).
* **"One grid step" move is suboptimal at fine grids**: a natural
  extension is to step `k` grid points at once, with `k` chosen by
  some line search or a trust region.
* **The fully adaptive grid (`AdaptiveGrid`) was retired** because
  the per-step range change produced runaway weight growth when the
  gradient sign is consistently negative. A safer variant would use a
  *slowly* updated range (e.g. EMA of `max|w|`).
* **No momentum**: the paper's per-direction moves are memoryless.
  Adding a momentum buffer to GDMC (in the spirit of Adam's `m`) is a
  natural improvement and should help convergence on fine grids.
* **Better baselines for comparison**: a recent SGLD variant with
  a Metropolis-Hastings acceptance (SG-MCMC) would be a closer
  comparison. We kept plain SGLD for simplicity.

## File map

```
plan.md                   The original plan.
README.md                 Quick-start.
requirements.txt          Python deps.

src/gdmc/                GDMC optimizer + grid utilities
  grid.py                UniformGrid, AdaptiveGrid, make_grid
  optimizer.py           GDMCOptimizer

src/baselines/           Five baseline optimizers
  sgd.py                 make_sgd
  momentum.py            make_momentum
  adam.py                make_adam
  projected_gd.py        ProjectedGD (wraps any torch.optim + projects)
  sgld.py                SGLD (Welling & Teh)

src/models/              MLP, SmallCNN
src/data/                Toy regression, MNIST, CIFAR-10 loaders
src/train.py             Generic training loop (handles closure-style opt)
src/runner.py            One-run runner + CSV writer

experiments/             5 experiment scripts (toy / mlp / cnn / cifar / β)
analysis/aggregate.py    Build tables + plots + REPORT.md from CSVs

results/
  raw/                   One CSV per experiment (one row per run)
  headline_table.{md,csv}
  plots/                 Per-task loss & acceptance curves
  REPORT.md              This file
```
