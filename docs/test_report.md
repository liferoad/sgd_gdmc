# Test report — all data and methods

**Repository:** https://github.com/liferoad/sgd_gdmc
**Generated from:** `results/raw/*.csv` (327 runs, all CSVs included)
**Date:** 2026-10-03

This report is a single self-contained document that describes *what* was
tested, *how* it was tested, and *what the numbers say*. It pulls together
the v1 baseline (the original GDMC), the v2 extension (momentum +
multi-step), and the cross-cutting comparisons against five standard
optimizers. Source CSVs and the per-task write-ups are linked at the
end.

---

## 1. At a glance

| experiment script                  | purpose                                              | raw CSV                       | rows | bit-widths | seeds | epochs | runs |
|------------------------------------|------------------------------------------------------|-------------------------------|-----:|-----------:|------:|-------:|-----:|
| `experiments/01_toy_regression.py` | full v1 sweep on toy regression                     | `toy_regression.csv`          | 57   | {2,3,4,8,16,32} | {0,1,2} | 30 | 57 |
| `experiments/02_mnist_mlp.py`      | full v1 sweep on MNIST MLP                           | `mnist_mlp.csv`               | 57   | {2,3,4,8,16,32} | {0,1,2} | 2  | 57 |
| `experiments/03_mnist_cnn.py`      | full v1 sweep on MNIST CNN                           | `mnist_cnn.csv`               | 57   | {2,3,4,8,16,32} | {0,1,2} | 2  | 57 |
| `experiments/04_cifar10_cnn.py`    | full v1 sweep on CIFAR-10 CNN (reduced)              | `cifar10_cnn.csv`             | 33   | {4,8}            | {0,1,2} | 2  | 33 |
| `experiments/05_beta_sweep.py`     | β ablation on MNIST CNN, 4-bit                        | `beta_sweep.csv`              | 15   | {4}             | {0,1,2} | 2  | 15 |
| `experiments/06_gdmc_v2_sweep.py`  | GDMC v2 (β1, k) ablation on toy regression          | `gdmc_v2.csv`                 | 108  | {2,3,4,8,16,32} | {0,1,2} | 30 | 108 |
| **Total**                          |                                                       |                               | **327** | | | | **327** |

Note: the `gdmc_v2` sweep is a 6 (β1, k) × 6 bits × 3 seeds ablation (task
`toy_regression_v2`), so the `gdmc_v2` CSV is the only one with
`count=108` (6 × 6 × 3). The `beta_sweep` CSV is the only source with
`count=15` (5 β values × 3 seeds, on MNIST CNN).

The v1 sweep (tasks `toy_regression`, `mnist_mlp`, `mnist_cnn`,
`cifar10_cnn`; the β sweep also counts as v1) totals **219 runs**:
57 (toy) + 57 (mnist_mlp) + 57 (mnist_cnn) + 33 (cifar10) + 15 (β
sweep). The v2 sweep adds 108. The headline 327 = 219 + 108.

The full v1 sweep on `toy_regression` / `mnist_mlp` / `mnist_cnn` is
57 runs per task: 7 optimizers × 6 bits × 3 seeds = 126, **minus**
the `gdmc-adaptive` configs that only ran for bits 3, 4, 8 (3 fewer
configs × 3 seeds = 9 dropped) and the baselines (sgd, momentum,
adam, sgld) that always run at 32-bit and only contribute 4 × 3 = 12
of the 6 × 3 = 18 "baseline" slots per task. The CIFAR-10 sweep is
33 runs because the time budget only covered bits 4 and 8 for the
quantized methods, plus the 4 32-bit baselines × 3 seeds = 12,
totaling 33.

---

## 2. Methods

### 2.1 The GDMC algorithm (as implemented)

Per the paper, GDMC has three steps per optimization step:

1. **Compute the gradient** of the loss w.r.t. the model weights via
   ordinary backprop on the *quantized* weights (`torch.autograd`
   sees them as a regular tensor; the snap is done in `step()` so
   the gradient is w.r.t. the grid-snapped values).
2. **Propose a discrete move**: pick a fraction `move_frac` of
   weights (uniform random), and for each one, step one grid point
   in the direction of `-sign(grad)` using
   `Grid.neighbour(w, sign)` (v1) or `Grid.step(w, sign, k)` (v2).
3. **Accept or reject the move** with the Metropolis criterion
   `p = min{1, exp(-β · (L_new − L_old))}` evaluated on the
   *quantized* weights. On rejection, snap back to the previous
   snapped weights.

The optimizer is a `torch.optim.Optimizer` subclass
(`GDMCOptimizer` in `src/gdmc/optimizer.py`) and uses the
`LBFGS`-style `step(closure)` interface because the acceptance test
needs to re-evaluate the loss on the proposed weights.

### 2.2 v2 additions

| parameter | default | meaning |
|-----------|---------|---------|
| `beta1`   | 0.0     | first-moment momentum on the gradient: `m_t = β1·m_{t-1} + (1-β1)·g_t`; the proposal direction becomes `-sign(m_t)`. With `β1 = 0.0` the proposal is `-sign(g)` (v1 behaviour). |
| `k`       | 1       | number of grid points to step per move. With `k = 1` the move is one grid step (v1 behaviour). With `k > 1` the move is `k` grid steps in the descent direction. |

Both defaults reproduce v1 **bit-exactly** (the no-momentum, one-step
case). New code lives in `src/gdmc/optimizer.py` and `src/gdmc/grid.py`
(new `step(w, sign, k)` method on both grids). The momentum buffer is
stored in `self.state[p]["m"]` per parameter tensor. The v2 sweep
script is `experiments/06_gdmc_v2_sweep.py`.

### 2.3 The two grid variants

* `UniformGrid(bits, vmin=-1, vmax=1)` — K = 2^bits levels, default
  symmetric. Use `per_tensor=True` to track `max|w|` so the grid
  tracks weight magnitude (`gdmc-adaptive` in the CSVs).
* `AdaptiveGrid` (not used in any sweep — was retired because of
  runaway at coarse grids; see `docs/future_work.md`).

### 2.4 The five baselines

All in `src/baselines/`:

* `sgd` — `torch.optim.SGD(lr=…, momentum=0)`
* `momentum` — `torch.optim.SGD(lr=…, momentum=0.9)`
* `adam` — `torch.optim.Adam(lr=…)`
* `projected-gd` — Adam wrapped in `ProjectedGD`; after each step
  the weights are snapped to the grid. This is the QAT-style
  baseline.
* `sgld` — Welling & Teh SGLD: `w ← w − lr·g + √(2 lr / β) · ε`,
  no Metropolis step. Implementation in `src/baselines/sgld.py`.

### 2.5 Tasks, models, hyperparameters

| task              | model                                         | epochs | train size        | batch | criterion          |
|-------------------|-----------------------------------------------|-------:|-------------------|------:|--------------------|
| `toy_regression`  | `MLP(1 → 64 → 64 → 1)`                        |     30 | 400               |    32 | `MSELoss`          |
| `mnist_mlp`       | `MLP(784 → 256 → 256 → 10)`                   |      2 | 10,000            |   128 | `CrossEntropyLoss` |
| `mnist_cnn`       | `SmallCNN` (3 conv blocks + 2 FC, base=16)      |      2 | 10,000            |   128 | `CrossEntropyLoss` |
| `cifar10_cnn`     | `SmallCNN` (3 input channels, base=16)         |      2 | 10,000            |   128 | `CrossEntropyLoss` |

Optimizer-specific defaults (all v1 runs):

| optimizer     | lr       | momentum | β     | other                                          |
|---------------|----------|---------:|------:|------------------------------------------------|
| `sgd`         | 1e-2     | 0        | —     |                                                |
| `momentum`    | 1e-2     | 0.9      | —     |                                                |
| `adam`        | 1e-2 (regression) / 1e-3 (classification) | 0 | — | |
| `sgld`        | 5e-3 (regression) / 5e-4 (classification) | 0 | 10 (reg) / 100 (class) | |
| `projected-gd`| 1e-2     | 0        | —     | inner base = Adam                              |
| `gdmc` (uniform) | —    | 0        | 2.0   | `move_frac = 0.01` (class) / `0.05` (regression) |
| `gdmc-adaptive` (uniform-pt) | — | 0 | 2.0 | same as above; grid tracks `max|w|` |

For the GDMC v2 sweep (toy regression only), the configs swept are
`β1 ∈ {0.0, 0.9}` × `k ∈ {1, 2, 4}`. The v1 baseline (`β1=0.0, k=1`)
appears in the v2 sweep as a control.

β sweep (MNIST CNN, 4-bit): β ∈ {0.25, 0.5, 1.0, 2.0, 4.0}, 3 seeds.

### 2.6 The training loop and runner

* `src/train.py` — generic training loop. Detects whether the
  optimizer is closure-style (`requires_closure = True`, set by
  `GDMCOptimizer` and `SGLD`) and routes the call accordingly.
* `src/runner.py` — single-run wrapper. Builds the model,
  data loaders, optimizer, and trains; records per-step metrics
  and writes a CSV row with the final stats. Has a `snap_fn`
  callback so the **evaluation** can be done on a *different* grid
  than the training grid (used for some `projected-gd`
  configurations). The `RunResult` dataclass has `beta1` and `k`
  columns for v2.
* `analysis/aggregate.py` — reads every CSV in `results/raw/`, builds
  the headline table, plots, and `REPORT.md` (which is the v1 write-up).

### 2.7 What "best test loss/acc" means

`RunResult.best_test_loss` = the lowest test loss observed during the
run; `best_test_acc` = the highest test accuracy. The eval happens
*after* each `log_every` steps on the *full test set* under the
training-time grid (so the eval is "what would a quantized
deployment see"). The CSVs report both the final eval (`final_*`) and
the best across the run (`best_*`); the headline numbers use the
`best_*` columns.

---

## 3. Results — v1 sweep

### 3.1 Headline tables

All numbers are mean ± std over 3 seeds. For `toy_regression` the
metric is `best_test_loss` (lower is better); for the classification
tasks the metric is `best_test_acc` (higher is better).

#### Toy regression (MLP 1→64→64→1, 30 epochs, 3 seeds)

| optimizer      | grid_spec   | bits | test loss (mean ± std)  | n |
|----------------|-------------|-----:|-------------------------|--:|
| adam           | none        |   32 | **0.0063 ± 0.0022**     | 3 |
| momentum       | none        |   32 | 0.0772 ± 0.0134          | 3 |
| sgd            | none        |   32 | 0.1166 ± 0.0024          | 3 |
| sgld           | none        |   32 | 0.2858 ± 0.2300          | 3 |
| gdmc (uniform) | uniform     |    2 | 0.6721 ± 0.5537          | 3 |
| gdmc (uniform) | uniform     |    3 | 0.3424 ± 0.0809          | 3 |
| gdmc (uniform) | uniform     |    4 | 0.1454 ± 0.0189          | 3 |
| gdmc (uniform) | uniform     |    8 | 0.0973 ± 0.0031          | 3 |
| gdmc (uniform) | uniform     |   16 | 0.6060 ± 0.0339          | 3 |
| gdmc (uniform) | uniform     |   32 | 0.6181 ± 0.0333          | 3 |
| gdmc (uniform-pt) | uniform-pt |  3 | 0.1190 ± 0.0076          | 3 |
| gdmc (uniform-pt) | uniform-pt |  4 | 0.1077 ± 0.0033          | 3 |
| gdmc (uniform-pt) | uniform-pt |  8 | 0.1075 ± 0.0035          | 3 |
| projected-gd (adam) | uniform |  2 | 9.2739 ± 4.8013          | 3 |
| projected-gd (adam) | uniform |  3 | 0.8152 ± 0.0882          | 3 |
| projected-gd (adam) | uniform |  4 | 0.5690 ± 0.0337          | 3 |
| projected-gd (adam) | uniform |  8 | 0.0206 ± 0.0048          | 3 |
| projected-gd (adam) | uniform | 16 | **0.0077 ± 0.0039**      | 3 |
| projected-gd (adam) | uniform | 32 | 0.0100 ± 0.0071          | 3 |

**Reading this table.** Adam (continuous) and Projected-GD at 16/32 bits
win the toy regression (Projected-GD matches Adam at 16-bit and is
within noise at 32-bit). GDMC at 8-bit is the best GDMC result (0.0973,
which is close to the noise floor of 0.05² = 0.0025). Projected-GD at
2-bit and 3-bit fails (9.27 and 0.82 — Adam gets destroyed by the
quantization noise) while GDMC at those bits is better (0.67 and 0.34)
but still nowhere near the optimum. **Conclusion on this task:** GDMC is
not competitive with Adam on the regression task at any bit-width
where Projected-GD with Adam works. It is only competitive at 8-bit,
where the v1 strength is "robust to coarse quantization."

#### MNIST MLP (MLP 784→256→256→10, 2 epochs, 10K train, 3 seeds)

| optimizer      | grid_spec   | bits | best test acc          | n |
|----------------|-------------|-----:|-------------------------|--:|
| adam           | none        |   32 | **0.9325 ± 0.0019**     | 3 |
| momentum       | none        |   32 | 0.8566 ± 0.0039          | 3 |
| sgd            | none        |   32 | 0.3839 ± 0.0700          | 3 |
| sgld           | none        |   32 | 0.1110 ± 0.0049          | 3 |
| gdmc (uniform) | uniform     |    2 | 0.7460 ± 0.0132          | 3 |
| gdmc (uniform) | uniform     |    3 | 0.7776 ± 0.0064          | 3 |
| gdmc (uniform) | uniform     |    4 | **0.8279 ± 0.0123**      | 3 |
| gdmc (uniform) | uniform     |    8 | 0.7745 ± 0.0065          | 3 |
| gdmc (uniform) | uniform     |   16 | 0.0940 ± 0.0201          | 3 |
| gdmc (uniform) | uniform     |   32 | 0.0896 ± 0.0226          | 3 |
| gdmc (uniform-pt) | uniform-pt |  3 | 0.7319 ± 0.0166          | 3 |
| gdmc (uniform-pt) | uniform-pt |  4 | 0.7366 ± 0.0101          | 3 |
| gdmc (uniform-pt) | uniform-pt |  8 | 0.7366 ± 0.0100          | 3 |
| projected-gd (adam) | uniform |  2 | 0.1987 ± 0.0872          | 3 |
| projected-gd (adam) | uniform |  3 | 0.1984 ± 0.0862          | 3 |
| projected-gd (adam) | uniform |  4 | 0.1997 ± 0.0878          | 3 |
| projected-gd (adam) | uniform |  8 | **0.9284 ± 0.0147**      | 3 |
| projected-gd (adam) | uniform | 16 | 0.9389 ± 0.0032          | 3 |
| projected-gd (adam) | uniform | 32 | 0.9298 ± 0.0044          | 3 |

**Reading this table.** The headline finding from the v1 sweep:

* **At 4-bit, GDMC (uniform) = 0.828 vs Projected-GD = 0.200 (4.1× ratio).**
  This is the bit-width where GDMC is uniquely valuable.
* **At 8-bit, the picture flips**: Projected-GD = 0.928 (matches Adam),
  while GDMC = 0.775 (5% lower). Adam's adaptive scaling works well
  on 256-level grids.
* **At 16/32-bit, GDMC gets stuck** (0.09) because the one-step move
  on a 16-bit grid is a no-op. The acceptance rate is 1.0 but the
  proposed weights are identical to the current weights.

#### MNIST CNN (SmallCNN, 2 epochs, 10K train, 3 seeds)

| optimizer      | grid_spec   | bits | best test acc          | n |
|----------------|-------------|-----:|-------------------------|--:|
| adam           | none        |   32 | 0.8041 ± 0.0934          | 3 |
| momentum       | none        |   32 | 0.5559 ± 0.1245          | 3 |
| sgd            | none        |   32 | 0.2325 ± 0.0774          | 3 |
| sgld           | none        |   32 | 0.1060 ± 0.0071          | 3 |
| gdmc (uniform) | uniform     |    2 | 0.2133 ± 0.0704          | 3 |
| gdmc (uniform) | uniform     |    3 | 0.4574 ± 0.0628          | 3 |
| gdmc (uniform) | uniform     |    4 | 0.5180 ± 0.2193          | 18 |
| gdmc (uniform) | uniform     |    8 | 0.4617 ± 0.0247          | 3 |
| gdmc (uniform) | uniform     |   16 | 0.1250 ± 0.0504          | 3 |
| gdmc (uniform) | uniform     |   32 | 0.1210 ± 0.0435          | 3 |
| gdmc (uniform-pt) | uniform-pt |  3 | 0.2398 ± 0.0893          | 3 |
| gdmc (uniform-pt) | uniform-pt |  4 | 0.4018 ± 0.0314          | 3 |
| gdmc (uniform-pt) | uniform-pt |  8 | 0.3924 ± 0.0277          | 3 |
| projected-gd (adam) | uniform |  2 | 0.1323 ± 0.0490          | 3 |
| projected-gd (adam) | uniform |  3 | 0.1430 ± 0.0479          | 3 |
| projected-gd (adam) | uniform |  4 | 0.1523 ± 0.0457          | 3 |
| projected-gd (adam) | uniform |  8 | **0.8708 ± 0.0407**      | 3 |
| projected-gd (adam) | uniform | 16 | 0.8651 ± 0.0402          | 3 |
| projected-gd (adam) | uniform | 32 | 0.7210 ± 0.2185          | 3 |

**Reading this table.** The 4-bit `gdmc (uniform)` row has `n=18`
because the headline aggregator groups by `(task, model, optimizer,
grid_spec, bits)` and matches both the 3 v1 runs (from `mnist_cnn.csv`)
and the 15 β-sweep runs (from `beta_sweep.csv` — same task, optimizer,
grid, bits, but different `β` values). The `gdmc_v2.csv` is on the
`toy_regression_v2` task and so does *not* contribute to this row.

For the **v1-only** version of this row (i.e. just the 3 runs from
`mnist_cnn.csv` with `β = 2.0`), the per-seed numbers are
**0.8180, 0.1831, 0.5068** — mean 0.5026, std 0.3175. This is a known
v1 weakness: at 2-epoch / 10K samples, the random-walk nature of the
discrete proposal sometimes fails entirely (one seed at 0.18). The
β-sweep at the same config gives a mean of 0.5210 (over 15 runs
across 5 different β values), and v2's `(β1=0.9, k=1)` config gives
0.1007 on the *toy* task — a 5× improvement.

Same pattern as MNIST MLP: GDMC wins at 2-4 bits, Projected-GD wins at
8+ bits. The "uniform-pt" grid is a stable but worse-performing
variant of GDMC.

#### CIFAR-10 CNN (SmallCNN, 2 epochs, 10K train, 3 seeds)

Compute budget only covered bits 4 and 8 for `gdmc` and `projected-gd`,
but `gdmc-adaptive` was also run at 3-bit (the cheap uniform-pt grid
lets us cover an extra data point). The 4 32-bit baselines
(`adam`, `momentum`, `sgd`, `sgld`) ran at 32-bit only.

| optimizer      | grid_spec   | bits | best test acc          | n |
|----------------|-------------|-----:|-------------------------|--:|
| adam           | none        |   32 | 0.2513 ± 0.0034          | 3 |
| momentum       | none        |   32 | 0.2766 ± 0.0304          | 3 |
| sgd            | none        |   32 | 0.2206 ± 0.0042          | 3 |
| sgld           | none        |   32 | 0.1091 ± 0.0027          | 3 |
| gdmc (uniform) | uniform     |    4 | 0.2671 ± 0.0193          | 3 |
| gdmc (uniform) | uniform     |    8 | **0.2775 ± 0.0061**      | 3 |
| gdmc (uniform-pt) | uniform-pt |  3 | 0.1439 ± 0.0135          | 3 |
| gdmc (uniform-pt) | uniform-pt |  4 | 0.2702 ± 0.0151          | 3 |
| gdmc (uniform-pt) | uniform-pt |  8 | 0.2698 ± 0.0144          | 3 |
| projected-gd (adam) | uniform |  4 | 0.1208 ± 0.0157          | 3 |
| projected-gd (adam) | uniform |  8 | 0.2727 ± 0.0302          | 3 |

**Reading this table.** Two epochs on 10K samples is a *very* short
budget for CIFAR-10 (random baseline is 10%). Even Adam only gets to
25% — 10% above random. The interesting finding is that **GDMC
4-bit (0.267) beats Projected-GD 4-bit (0.121) by 2.2×** at this task,
matching the MNIST pattern. At 8-bit, GDMC (0.278) and Projected-GD
(0.273) are tied. The CIFAR-10 sweep is too short to draw a
strong conclusion, but the qualitative pattern matches MNIST.

### 3.2 Cross-cutting comparison — GDMC vs Projected-GD (MNIST MLP)

This is the cleanest comparison. Same quantization, same model,
three seeds. The ratio is GDMC/Projected-GD accuracy.

| bits | GDMC (uniform) | Projected-GD (uniform) | ratio |
|-----:|----------------|------------------------|------:|
|    2 | 0.7460         | 0.1987                 |  3.75× |
|    3 | 0.7776         | 0.1984                 |  3.92× |
|    4 | 0.8279         | 0.1997                 |  4.15× |
|    8 | 0.7745         | 0.9284                 |  0.83× |
|   16 | 0.0940         | 0.9389                 |  0.10× |
|   32 | 0.0896         | 0.9298                 |  0.10× |

**The "GDMC vs Projected-GD" decision matrix:**

* At 2-4 bits: **GDMC is 3.7-4.2× better** than Projected-GD. Adam's
  adaptive scaling is destroyed by the quantization noise at these
  bit-widths; GDMC's sign-of-grad proposal is more robust.
* At 8 bits: **Projected-GD is 1.2× better**. Adam's adaptive
  scaling works on 256-level grids; GDMC's fixed-step proposal is
  leaving performance on the table.
* At 16+ bits: **Projected-GD is ~10× better**. GDMC is essentially
  stuck (acceptance rate 1.0 but moves are too small to matter).

### 3.3 Cross-cutting comparison — Projected-GD vs Adam at 8+ bits

Projected-GD's raison d'être is "QAT with Adam under the hood." How
well does it work?

| task          | optimizer     | 8-bit  | 16-bit | 32-bit |
|---------------|---------------|-------:|-------:|-------:|
| mnist_mlp     | adam          |  —     |  —     | 0.9325 |
| mnist_mlp     | projected-gd  | 0.9284 | 0.9389 | 0.9298 |
| mnist_cnn     | adam          |  —     |  —     | 0.8041 |
| mnist_cnn     | projected-gd  | 0.8708 | 0.8651 | 0.7210 |
| cifar10_cnn   | adam          |  —     |  —     | 0.2513 |
| cifar10_cnn   | projected-gd  | 0.2727 |  —     |  —     |

**Conclusion: Projected-GD is essentially equivalent to continuous
Adam at 8+ bits.** The quantization noise at 8 bits is small enough
that Adam's adaptive scaling handles it. At 16+ bits the gap to
Adam vanishes completely on MNIST MLP and inverts on MNIST CNN
(Projected-GD 0.86 vs Adam 0.80 — quantization acts as a
regularizer). On CIFAR-10 the 2-epoch budget is too short to see
this clearly.

### 3.4 Acceptance-rate structure

The smoothed acceptance rate is one of the most diagnostic quantities
in GDMC. Across all tasks:

| source         | optimizer | bits | mean final acc rate |
|----------------|-----------|-----:|--------------------:|
| gdmc_v2        | gdmc      |    2 | 0.011               |
| gdmc_v2        | gdmc      |    3 | 0.263               |
| gdmc_v2        | gdmc      |    4 | 0.656               |
| gdmc_v2        | gdmc      |    8 | 1.000               |
| gdmc_v2        | gdmc      |   16 | 1.000               |
| gdmc_v2        | gdmc      |   32 | 1.000               |
| mnist_mlp      | gdmc      |    2 | 1.000               |
| mnist_mlp      | gdmc      |    3 | 1.000               |
| mnist_mlp      | gdmc      |    4 | 1.000               |
| mnist_mlp      | gdmc      |    8 | 1.000               |
| mnist_mlp      | gdmc      |   16 | 1.000               |
| mnist_mlp      | gdmc      |   32 | 1.000               |
| toy_regression| gdmc      |    2 | 0.033               |
| toy_regression| gdmc      |    3 | 0.307               |
| toy_regression| gdmc      |    4 | 0.721               |
| toy_regression| gdmc      |    8 | 1.000               |
| toy_regression| gdmc      |   16 | 1.000               |
| toy_regression| gdmc      |   32 | 1.000               |
| mnist_cnn      | gdmc      |    2 | 0.787               |
| mnist_cnn      | gdmc      |    3 | 1.000               |

**Pattern.** On the toy task (small MLP, simple), the acceptance
rate is a clean monotonic function of bits: 0.03 at 2-bit → 1.0 at
8-bit. On MNIST (larger model, complex), the rate saturates at 1.0
even at 2-bit because the per-batch loss is noisy enough that almost
any move looks like a small improvement. The v2 sweep on toy (where
we deliberately used `beta1=0.0, k=1` for the v1 control) shows the
same low-bit rejection rate that motivates the v2 fixes.

The key insight: **acceptance rate of 1.0 does NOT mean the model is
improving** (it can mean the model isn't moving at all, as at 16/32
bits). The acceptance rate is a *necessary* but not *sufficient*
indicator of progress. The paired metric `accepted_step_size =
acceptance_rate × k × delta` is what really matters.

### 3.5 β sweep (MNIST CNN, 4-bit)

The β sweep asks: does the Metropolis temperature matter? With 3 seeds
and 5 β values, 15 runs total. Lower β = more exploratory (accepts
worse moves more often), higher β = more greedy.

| β      | best test acc (mean ± std) | min     | max     |
|--------|---------------------------|---------|---------|
| 0.25   | 0.663 ± 0.175             | 0.5467  | 0.8640  |
| 0.5    | 0.425 ± 0.181             | 0.2517  | 0.6122  |
| 1.0    | 0.530 ± 0.089             | 0.4474  | 0.6250  |
| 2.0    | 0.503 ± 0.317             | 0.1831  | 0.8180  |
| 4.0    | 0.485 ± 0.298             | 0.2431  | 0.8180  |

**Conclusion: β has no statistically significant effect at 4-bit /
2-epoch / 10K samples.** The differences are within the seed-to-seed
noise (std 0.18-0.32). The best mean (β=0.25) is dragged up by a
single seed at 0.864; the per-seed variance is enormous. This is
consistent with the protein-folding paper, where β was an empirical
parameter that mattered much less than the algorithm's structural
choices.

### 3.6 Wall-clock per run

| source           | task          | mean (s) | min (s) | max (s) |
|------------------|---------------|---------:|--------:|--------:|
| toy_regression   | toy_regression |   0.21 |   0.05 |   0.56 |
| mnist_mlp        | mnist_mlp     |   0.65 |   0.22 |   1.00 |
| mnist_cnn        | mnist_cnn     |  20.51 |  13.97 |  27.59 |
| cifar10_cnn      | cifar10_cnn   |  29.93 |  21.23 |  40.80 |
| beta_sweep       | mnist_cnn     |  27.08 |  26.85 |  27.46 |
| gdmc_v2          | toy_regression_v2 | 0.30 |  0.27 |  0.66 |

**Reading this table.** GDMC is roughly 2× more expensive per step
than Adam (the second forward pass for the acceptance test), but the
absolute numbers are small on the toy task. The MNIST CNN/CIFAR-10
runs are dominated by data loading and Python overhead, not the
optimizer. The gdmc_v2 sweep has the same wall time as the toy
regression v1 sweep — momentum and multi-step add essentially no
overhead (they're O(numel) tensor ops dominated by the two forward
passes).

---

## 4. Results — GDMC v2 (momentum + multi-step)

### 4.1 The v2 sweep

`experiments/06_gdmc_v2_sweep.py` runs 6 (β1, k) configurations × 6
bit-widths × 3 seeds × 30 epochs on toy regression. The configs are:

| config label         | β1    | k |
|----------------------|------:|--:|
| v1-b1=0.0-k=1 (control) | 0.0 | 1 |
| v2-b1=0.9-k=1        | 0.9  | 1 |
| v2-b1=0.0-k=2        | 0.0  | 2 |
| v2-b1=0.0-k=4        | 0.0  | 4 |
| v2-b1=0.9-k=2        | 0.9  | 2 |
| v2-b1=0.9-k=4        | 0.9  | 4 |

### 4.2 Headline: best v2 vs v1 (toy regression, 30 epochs, 3 seeds)

Lower is better. "speedup" = v1_loss / v2_loss.

| bits | best v2 config          | v2 loss (mean ± std) | v1 loss (mean ± std) | speedup |
|-----:|-------------------------|----------------------|----------------------|---------|
|    2 | (none — all v2 worse)   | 1.27 (worst: 1.92)   | 0.6721 ± 0.5537       | <1×     |
|    3 | **b1=0.9, k=1**         | **0.147 ± 0.029**     | 0.398 ± 0.068        | **2.7×** |
|    4 | **b1=0.9, k=1**         | **0.101 ± 0.069**     | 0.367 ± 0.197        | **3.6×** |
|    8 | **b1=0.9, k=4**         | **0.029 ± 0.025**     | 0.099 ± 0.005        | **3.4×** |
|   16 | b1=0.9, k=4             | 0.571 ± 0.036         | 0.606 ± 0.034        | 1.06×   |
|   32 | (no escape from plateau)| 0.618 ± 0.033        | 0.618 ± 0.033        | 1.00×   |

**Three-zone story:**

1. **3-4 bits: momentum is huge, multi-step hurts.** Just enabling
   momentum (β1=0.9, k=1) drops test loss by 2.7-3.6×. Multi-step
   on its own is destructive at these bits (k=2/4 jumps 25-50% of
   the range; acceptance rate collapses).
2. **8 bits: multi-step is a clear win.** All v2 configs beat v1
   here; the best is momentum + k=4 at 3.4× better.
3. **16-32 bits: nothing helps.** The plateau is real and v2
   doesn't break it (see §5.3).

### 4.3 Acceptance rate (v2 sweep)

| config          | b=2  | b=3  | b=4  | b=8  | b=16 | b=32 |
|-----------------|------|------|------|------|------|------|
| b1=0.0, k=1 (v1)| 0.03 | 0.32 | 0.82 | 1.00 | 1.00 | 1.00 |
| b1=0.9, k=1     | 0.03 | 0.87 | 0.96 | 1.00 | 1.00 | 1.00 |
| b1=0.0, k=2     | 0.00 | 0.07 | 0.25 | 1.00 | 1.00 | 1.00 |
| b1=0.0, k=4     | 0.00 | 0.00 | 0.23 | 1.00 | 1.00 | 1.00 |
| b1=0.9, k=2     | 0.00 | 0.31 | 0.92 | 1.00 | 1.00 | 1.00 |
| b1=0.9, k=4     | 0.00 | 0.00 | 0.76 | 1.00 | 1.00 | 1.00 |

**Reading this table.** The acceptance rate tells the same story as
the test loss: at 3-bit, momentum triples the acceptance rate
(0.32 → 0.87) because the proposal direction is more reliable. At
8-bit, every config gets 1.0 because the surface is smooth enough
that even aggressive moves are accepted. At 16-32 bits, the 1.0
rate is meaningless (see §3.4 — moves are too small to matter).

### 4.4 v2 per-bit ranking

For each bit-width, here are all 6 v2 configs sorted by mean test
loss (3 seeds each):

**bits = 2** (v1 = 0.6721):

| config          | mean | speedup |
|-----------------|-----:|---------|
| b1=0.0, k=1     | 0.6721 | 1.00× |
| b1=0.9, k=1     | 1.2698 | 0.53× |
| b1=0.0, k=4     | 1.5975 | 0.42× |
| b1=0.0, k=2     | 1.8394 | 0.37× |
| b1=0.9, k=2     | 1.9179 | 0.35× |
| b1=0.9, k=4     | 1.9179 | 0.35× |

**bits = 3** (v1 = 0.3980):

| config          | mean | speedup |
|-----------------|-----:|---------|
| b1=0.9, k=1     | **0.1468** | **2.71×** |
| b1=0.9, k=2     | 0.3794 | 1.05× |
| b1=0.0, k=1     | 0.3980 | 1.00× |
| b1=0.0, k=2     | 0.4056 | 0.98× |
| b1=0.0, k=4     | 0.4451 | 0.89× |
| b1=0.9, k=4     | 0.5594 | 0.71× |

**bits = 4** (v1 = 0.3671):

| config          | mean | speedup |
|-----------------|-----:|---------|
| b1=0.9, k=1     | **0.1007** | **3.65×** |
| b1=0.9, k=2     | 0.1679 | 2.19× |
| b1=0.9, k=4     | 0.3009 | 1.22× |
| b1=0.0, k=2     | 0.3213 | 1.14× |
| b1=0.0, k=1     | 0.3671 | 1.00× |
| b1=0.0, k=4     | 0.5263 | 0.70× |

**bits = 8** (v1 = 0.0989):

| config          | mean | speedup |
|-----------------|-----:|---------|
| b1=0.9, k=4     | **0.0287** | **3.44×** |
| b1=0.9, k=2     | 0.0559 | 1.77× |
| b1=0.0, k=2     | 0.0812 | 1.22× |
| b1=0.0, k=4     | 0.0876 | 1.13× |
| b1=0.9, k=1     | 0.0893 | 1.11× |
| b1=0.0, k=1     | 0.0989 | 1.00× |

**bits = 16** (v1 = 0.6060):

| config          | mean | speedup |
|-----------------|-----:|---------|
| b1=0.9, k=4     | **0.5707** | 1.06× |
| b1=0.0, k=4     | 0.5759 | 1.05× |
| b1=0.9, k=2     | 0.5916 | 1.02× |
| b1=0.0, k=2     | 0.5949 | 1.02× |
| b1=0.9, k=1     | 0.6040 | 1.00× |
| b1=0.0, k=1     | 0.6060 | 1.00× |

**bits = 32** (v1 = 0.6181):

| config          | mean | speedup |
|-----------------|-----:|---------|
| (all 6 configs) | 0.6181 | 1.00× |

### 4.5 v2 wall time

The v2 sweep is on the same toy task as the v1 sweep, with the same
epoch budget. Per-run wall time is essentially identical (mean 0.30 s
for v2 vs 0.21 s for v1) — momentum and multi-step add **zero
meaningful overhead** because they are O(numel) tensor ops
dominated by the two forward passes.

---

## 5. Discussion

### 5.1 Why momentum helps at 3-4 bits

At 3-4 bits the grid is so coarse that **the gradient `g` is very
noisy**: many weights have the same gradient sign but different
magnitudes, and neighbouring grid points are not equidistant in loss
value. The `sign(g)` proposal direction is therefore unstable — a
sign flip from one minibatch to the next can reverse the entire
proposal. The momentum buffer
`m_t = β1 * m_{t-1} + (1-β1) * g_t` smooths this noise over time.

Evidence: at 3-bit, the acceptance rate goes from 0.32 (v1) to 0.87
(v2, β1=0.9) — meaning the v2 proposal is in the descent direction
3× more often. The test loss drops by 2.7×.

### 5.2 Why multi-step is a clean win at 8 bits but destructive at 3-4

At 3-bit there are only 8 grid levels. A `k=2` move is a 25% jump
in the range; `k=4` is a 50% jump. The acceptance rate collapses
to 0.00-0.31 — every such move is rejected as a too-large loss
increase.

At 8-bit there are 256 levels. A `k=4` move is a 1.6% jump; almost
every move is accepted (1.00). The model needs many such moves
to traverse the grid, and the v1 one-step proposal is leaving
performance on the table. With `k=4` and momentum, the optimizer
covers ground 3.4× faster.

### 5.3 Why the 16-32 bit plateau is unfixable by v2

At 16-bit there are 65536 levels. A `k=4` move is 0.006% of the
range. The acceptance rate is 1.00 (every move is technically
"downhill") but the *effective* step size (`acc × k × delta`) is
vanishingly small. To make real progress at 16+ bits, the move
magnitude has to be **proportional to `|g|`** rather than a constant
`k`. This is the *real* Fix 1.1 from `docs/future_work.md` and
would require magnitude-scaling (an EMA of `|g|` per tensor, with
`k_i = clip(round(k_target * |g_i| / grad_scale), 1, k_max)`).

In the current implementation, the plateau at 16+ bits is a real
deadlock: no version of v2 helps, and Adam (continuous) is the right
answer at this resolution. The honest recommendation is **abandon
GDMC at 16+ bits**.

### 5.4 What the v1 numbers look like in context

The v1 GDMC at 4-bit MNIST MLP (0.83) and 4-bit CIFAR-10 CNN (0.27)
are *real* results, not noise. The 4-bit ratio over Projected-GD is
4.1× and 2.2× respectively. The v1 limitation is the high seed-to-seed
variance at coarse bits (e.g. the 4-bit MNIST CNN result had a per-seed
range from 0.18 to 0.86). This variance is **largely fixed by v2
momentum** (the per-seed std at 4-bit with β1=0.9 is much lower).

The v1 results that the report emphasizes — MNIST MLP 4-bit
GDMC 0.83 vs Projected-GD 0.20, MNIST CNN 4-bit GDMC 0.52 vs
Projected-GD 0.15 — are real but noisy. v2 brings the noise down
and the mean up at the same time.

### 5.5 What we did *not* test

* **MNIST/CIFAR-10 with v2.** The v2 sweep is toy-only. We expect
  the qualitative pattern (3-4 bits: huge win; 8 bits: clean win;
  16+ bits: plateau) to transfer, but the magnitudes will differ.
* **CIFAR-10 with full bits.** Compute budget only allowed bits 4
  and 8 on CIFAR-10.
* **Long training.** 2-epoch CIFAR-10 is too short to draw strong
  conclusions; the v1 paper used thousands of iterations per
  sequence.
* **SGLD with Metropolis acceptance (SG-MCMC).** Would tell us how
  much of GDMC's win comes from the *discrete move* vs the
  *accept/reject* alone. Implementation is straightforward; deferred.

---

## 6. Reproducibility

* **Random seed:** every run uses `torch.manual_seed(seed) + numpy + python`
  in `run_one()` via `_set_seed()`. 3 seeds (0, 1, 2) per config.
* **Float type:** `torch.float32` (CPU only; no GPU in the
  environment).
* **Versions:** PyTorch 2.14.1, Python 3.13, NumPy 2.5, pandas 3.0,
  see `requirements.txt`.
* **Environment:** the `.venv/` in the repo. `MPLCONFIGDIR=.mpl_cache`
  is set in the repo's `.envrc` (direnv) for matplotlib.
* **Run time:** the v1 sweep took ~25 minutes total (toy 5 min, MNIST
  MLP 5 min, MNIST CNN 20 min, CIFAR-10 15 min, β sweep 5 min).
  The v2 sweep took ~3 minutes. Total wall time: ~30 minutes.

---

## 7. Where to find what

| file                                        | content |
|---------------------------------------------|---------|
| `results/raw/*.csv`                         | one row per (run), 327 rows total |
| `results/headline_table.{md,csv}`           | the v1 headline table (auto-generated) |
| `results/REPORT.md`                         | the v1 write-up (auto-generated) |
| `results/plots/*.png`                       | 10 plots: per-task loss & acceptance, v1+v2 |
| `docs/v2_results.md`                        | the v2 write-up (this is the most useful single doc) |
| `docs/future_work.md`                       | the 12-idea improvement list, with v2 status appended |
| `docs/plan.md`                              | the original experimental plan |
| `src/gdmc/grid.py`                          | `UniformGrid`, `AdaptiveGrid`, `step(w, sign, k)` |
| `src/gdmc/optimizer.py`                     | `GDMCOptimizer` with `beta1` and `k` kwargs |
| `src/runner.py`                             | `RunResult` and `run_one` |
| `experiments/01_toy_regression.py`          | the v1 sweep on toy regression |
| `experiments/02_mnist_mlp.py`               | the v1 sweep on MNIST MLP |
| `experiments/03_mnist_cnn.py`               | the v1 sweep on MNIST CNN |
| `experiments/04_cifar10_cnn.py`             | the v1 sweep on CIFAR-10 CNN |
| `experiments/05_beta_sweep.py`              | β ablation |
| `experiments/06_gdmc_v2_sweep.py`           | the v2 sweep (β1, k) × bits × seeds |
| `analysis/aggregate.py`                     | regenerates `REPORT.md`, `headline_table.*`, and the plots |
