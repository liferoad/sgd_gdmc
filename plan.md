# GDMC for Deep Learning Weight Optimization — Plan

## Idea (from Hu, Beratan, Yang, J. Chem. Phys. 131, 154117, 2009)

GDMC (Gradient-Directed Monte Carlo) optimizes over a **discrete** variable
space by:

1. **Building a "virtual continuous surface"** over the discrete variables
   so that analytical or numerical gradients ∂f/∂xᵢ are defined.
2. **Using the gradient to propose a discrete move** (in the protein
   example: rank sites by gradient, set the Nᴴ lowest-gradient sites to H,
   others to P; in the protein-folding example: compute the per-direction
   numerical gradient of a pull move and pick the steepest one).
3. **Accepting/rejecting the move with the Metropolis criterion**
   `p = min{1, exp(−β(Eᵢ₊₁ − Eᵢ))}`. The stochastic acceptance lets the
   search escape local minima that pure gradient descent would get stuck in.
4. A temperature parameter β controls the trade-off between
   gradient-directedness and exploration.

## Translation to deep learning

- **Discretize the weight space to a grid** (uniform or adaptive). Each
  weight w has a discrete set of allowed values, e.g. {−1, 0, 1} (binary)
  or a uniform grid of K levels in [w_min, w_max].
- **Compute the normal gradient** of the loss w.r.t. weights wᵢ via
  backprop. The gradient tells us which direction on the grid to move the
  weight.
- **Propose a discrete move**: pick a weight (or a block) and snap it to
  the neighbouring grid point in the direction of −∂L/∂w. This is the
  "gradient-directed" proposal.
- **Accept/reject with Metropolis**: `p = min{1, exp(−β(L_new − L_old))}`.
  The acceptance test uses the **discretized** loss so the energy landscape
  is the actual quantized one.
- For **adaptive grids**, periodically refine the grid around the current
  weights (center the grid on the current value, shrink the spacing).

This is **not** the same as just adding gradient noise (SGLD) because:

- Proposals are restricted to grid points → we always evaluate the
  *quantized* model.
- The acceptance test ensures the discrete energy never drifts upward
  without a probability bound.

## Tasks / Models

1. **Toy regression** — fit a small MLP on a synthetic 1-D regression
   problem. Easy to visualize the loss landscape and verify GDMC reaches
   the same minimum as continuous baselines.
2. **MNIST** — MLP and small CNN. ~1–3% test error achievable.
3. **CIFAR-10** — small CNN (3-conv + 2-FC, or a tiny VGG-like net),
   aiming for 80–88% test accuracy.

All three will be evaluated under the same **quantization constraint** so
the comparison is fair.

## Quantization levels (sweep)

`{2, 3, 4, 8, 16, 32}` bits. 1 bit would be ternary-without-zero; 32 bits
is essentially continuous.

## Baselines

- **Plain SGD**
- **Momentum SGD** (PyTorch `SGD(momentum=0.9)`)
- **Adam** (PyTorch `Adam`)
- **Projected GD** (QAT-style: normal GD, then project to grid each step)
- **SGLD** (stochastic gradient Langevin dynamics — continuous-space MCMC
  cousin of GDMC, included to show the value of the *discrete* move)
- **GDMC (ours)** with two grid variants:
  - **Uniform grid** of K levels in [−1, 1] (or data-driven min/max).
  - **Adaptive grid** — per-tensor grid centered on current values with
    shrinking spacing.

## Evaluation metrics

For each (optimizer × grid setting × seed) run, record:

- Final test accuracy / final test loss
- Training loss curve (per epoch and per step)
- Convergence speed: steps to reach a target loss or accuracy threshold
- Acceptance rate of the Metropolis step over time (for GDMC)
- Quantization error ||w_quantized − w_continuous||
- Run ≥ 3 seeds, report mean ± std

## Ablations

| # | Question | What we vary |
|---|----------|--------------|
| A1 | Does grid coarseness matter? | bits = 2, 3, 4, 8, 16, 32 |
| A2 | Uniform vs adaptive grid | both implemented |
| A3 | Effect of β | sweep over a few orders of magnitude |
| A4 | Effect of move-set size | propose 1 / 10 / 100% of weights per step |
| A5 | Acceptance batch | full / minibatch / held-out fixed batch |
| A6 | Effect of network size | tiny → CIFAR CNN |

## Run budget

- 3 seeds for headline numbers
- 5-point β sweep on the main task (MNIST CNN) for ablation
- One full ablation table

## Risks / honest caveats

- GDMC evaluates the loss twice per step. For MNIST/CIFAR-10 this is
  fine; for ImageNet it would be expensive.
- The right β is problem-dependent (the paper also had to tune it).
- Expected result: GDMC is competitive but rarely a clear winner on
  standard benchmarks. The likely interesting finding is on **coarse
  grids**, where the Metropolis step helps it find **better
  low-precision weights** than projected-GD.
- Acceptance rate is a useful diagnostic that drops as the model
  converges.

## Deliverables

1. A clean `gdmc` Python package with the optimizer and grid utilities.
2. Headline comparison table: optimizer × task (test accuracy, final
   loss, convergence steps).
3. Loss curves and acceptance-rate plots.
4. Ablation sweep results (β, K, move-set) for the main task.
5. A short `results/REPORT.md` summarizing findings with figures and
   tables, and an honest discussion of where GDMC does and doesn't help.
