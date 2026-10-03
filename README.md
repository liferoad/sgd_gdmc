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

See `plan.md` for the full experimental plan and `results/REPORT.md`
(written by `analysis/aggregate.py`) for the final write-up.
