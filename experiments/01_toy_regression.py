"""Toy 1-D regression sweep.

Runs the 6 optimizers (SGD, Momentum, Adam, Projected-GD, SGLD, GDMC)
with several quantization levels and a few seeds. Writes a single CSV
to ``results/raw/toy_regression.csv``.

The script is intentionally lightweight — the toy problem is small
enough to run all configs in a few minutes — so it also doubles as the
end-to-end smoke test for the pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.toy_regression import ToyRegression
from src.models.mlp import MLP
from src.runner import RunResult, run_one, write_results_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bits-list", type=int, nargs="+", default=[2, 3, 4, 8, 16, 32])
    parser.add_argument("--out", type=str, default="results/raw/toy_regression.csv")
    args = parser.parse_args()

    # Data
    train_ds = ToyRegression(N=400, noise=0.05, seed=0)
    test_ds = ToyRegression(N=200, noise=0.05, seed=1)
    batch_size = 32
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    te = DataLoader(test_ds, batch_size=128, shuffle=False)
    criterion = nn.MSELoss()

    def model_fn():
        return MLP(in_dim=1, hidden=(64, 64), out_dim=1)

    # Configurations to run.
    # Each entry: (optimizer, grid_spec, bits, kwargs)
    cfgs = []

    # Continuous baselines (no quantization, but we still evaluate under
    # the same grid as the GDMC runs to be fair — see ``eval_grid``).
    for opt in ["sgd", "momentum", "adam", "sgld"]:
        cfgs.append((opt, "none", 32, dict(lr=1e-2 if opt != "sgld" else 5e-3,
                                            momentum=0.9 if opt == "momentum" else 0.0,
                                            beta=10.0 if opt == "sgld" else 0.0,
                                            move_frac=0.0)))

    # Projected-GD: train with Adam, project to grid each step.
    for bits in args.bits_list:
        cfgs.append(("projected-gd", "uniform", bits, dict(lr=1e-2, project_base="adam",
                                                          momentum=0.0, beta=0.0,
                                                          move_frac=0.0)))

    # GDMC uniform
    for bits in args.bits_list:
        cfgs.append(("gdmc", "uniform", bits, dict(lr=0.0, beta=2.0, momentum=0.0,
                                                   move_frac=0.05)))

    # GDMC per-tensor-uniform (range tracks weight magnitude, but the
    # grid is fixed-shape per step, so it's more stable than a fully
    # adaptive grid).
    for bits in [3, 4, 8]:
        cfgs.append(("gdmc-adaptive", "uniform-pt", bits, dict(lr=0.0, beta=2.0,
                                                               momentum=0.0, move_frac=0.05)))

    results = []
    for seed in args.seeds:
        for opt, grid_spec, bits, hp in cfgs:
            label = f"{opt:>14s}/{grid_spec:>8s}/b{bits:02d}/s{seed}"
            t0 = time.time()
            res = run_one(
                task="toy_regression",
                model_name="mlp_64x64",
                model_fn=model_fn,
                optimizer_name=opt,
                grid_spec=grid_spec,
                bits=bits,
                train_loader=tr,
                test_loader=te,
                criterion=criterion,
                epochs=args.epochs,
                batch_size=batch_size,
                lr=hp.get("lr", 0.0),
                momentum=hp.get("momentum", 0.0),
                beta=hp.get("beta", 0.0),
                move_frac=hp.get("move_frac", 0.0),
                seed=seed,
                device="cpu",
                is_classification=False,
                log_every=200,
                extra={"project_base": hp.get("project_base", "adam")},
                # Continuous baselines are also evaluated under the same
                # grid as the GDMC runs (we project to that grid before
                # evaluating). For bits=32 the grid is essentially
                # continuous so this is a no-op.
                eval_grid_spec=grid_spec if grid_spec != "none" else None,
                eval_bits=bits if grid_spec != "none" else None,
            )
            res.wall_time_sec = time.time() - t0
            results.append(res)
            print(f"  {label}: test_loss={res.final_test_loss:.4f}",
                  f"best={res.best_test_loss:.4f}",
                  f"acc_rate={res.final_acceptance_rate:.3f}" if opt.startswith("gdmc") else "")

    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
