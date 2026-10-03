"""GDMC v2 sweep: characterize the momentum + multi-step changes.

This experiment compares the original GDMC (``beta1=0, k=1``) against
v2 with various ``beta1`` and ``k`` settings, across the same
quantization levels as the headline sweep. The goal is to find the
configurations where v2 helps and where it hurts.

The headline finding (see the smoke test) is that v2 helps
*marginally* at fine grids (8+ bits) where the v1 one-step move is
too small, and *hurts* at coarse grids (2-4 bits) where v1's
conservative step is exactly right. The experiment confirms this
trend on a fuller test set.

Run with --quick for a fast version (2 seeds, fewer bits).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.toy_regression import ToyRegression
from src.models.mlp import MLP
from src.runner import run_one, write_results_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bits-list", type=int, nargs="+",
                        default=[2, 3, 4, 8, 16, 32])
    parser.add_argument("--out", type=str, default="results/raw/gdmc_v2.csv")
    parser.add_argument("--quick", action="store_true",
                        help="small fast config for CI / smoke test")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [0]
        args.epochs = 10
        args.bits_list = [4, 8, 16]

    print("Building toy-regression data...")
    train_ds = ToyRegression(N=400, noise=0.05, seed=0)
    test_ds = ToyRegression(N=200, noise=0.05, seed=1)
    tr = DataLoader(train_ds, batch_size=32, shuffle=True)
    te = DataLoader(test_ds, batch_size=128, shuffle=False)
    criterion = nn.MSELoss()

    def model_fn():
        return MLP(in_dim=1, hidden=(64, 64), out_dim=1)

    # Each tuple: (label, beta1, k).
    # The v1 baseline is (0.0, 1).
    # The v2 candidates sweep (beta1, k) but always at the same
    # 5-bit-flavoured budget: small (1, 1), with momentum (0.9, 1),
    # with multi-step (0, 2), (0, 4), and the full v2 (0.9, 2).
    # At coarse bits (2-4) large k is dangerous; at fine bits (16-32)
    # it is necessary.
    cfgs = [
        ("v1-b1=0.0-k=1",   0.0, 1),
        ("v2-b1=0.9-k=1",   0.9, 1),
        ("v2-b1=0.0-k=2",   0.0, 2),
        ("v2-b1=0.0-k=4",   0.0, 4),
        ("v2-b1=0.9-k=2",   0.9, 2),
        ("v2-b1=0.9-k=4",   0.9, 4),
    ]

    results = []
    for seed in args.seeds:
        for bits in args.bits_list:
            for label, beta1, k in cfgs:
                tag = f"{label}/b{bits:02d}/s{seed}"
                t0 = time.time()
                res = run_one(
                    task="toy_regression_v2",
                    model_name="mlp_64x64",
                    model_fn=model_fn,
                    optimizer_name="gdmc",
                    grid_spec="uniform",
                    bits=bits,
                    train_loader=tr,
                    test_loader=te,
                    criterion=criterion,
                    epochs=args.epochs,
                    batch_size=32,
                    lr=0.0,
                    momentum=0.0,
                    beta=2.0,
                    move_frac=0.05,
                    seed=seed,
                    device="cpu",
                    is_classification=False,
                    log_every=500,
                    beta1=beta1,
                    k=k,
                    eval_grid_spec="uniform" if bits != 32 else None,
                    eval_bits=bits if bits != 32 else None,
                )
                res.wall_time_sec = time.time() - t0
                # Stash the (beta1, k) tuple in the extra column for
                # round-tripping through the CSV.
                res.extra = f'{{"v2_label": "{label}"}}'
                results.append(res)
                print(f"  {tag}: test_loss={res.final_test_loss:.4f}",
                      f"best={res.best_test_loss:.4f}",
                      f"acc_rate={res.final_acceptance_rate:.3f}")

    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
