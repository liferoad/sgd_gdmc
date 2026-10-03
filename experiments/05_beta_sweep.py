"""Beta sweep on MNIST CNN.

The Metropolis temperature ``beta`` is the main GDMC hyperparameter.
This experiment sweeps it across a few orders of magnitude and reports
how the test accuracy and acceptance rate respond.

Defaults: 5 values of beta, 3 seeds each, 1 quantization level (4-bit).
Run with --quick for a fast version (2 beta values, 1 seed, 1 epoch).
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

from src.data.mnist import make_mnist
from src.models.cnn import SmallCNN
from src.gdmc import GDMCOptimizer, make_grid
from src.runner import run_one, write_results_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="results/raw/beta_sweep.csv")
    parser.add_argument("--limit-train", type=int, default=20000)
    parser.add_argument("--quick", action="store_true",
                        help="use a much smaller config for fast iteration")
    args = parser.parse_args()

    if args.quick:
        args.betas = [0.5, 2.0]
        args.seeds = [0]
        args.epochs = 1
        args.limit_train = 5000
        args.out = "results/raw/beta_sweep_quick.csv"

    print("Loading MNIST...")
    train_ds, test_ds = make_mnist(root="data")
    if args.limit_train and len(train_ds) > args.limit_train:
        train_ds = torch.utils.data.TensorDataset(
            train_ds.tensors[0][:args.limit_train],
            train_ds.tensors[1][:args.limit_train],
        )
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return SmallCNN(in_channels=1, num_classes=10, base_width=16)

    grid = make_grid("uniform", bits=args.bits)

    results = []
    for seed in args.seeds:
        for beta in args.betas:
            label = f"beta={beta:.2f}/s{seed}"
            t0 = time.time()
            res = run_one(
                task="mnist_cnn",
                model_name="smallcnn_b16",
                model_fn=model_fn,
                optimizer_name="gdmc",
                grid_spec="uniform",
                bits=args.bits,
                train_loader=tr,
                test_loader=te,
                criterion=criterion,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=0.0,
                momentum=0.0,
                beta=beta,
                move_frac=0.01,
                seed=seed,
                device="cpu",
                is_classification=True,
                log_every=2000,
                eval_grid_spec="uniform",
                eval_bits=args.bits,
            )
            res.wall_time_sec = time.time() - t0
            results.append(res)
            print(f"  {label}: test_acc={res.final_test_acc:.4f}",
                  f"best={res.best_test_acc:.4f}",
                  f"acc_rate={res.final_acceptance_rate:.3f}")

    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
