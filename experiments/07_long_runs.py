"""Long-run v1 sweep + v2 ablation on MNIST MLP.

The v1 sweep in experiments/01-04 uses a short epoch budget
(MNIST MLP 3 epochs, MNIST/CIFAR CNN 2 epochs) which is fine for
smoke-testing but is not enough to draw firm conclusions on
classification tasks. At 2 epochs on CIFAR-10 even Adam barely
beats the random baseline (0.25 vs 0.10). This script runs the v1
sweep on MNIST MLP at 30 epochs and a v2 long-run (β1=0 vs β1=0.9)
at 4-bit and 8-bit to test whether the v2 win carries over from
toy regression.

Why these specific choices:
  * MNIST MLP at 30 epochs is the cheapest long task (~14 min total
    for the full 126-run sweep). It is enough to learn MNIST well
    (>98% accuracy for Adam) so the relative ordering of optimizers is
    meaningful.
  * The v2 ablation is the open question from the v1 run: the v2
    sweep was only on toy regression. Whether momentum + multi-step
    helps on a real task at the long-run horizon is unknown.

This script writes to `results/raw/long_runs_mlp.csv` and
`results/raw/long_runs_v2_mlp.csv` so they are not mixed with the
short-run data.

Usage::

    .venv/bin/python experiments/07_long_runs.py                 # full Pass 1
    .venv/bin/python experiments/07_long_runs.py --quick          # smoke (1 epoch)
    .venv/bin/python experiments/07_long_runs.py --no-cnn        # skip MNIST CNN long
    .venv/bin/python experiments/07_long_runs.py --epochs-mlp 50  # override MLP epochs
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
from src.models.mlp import MLP
from src.gdmc import GDMCOptimizer, make_grid
from src.baselines import make_adam, make_momentum, make_sgd, make_sgld, make_projected_gd
from src.runner import run_one, write_results_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-mlp", type=int, default=30)
    parser.add_argument("--epochs-cnn", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 1 epoch, 1 seed")
    parser.add_argument("--no-cnn", action="store_true",
                        help="skip the MNIST CNN long-run")
    parser.add_argument("--out-mlp", type=str,
                        default="results/raw/long_runs_mlp.csv")
    parser.add_argument("--out-cnn", type=str,
                        default="results/raw/long_runs_cnn.csv")
    parser.add_argument("--out-v2", type=str,
                        default="results/raw/long_runs_v2_mlp.csv")
    args = parser.parse_args()

    if args.quick:
        args.epochs_mlp = 1
        args.epochs_cnn = 1
        args.seeds = [0]

    print(f"Loading MNIST (full dataset)...")
    train_ds, test_ds = make_mnist(root="data")
    # No train subset for the long run — we want the full signal.
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # MNIST MLP long run — full v1 sweep at the long horizon
    # ------------------------------------------------------------------
    print(f"\n=== MNIST MLP long run: {args.epochs_mlp} epochs, seeds={args.seeds} ===")
    train_ds_mlp = torch.utils.data.TensorDataset(
        train_ds.tensors[0].view(-1, 784),
        train_ds.tensors[1],
    )
    test_ds_mlp = torch.utils.data.TensorDataset(
        test_ds.tensors[0].view(-1, 784),
        test_ds.tensors[1],
    )
    tr_mlp = DataLoader(train_ds_mlp, batch_size=128, shuffle=True)
    te_mlp = DataLoader(test_ds_mlp, batch_size=512, shuffle=False)

    def model_fn_mlp():
        return MLP(in_dim=784, hidden=(256, 256), out_dim=10)

    bits_list = [2, 3, 4, 8, 16, 32]
    # Configs: (label, optimizer_name, grid_spec, bits, lr, beta, move_frac, beta1, k)
    cfgs_mlp = []
    for bits in bits_list:
        for opt in ["sgd", "momentum", "adam", "sgld"]:
            lr = 1e-2 if opt != "sgld" else 5e-4
            beta = 10.0 if opt == "sgld" else 0.0
            cfgs_mlp.append((f"{opt}-b{bits}", opt, "none", bits, lr, beta, 0.0, 0.0, 1))
    for bits in bits_list:
        cfgs_mlp.append((f"projected-gd-b{bits}", "projected-gd", "uniform",
                         bits, 1e-2, 0.0, 0.0, 0.0, 1))
    for bits in bits_list:
        cfgs_mlp.append((f"gdmc-b{bits}", "gdmc", "uniform",
                         bits, 0.0, 2.0, 0.01, 0.0, 1))
    for bits in [3, 4, 8]:
        cfgs_mlp.append((f"gdmc-adaptive-b{bits}", "gdmc-adaptive",
                         "uniform-pt", bits, 0.0, 2.0, 0.01, 0.0, 1))

    results_mlp = []
    t_start = time.time()
    for seed in args.seeds:
        for label, opt, grid, bits, lr, beta, mf, beta1, k in cfgs_mlp:
            t0 = time.time()
            res = run_one(
                task="mnist_mlp_long",
                model_name="mlp_256x256",
                model_fn=model_fn_mlp,
                optimizer_name=opt,
                grid_spec=grid,
                bits=bits,
                train_loader=tr_mlp,
                test_loader=te_mlp,
                criterion=criterion,
                epochs=args.epochs_mlp,
                batch_size=128,
                lr=lr,
                momentum=0.9 if opt == "momentum" else 0.0,
                beta=beta,
                move_frac=mf,
                seed=seed,
                device="cpu",
                is_classification=True,
                log_every=2000,
                beta1=beta1,
                k=k,
                eval_grid_spec=grid if grid != "none" else None,
                eval_bits=bits if grid != "none" else None,
            )
            res.wall_time_sec = time.time() - t0
            results_mlp.append(res)
            print(f"  {label:25s} s{seed}: test_acc={res.best_test_acc:.4f}"
                  f"  ({res.wall_time_sec:.0f}s)", flush=True)
    write_results_csv(results_mlp, args.out_mlp)
    print(f"Wrote {len(results_mlp)} MLP long-run rows to {args.out_mlp}"
          f" (elapsed {time.time() - t_start:.0f}s)")

    # ------------------------------------------------------------------
    # MNIST CNN long run — reduced sweep (skip 32-bit baselines since
    # the short-run results are already 0.97-1.00 and not interesting)
    # ------------------------------------------------------------------
    if not args.no_cnn:
        print(f"\n=== MNIST CNN long run: {args.epochs_cnn} epochs, seeds={args.seeds} ===")
        from src.models.cnn import SmallCNN
        train_ds_cnn = torch.utils.data.TensorDataset(
            train_ds.tensors[0],
            train_ds.tensors[1],
        )
        test_ds_cnn = torch.utils.data.TensorDataset(
            test_ds.tensors[0],
            test_ds.tensors[1],
        )
        tr_cnn = DataLoader(train_ds_cnn, batch_size=128, shuffle=True)
        te_cnn = DataLoader(test_ds_cnn, batch_size=512, shuffle=False)

        def model_fn_cnn():
            return SmallCNN(in_channels=1, num_classes=10, base_width=16)

        # Reduced: skip 32-bit baselines, only do bits {2,3,4,8,16}
        bits_list_cnn = [2, 3, 4, 8, 16]
        cfgs_cnn = []
        for bits in bits_list_cnn:
            for opt in ["sgd", "momentum", "adam", "sgld"]:
                lr = 1e-2 if opt != "sgld" else 5e-4
                beta = 10.0 if opt == "sgld" else 0.0
                cfgs_cnn.append((f"{opt}-b{bits}", opt, "none", bits, lr, beta, 0.0, 0.0, 1))
        for bits in bits_list_cnn:
            cfgs_cnn.append((f"projected-gd-b{bits}", "projected-gd", "uniform",
                             bits, 1e-2, 0.0, 0.0, 0.0, 1))
        for bits in bits_list_cnn:
            cfgs_cnn.append((f"gdmc-b{bits}", "gdmc", "uniform",
                             bits, 0.0, 2.0, 0.01, 0.0, 1))
        for bits in [3, 4, 8]:
            cfgs_cnn.append((f"gdmc-adaptive-b{bits}", "gdmc-adaptive",
                             "uniform-pt", bits, 0.0, 2.0, 0.01, 0.0, 1))

        results_cnn = []
        t_start = time.time()
        for seed in args.seeds:
            for label, opt, grid, bits, lr, beta, mf, beta1, k in cfgs_cnn:
                t0 = time.time()
                res = run_one(
                    task="mnist_cnn_long",
                    model_name="smallcnn_b16",
                    model_fn=model_fn_cnn,
                    optimizer_name=opt,
                    grid_spec=grid,
                    bits=bits,
                    train_loader=tr_cnn,
                    test_loader=te_cnn,
                    criterion=criterion,
                    epochs=args.epochs_cnn,
                    batch_size=128,
                    lr=lr,
                    momentum=0.9 if opt == "momentum" else 0.0,
                    beta=beta,
                    move_frac=mf,
                    seed=seed,
                    device="cpu",
                    is_classification=True,
                    log_every=2000,
                    beta1=beta1,
                    k=k,
                    eval_grid_spec=grid if grid != "none" else None,
                    eval_bits=bits if grid != "none" else None,
                )
                res.wall_time_sec = time.time() - t0
                results_cnn.append(res)
                print(f"  {label:25s} s{seed}: test_acc={res.best_test_acc:.4f}"
                      f"  ({res.wall_time_sec:.0f}s)", flush=True)
        write_results_csv(results_cnn, args.out_cnn)
        print(f"Wrote {len(results_cnn)} CNN long-run rows to {args.out_cnn}"
              f" (elapsed {time.time() - t_start:.0f}s)")

    # ------------------------------------------------------------------
    # v2 long run on MNIST MLP at 4-bit and 8-bit
    # (the v2 win on toy needs to be confirmed on a real task)
    # ------------------------------------------------------------------
    print(f"\n=== v2 long run on MNIST MLP at 4-bit and 8-bit ===")
    results_v2 = []
    t_start = time.time()
    for bits in [4, 8]:
        for seed in args.seeds:
            for beta1, k in [(0.0, 1), (0.9, 1), (0.9, 4)]:
                label = f"gdmc-b{bits}-b1={beta1}-k={k}"
                t0 = time.time()
                res = run_one(
                    task="mnist_mlp_long_v2",
                    model_name="mlp_256x256",
                    model_fn=model_fn_mlp,
                    optimizer_name="gdmc",
                    grid_spec="uniform",
                    bits=bits,
                    train_loader=tr_mlp,
                    test_loader=te_mlp,
                    criterion=criterion,
                    epochs=args.epochs_mlp,
                    batch_size=128,
                    lr=0.0,
                    momentum=0.0,
                    beta=2.0,
                    move_frac=0.01,
                    seed=seed,
                    device="cpu",
                    is_classification=True,
                    log_every=2000,
                    beta1=beta1,
                    k=k,
                    eval_grid_spec="uniform",
                    eval_bits=bits,
                )
                res.wall_time_sec = time.time() - t0
                results_v2.append(res)
                print(f"  {label:25s} s{seed}: test_acc={res.best_test_acc:.4f}"
                      f"  ({res.wall_time_sec:.0f}s)", flush=True)
    write_results_csv(results_v2, args.out_v2)
    print(f"Wrote {len(results_v2)} v2 long-run rows to {args.out_v2}"
          f" (elapsed {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
