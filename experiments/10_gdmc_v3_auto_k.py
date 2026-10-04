"""GDMC v3: magnitude-scaled (auto) grid steps.

Motivation
----------
The v1/v2 sweeps found that GDMC stalls at 16-32 bits: a single grid
step is far below float32 precision (at 32 bits delta = 4.7e-10, which
is smaller than the float32 epsilon 6e-8 near 0.5), so with k=1 the
proposal is a no-op and the model never moves.

v3 fixes this by choosing the number of grid steps per weight so that
the *continuous* displacement matches a target update size:

    k_i = clip( round( step_scale * |g_i| / g_ref / delta ), 1, k_max )

where g_ref is a per-tensor EMA of mean|g| and delta is the grid
spacing.  At coarse bit-widths the target displacement is smaller than
one grid step so k_i clips to 1 (preserving v1/v2 behaviour); at fine
bit-widths k_i grows to hundreds or millions so the model can actually
move.  The Metropolis test is unchanged, so large proposals are still
rejected when they hurt.

This script compares, on MNIST MLP at 30 epochs:
  * adam (32-bit continuous reference)
  * projected-gd (QAT-style Adam + snap)
  * gdmc v1  (no momentum, k=1)
  * gdmc v2  (beta1=0.9, k=1)
  * gdmc v3  (beta1=0.9, auto k) with a step_scale sweep

It also writes per-step training curves (train loss, test loss, test
accuracy, acceptance rate) to --curves-dir for later plotting.

Usage::

    .venv/bin/python experiments/10_gdmc_v3_auto_k.py --quick
    .venv/bin/python experiments/10_gdmc_v3_auto_k.py
"""

from __future__ import annotations

import argparse
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
from src.runner import run_one, write_results_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit-train", type=int, default=0,
                   help="0 = use the full 60K training set")
    p.add_argument("--out", type=str,
                   default="results/raw/gdmc_v3_auto_k.csv")
    p.add_argument("--curves-dir", type=str,
                   default="results/curves/mnist_mlp_v3")
    p.add_argument("--no-curves", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs = 2
        args.seeds = [0]
        args.limit_train = 5000
        # Never clobber the full-run CSV from a smoke test.
        if "--out" not in sys.argv:
            args.out = args.out.replace(".csv", "_quick.csv")

    train_ds, test_ds = make_mnist(root="data")
    if args.limit_train and len(train_ds) > args.limit_train:
        train_ds = torch.utils.data.TensorDataset(
            train_ds.tensors[0][:args.limit_train],
            train_ds.tensors[1][:args.limit_train],
        )
    train_ds = torch.utils.data.TensorDataset(
        train_ds.tensors[0].view(-1, 784), train_ds.tensors[1])
    test_ds = torch.utils.data.TensorDataset(
        test_ds.tensors[0].view(-1, 784), test_ds.tensors[1])
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return MLP(in_dim=784, hidden=(256, 256), out_dim=10)

    # (label, optimizer_name, grid_spec, bits, lr, beta1, k, k_mode, step_scale)
    cfgs = []
    cfgs.append(("adam-32", "adam", "none", 32, 1e-3, 0.0, 1, "fixed", 0.0))
    for bits in (4, 8, 16, 32):
        cfgs.append((f"projected-gd-b{bits}", "projected-gd", "uniform", bits,
                     1e-2, 0.0, 1, "fixed", 0.0))
        cfgs.append((f"gdmc-v1-b{bits}", "gdmc", "uniform", bits,
                     0.0, 0.0, 1, "fixed", 0.0))
        cfgs.append((f"gdmc-v2-b{bits}", "gdmc", "uniform", bits,
                     0.0, 0.9, 1, "fixed", 0.0))
        cfgs.append((f"gdmc-v3-ss1e-3-b{bits}", "gdmc-auto", "uniform", bits,
                     0.0, 0.9, 1, "auto", 1e-3))
    # step_scale sweep at the two fine grids
    for bits in (16, 32):
        for ss in (1e-4, 1e-2):
            cfgs.append((f"gdmc-v3-ss{ss:.0e}-b{bits}", "gdmc-auto", "uniform",
                         bits, 0.0, 0.9, 1, "auto", ss))

    curves_dir = None if args.no_curves else args.curves_dir
    results = []
    t_start = time.time()
    for seed in args.seeds:
        for (label, opt, grid, bits, lr, beta1, k, kmode, ss) in cfgs:
            ts = time.time()
            res = run_one(
                task="mnist_mlp_v3",
                model_name="mlp_256x256",
                model_fn=model_fn,
                optimizer_name=opt,
                grid_spec=grid,
                bits=bits,
                train_loader=tr,
                test_loader=te,
                criterion=criterion,
                epochs=args.epochs,
                batch_size=128,
                lr=lr,
                momentum=0.0,
                beta=2.0,
                move_frac=0.01,
                seed=seed,
                device="cpu",
                is_classification=True,
                log_every=500,
                beta1=beta1,
                k=k,
                k_mode=kmode,
                step_scale=ss,
                project_lr_scale=0.5,
                eval_grid_spec=grid if grid != "none" else None,
                eval_bits=bits if grid != "none" else None,
                curves_dir=curves_dir,
                extra={"v3_label": label},
            )
            res.wall_time_sec = time.time() - ts
            results.append(res)
            print(f"  {label:26s} s{seed}: acc={res.best_test_acc:.4f}"
                  f"  ({res.wall_time_sec:.0f}s)", flush=True)
    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} rows to {args.out} "
          f"(elapsed {time.time() - t_start:.0f}s)")
    if curves_dir:
        print(f"Curves in {curves_dir}")


if __name__ == "__main__":
    main()
