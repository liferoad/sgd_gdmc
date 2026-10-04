"""GDMC on Fashion-MNIST: a second, harder use case.

Fashion-MNIST is a drop-in replacement for MNIST (same 28x28 grayscale,
10 classes) but substantially harder: a small MLP reaches ~88% rather
than ~98%. Running the same optimizer comparison here tests whether the
conclusions from MNIST transfer to a different data distribution.

Compares, at 20 epochs on the full 60K training set:
  * adam (32-bit continuous reference)
  * projected-gd (QAT-style Adam + snap)
  * gdmc v1  (no momentum, k=1)
  * gdmc v2  (beta1=0.9, k=1)
  * gdmc v3  (beta1=0.9, auto magnitude-scaled k, step_scale=1e-3)

at bit-widths {2, 4, 8, 16, 32}, 3 seeds, with per-step curves written
for plotting.

Usage::

    .venv/bin/python experiments/11_fashion_mnist.py --quick
    .venv/bin/python experiments/11_fashion_mnist.py
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

from src.data.fashion_mnist import make_fashion_mnist
from src.models.mlp import MLP
from src.runner import run_one, write_results_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--bits", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--step-scale", type=float, default=1e-3)
    p.add_argument("--out", type=str,
                   default="results/raw/fashion_mnist_mlp.csv")
    p.add_argument("--curves-dir", type=str,
                   default="results/curves/fashion_mnist_mlp")
    p.add_argument("--no-curves", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.epochs = 2
        args.seeds = [0]
        args.bits = [4, 16]

    train_ds, test_ds = make_fashion_mnist(root="data")
    train_ds = torch.utils.data.TensorDataset(
        train_ds.tensors[0].view(-1, 784), train_ds.tensors[1])
    test_ds = torch.utils.data.TensorDataset(
        test_ds.tensors[0].view(-1, 784), test_ds.tensors[1])
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return MLP(in_dim=784, hidden=(256, 256), out_dim=10)

    # (label, optimizer_name, bits, lr, beta1, k_mode, step_scale)
    cfgs = [("adam-32", "adam", 32, 1e-3, 0.0, "fixed", 0.0)]
    for bits in args.bits:
        cfgs.append((f"projected-gd-b{bits}", "projected-gd", bits,
                     1e-2, 0.0, "fixed", 0.0))
        cfgs.append((f"gdmc-v1-b{bits}", "gdmc", bits, 0.0, 0.0, "fixed", 0.0))
        cfgs.append((f"gdmc-v2-b{bits}", "gdmc", bits, 0.0, 0.9, "fixed", 0.0))
        cfgs.append((f"gdmc-v3-b{bits}", "gdmc-auto", bits, 0.0, 0.9,
                     "auto", args.step_scale))

    curves_dir = None if args.no_curves else args.curves_dir
    results = []
    t_start = time.time()
    for seed in args.seeds:
        for (label, opt, bits, lr, beta1, kmode, ss) in cfgs:
            ts = time.time()
            res = run_one(
                task="fashion_mnist_mlp",
                model_name="mlp_256x256",
                model_fn=model_fn,
                optimizer_name=opt,
                grid_spec="none" if opt == "adam" else "uniform",
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
                k=1,
                k_mode=kmode,
                step_scale=ss,
                eval_grid_spec=None if opt == "adam" else "uniform",
                eval_bits=None if opt == "adam" else bits,
                curves_dir=curves_dir,
                extra={"v3_label": label},
            )
            res.wall_time_sec = time.time() - ts
            results.append(res)
            print(f"  {label:22s} s{seed}: acc={res.best_test_acc:.4f}"
                  f"  ({res.wall_time_sec:.0f}s)", flush=True)
    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} rows to {args.out} "
          f"(elapsed {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
