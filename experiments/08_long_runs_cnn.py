"""Focused long-run comparison on MNIST CNN.

The short MNIST CNN sweep in experiments/03 used 2 epochs and 10K
training samples, which produced a very noisy 4-bit result
(0.5180 +/- 0.2193, with one seed at 0.18). This script re-runs the
*key* comparison at a longer horizon to see whether the qualitative
conclusions survive:

  * gdmc (v1, one-step, no momentum) at 4 and 8 bits
  * gdmc v2 (momentum, beta1=0.9, k=1) at 4 and 8 bits
  * projected-gd (QAT-style Adam + snap) at 4 and 8 bits
  * adam at 32-bit (continuous reference, no quantization)

8 epochs on the 10K training subset, 3 seeds. The full 126-config
sweep is prohibitively expensive on CPU (~15 h); this focused slice
answers the questions that matter in ~1 h.

Writes results/raw/long_runs_cnn_focused.csv.
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
from src.models.cnn import SmallCNN
from src.runner import run_one, write_results_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit-train", type=int, default=10000)
    p.add_argument("--out", type=str,
                   default="results/raw/long_runs_cnn_focused.csv")
    args = p.parse_args()

    train_ds, test_ds = make_mnist(root="data")
    if args.limit_train and len(train_ds) > args.limit_train:
        train_ds = torch.utils.data.TensorDataset(
            train_ds.tensors[0][:args.limit_train],
            train_ds.tensors[1][:args.limit_train],
        )
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return SmallCNN(in_channels=1, num_classes=10, base_width=16)

    # (label, optimizer, grid_spec, bits, lr, beta1, k)
    cfgs = [("adam-32", "adam", "none", 32, 1e-3, 0.0, 1)]
    for bits in (4, 8):
        cfgs += [
            (f"projected-gd-b{bits}", "projected-gd", "uniform", bits, 1e-2, 0.0, 1),
            (f"gdmc-v1-b{bits}", "gdmc", "uniform", bits, 0.0, 0.0, 1),
            (f"gdmc-v2-mom-b{bits}", "gdmc", "uniform", bits, 0.0, 0.9, 1),
        ]

    results = []
    t0 = time.time()
    for seed in args.seeds:
        for label, opt, grid, bits, lr, beta1, k in cfgs:
            ts = time.time()
            res = run_one(
                task="mnist_cnn_long_focused",
                model_name="smallcnn_b16",
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
                log_every=1000,
                beta1=beta1,
                k=k,
                project_lr_scale=0.5,
                eval_grid_spec=grid if grid != "none" else None,
                eval_bits=bits if grid != "none" else None,
            )
            res.wall_time_sec = time.time() - ts
            results.append(res)
            print(f"  {label:20s} s{seed}: test_acc={res.best_test_acc:.4f}"
                  f"  ({res.wall_time_sec:.0f}s)", flush=True)
    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} rows to {args.out} (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
