"""Focused long-run comparison on CIFAR-10 CNN.

The short CIFAR-10 sweep in experiments/04 used 2 epochs and 10K
training samples, which produced results barely above random (Adam at
0.2513, GDMC 4-bit at 0.2671) — so no firm conclusion was possible.
This script re-runs the key comparison at 20 epochs on the 10K
training subset.

Why these specific choices:
  * The 2-epoch budget for CIFAR was too short; even Adam only got
    to 25% vs 10% random. The probe shows Adam reaches 0.40 in
    3 epochs, so 20 epochs is enough to see real differences between
    optimizers.
  * Following the MLP and CNN long-run pattern, the focus is on
    the headline result: does GDMC v2 with momentum match or beat
    continuous Adam at 8 bits on CIFAR too?

Wall time estimate (from probe): Adam 3 epochs = 32s on 10K subset,
so 20 epochs ~= 213s. GDMC is 2x cost ~= 387s. With 12 configs *
3 seeds, total ~= 3 hours on this CPU.

Writes results/raw/long_runs_cifar_focused.csv.
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

from src.data.cifar10 import make_cifar10
from src.models.cnn import SmallCNN
from src.runner import run_one, write_results_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit-train", type=int, default=10000)
    p.add_argument("--out", type=str,
                   default="results/raw/long_runs_cifar_focused.csv")
    args = p.parse_args()

    train_ds, test_ds = make_cifar10(root="data")
    if args.limit_train and len(train_ds) > args.limit_train:
        train_ds = torch.utils.data.TensorDataset(
            train_ds.tensors[0][:args.limit_train],
            train_ds.tensors[1][:args.limit_train],
        )
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return SmallCNN(in_channels=3, num_classes=10, base_width=16)

    # (label, optimizer, grid_spec, bits, lr, beta1, k)
    cfgs = [
        # Continuous baselines (32-bit, no quantization)
        ("adam-32",      "adam",     "none", 32, 1e-3, 0.0, 1),
        ("momentum-32",  "momentum", "none", 32, 1e-2, 0.0, 1),
        # Quantized at 4 and 8 bits (the interesting cells)
        ("projected-gd-b4", "projected-gd", "uniform", 4, 1e-2, 0.0, 1),
        ("projected-gd-b8", "projected-gd", "uniform", 8, 1e-2, 0.0, 1),
        ("gdmc-v1-b4",      "gdmc",       "uniform", 4, 0.0, 0.0, 1),
        ("gdmc-v1-b8",      "gdmc",       "uniform", 8, 0.0, 0.0, 1),
        ("gdmc-v2-mom-b4",  "gdmc",       "uniform", 4, 0.0, 0.9, 1),
        ("gdmc-v2-mom-b8",  "gdmc",       "uniform", 8, 0.0, 0.9, 1),
    ]

    results = []
    t_start = time.time()
    for seed in args.seeds:
        for label, opt, grid, bits, lr, beta1, k in cfgs:
            ts = time.time()
            res = run_one(
                task="cifar10_cnn_long",
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
                log_every=2000,
                beta1=beta1,
                k=k,
                project_lr_scale=0.5,
                eval_grid_spec=grid if grid != "none" else None,
                eval_bits=bits if grid != "none" else None,
            )
            res.wall_time_sec = time.time() - ts
            results.append(res)
            print(f"  {label:18s} s{seed}: test_acc={res.best_test_acc:.4f}"
                  f"  ({res.wall_time_sec:.0f}s)", flush=True)
    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} rows to {args.out} (elapsed {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
