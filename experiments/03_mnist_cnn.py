"""MNIST CNN experiment sweep.

Same 6 optimizers and 6 quantization levels as 02_mnist_mlp, but with a
small CNN. We keep the CNN tiny (the SmallCNN from src.models.cnn) so
this still runs in a few minutes per config on CPU.
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
from src.runner import run_one, write_results_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bits-list", type=int, nargs="+",
                        default=[2, 3, 4, 8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=str, default="results/raw/mnist_cnn.csv")
    parser.add_argument("--limit-train", type=int, default=20000)
    args = parser.parse_args()

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

    cfgs = []
    for opt in ["sgd", "momentum", "adam", "sgld"]:
        cfgs.append((opt, "none", 32, dict(
            lr=1e-2 if opt != "sgld" else 5e-4,
            momentum=0.9 if opt == "momentum" else 0.0,
            beta=100.0 if opt == "sgld" else 0.0,
            move_frac=0.0)))

    for bits in args.bits_list:
        cfgs.append(("projected-gd", "uniform", bits, dict(
            lr=1e-2, project_base="adam",
            momentum=0.0, beta=0.0, move_frac=0.0)))

    for bits in args.bits_list:
        cfgs.append(("gdmc", "uniform", bits, dict(
            lr=0.0, beta=2.0, momentum=0.0, move_frac=0.01)))

    for bits in [3, 4, 8]:
        cfgs.append(("gdmc-adaptive", "uniform-pt", bits, dict(
            lr=0.0, beta=2.0, momentum=0.0, move_frac=0.01)))

    results = []
    for seed in args.seeds:
        for opt, grid_spec, bits, hp in cfgs:
            label = f"{opt:>14s}/{grid_spec:>9s}/b{bits:02d}/s{seed}"
            t0 = time.time()
            res = run_one(
                task="mnist_cnn",
                model_name="smallcnn_b16",
                model_fn=model_fn,
                optimizer_name=opt,
                grid_spec=grid_spec,
                bits=bits,
                train_loader=tr,
                test_loader=te,
                criterion=criterion,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=hp.get("lr", 0.0),
                momentum=hp.get("momentum", 0.0),
                beta=hp.get("beta", 0.0),
                move_frac=hp.get("move_frac", 0.0),
                seed=seed,
                device="cpu",
                is_classification=True,
                log_every=2000,
                extra={"project_base": hp.get("project_base", "adam")},
                eval_grid_spec=grid_spec if grid_spec != "none" else None,
                eval_bits=bits if grid_spec != "none" else None,
            )
            res.wall_time_sec = time.time() - t0
            results.append(res)
            print(f"  {label}: test_acc={res.final_test_acc:.4f}",
                  f"best={res.best_test_acc:.4f}",
                  f"acc_rate={res.final_acceptance_rate:.3f}" if opt.startswith("gdmc") else "")

    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
