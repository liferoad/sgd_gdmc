"""Corrected baseline comparison on MNIST MLP.

Re-runs the headline comparison with the defects from
docs/review_2026-10-04.md fixed:

* Adam at its tuned learning rate (1e-3 / 3e-3) instead of 1e-2;
* Projected-GD with lr scaled to the grid spacing (it was frozen at 4 bits);
* the two missing low-bit baselines: 8-bit Adam and momentum-signSGD;
* GDMC v2 with and without a separate Metropolis acceptance batch.

Writes results/raw/corrected_baselines.csv.

Usage::

    .venv/bin/python experiments/12_corrected_baselines.py --quick
    .venv/bin/python experiments/12_corrected_baselines.py
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--out", type=str,
                   default="results/raw/corrected_baselines.csv")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = 2
        args.seeds = [0]
        args.limit_train = 5000
        if "--out" not in sys.argv:
            args.out = args.out.replace(".csv", "_quick.csv")

    train_ds, test_ds = make_mnist(root="data")
    if args.limit_train and len(train_ds) > args.limit_train:
        train_ds = torch.utils.data.TensorDataset(
            train_ds.tensors[0][:args.limit_train],
            train_ds.tensors[1][:args.limit_train])
    train_ds = torch.utils.data.TensorDataset(
        train_ds.tensors[0].view(-1, 784), train_ds.tensors[1])
    test_ds = torch.utils.data.TensorDataset(
        test_ds.tensors[0].view(-1, 784), test_ds.tensors[1])
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return MLP(in_dim=784, hidden=(256, 256), out_dim=10)

    def add(cfgs, label, opt, bits, lr, **kw):
        cfgs.append((label, opt, bits, lr, kw))

    cfgs = []
    for lr in (1e-3, 3e-3):
        add(cfgs, f"adam-lr{lr:g}", "adam", 32, lr)
        add(cfgs, f"adam8bit-lr{lr:g}", "adam8bit", 32, lr)
    for lr in (3e-3, 1e-2):
        add(cfgs, f"signsgd-lr{lr:g}", "signsgd", 32, lr, momentum=0.9)
    for bits in (4, 8):
        for scale in (0.25, 0.5):
            add(cfgs, f"projected-gd-b{bits}-s{scale:g}", "projected-gd", bits,
                1e-2, project_lr_scale=scale)
    for bits in (4, 8):
        add(cfgs, f"gdmc-v2-b{bits}", "gdmc", bits, 0.0, beta1=0.9, k=1)
    add(cfgs, "gdmc-v2-b8-accept-separate", "gdmc", 8, 0.0, beta1=0.9, k=1,
        accept_on="separate", accept_split=0.5)

    results = []
    t_start = time.time()
    for seed in args.seeds:
        for label, opt, bits, lr, kw in cfgs:
            quantized = bits < 32
            ts = time.time()
            res = run_one(
                task="mnist_mlp_corrected",
                model_name="mlp_256x256",
                model_fn=model_fn,
                optimizer_name=opt,
                grid_spec="uniform" if quantized else "none",
                bits=bits,
                train_loader=tr,
                test_loader=te,
                criterion=criterion,
                epochs=args.epochs,
                batch_size=128,
                lr=lr,
                momentum=kw.get("momentum", 0.0),
                beta=2.0,
                move_frac=0.01,
                seed=seed,
                device="cpu",
                is_classification=True,
                log_every=500,
                beta1=kw.get("beta1", 0.0),
                k=kw.get("k", 1),
                accept_on=kw.get("accept_on", "minibatch"),
                accept_split=kw.get("accept_split", 0.0),
                project_lr_scale=kw.get("project_lr_scale"),
                eval_grid_spec="uniform" if quantized else None,
                eval_bits=bits if quantized else None,
                extra={"label": label},
            )
            res.wall_time_sec = time.time() - ts
            results.append(res)
            print(f"  {label:28s} s{seed}: acc={res.best_test_acc:.4f} "
                  f"acc_rate={res.mean_acceptance_rate:.3f} "
                  f"peak={res.peak_rss_mb:.0f}MB ({res.wall_time_sec:.0f}s)",
                  flush=True)
    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} rows to {args.out} "
          f"(elapsed {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
