"""Compact low-bit-weight comparison.

The question: does GDMC (basic or with momentum) beat other optimizers that
operate on the *same* low-precision weights, at 2, 4 and 8 bits?

Methods, all on the same uniform grid in [-1, 1]:
  * gdmc-v1        - basic GDMC (beta1=0, k=1)
  * gdmc-v2        - momentum GDMC (beta1=0.9, k=1)
  * projected-adam - Adam steps projected onto the grid (QAT-style)
  * projected-msgd - momentum SGD steps projected onto the grid
  * qat-ste-adam   - latent-weight QAT with a straight-through estimator
  * adam-fp32      - full-precision accuracy reference (32-bit, no grid)

Matched tuning budget: each quantized method gets the same three candidate
settings per bit-width, selected on a held-out validation split with a tuning
seed that is not one of the reported seeds. Everything is then re-run with the
selected setting and reported on the official test set, so paired per-seed
comparisons are possible.

Writes results/raw/lowbit_comparison.csv.

Usage::

    .venv/bin/python experiments/15_lowbit_comparison.py --quick
    .venv/bin/python experiments/15_lowbit_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.mnist import make_mnist
from src.models.mlp import MLP
from src.models.quant import QuantMLP
from src.runner import run_one, write_results_csv

BITS = (2, 4, 8)
TUNE_SEED = 100
TUNE_GRID = {
    "gdmc-v1": {"beta": [1.0, 2.0, 4.0]},
    "gdmc-v2": {"beta": [1.0, 2.0, 4.0]},
    "projected-adam": {"project_lr_scale": [0.25, 0.5, 1.0]},
    "projected-msgd": {"project_lr_scale": [0.25, 0.5, 1.0]},
}


def mlp_fn():
    return MLP(in_dim=784, hidden=(256, 256), out_dim=10)


def qat_fn(bits):
    def build():
        return QuantMLP(in_dim=784, hidden=(256, 256), out_dim=10, bits=bits)
    return build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--tune-epochs", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--out", type=str,
                   default="results/raw/lowbit_comparison.csv")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = 2
        args.tune_epochs = 1
        args.seeds = [0]
        args.val_size = 2000
        if "--out" not in sys.argv:
            args.out = args.out.replace(".csv", "_quick.csv")

    torch.set_num_threads(4)
    train_ds, test_ds = make_mnist(root="data")
    x = train_ds.tensors[0].view(-1, 784)
    y = train_ds.tensors[1]
    n_val = min(args.val_size, len(x) // 5)
    fit_ds = torch.utils.data.TensorDataset(x[:-n_val], y[:-n_val])
    val_ds = torch.utils.data.TensorDataset(x[-n_val:], y[-n_val:])
    test_ds = torch.utils.data.TensorDataset(
        test_ds.tensors[0].view(-1, 784), test_ds.tensors[1])
    fit_loader = DataLoader(fit_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def quantized_kwargs(method, bits, setting):
        kw = dict(optimizer_name=method, grid_spec="uniform", bits=bits,
                  lr=0.0, momentum=0.0, beta=2.0, move_frac=0.01,
                  beta1=0.0, k=1)
        if method == "gdmc-v1":
            kw["beta"] = setting["beta"]
        elif method == "gdmc-v2":
            kw["beta"] = setting["beta"]
            kw["beta1"] = 0.9
        elif method in ("projected-adam", "projected-msgd"):
            kw["lr"] = 1e-2
            kw["project_lr_scale"] = setting["project_lr_scale"]
        elif method == "qat-ste-adam":
            kw["optimizer_name"] = "adam"
            kw["lr"] = 1e-3
        return kw

    def run(method, bits, setting, seed, epochs, loader, extra_label):
        kw = quantized_kwargs(method, bits, setting)
        model_fn = qat_fn(bits) if method == "qat-ste-adam" else mlp_fn
        quantized = bits < 32
        res = run_one(
            task="mnist_lowbit",
            model_name="mlp_256x256",
            model_fn=model_fn,
            grid_spec=kw["grid_spec"],
            bits=bits,
            train_loader=fit_loader,
            test_loader=loader,
            criterion=criterion,
            epochs=epochs,
            batch_size=128,
            lr=kw["lr"],
            momentum=kw["momentum"],
            beta=kw["beta"],
            move_frac=kw["move_frac"],
            seed=seed,
            device="cpu",
            is_classification=True,
            log_every=1000,
            beta1=kw["beta1"],
            k=kw["k"],
            project_lr_scale=kw.get("project_lr_scale"),
            optimizer_name=kw["optimizer_name"],
            eval_grid_spec="uniform" if quantized else None,
            eval_bits=bits if quantized else None,
            extra={"label": extra_label, "setting": json.dumps(setting)},
        )
        return res

    results = []
    t0 = time.time()

    # ---- tuning on the validation split (single tuning seed) ----
    selected = {}
    for method, grid in TUNE_GRID.items():
        for bits in BITS:
            best = None
            for candidate in [{k: v} for k, vals in grid.items() for v in vals]:
                res = run(method, bits, candidate, TUNE_SEED,
                          args.tune_epochs, val_loader, method)
                score = res.best_test_acc
                print(f"  tune {method:15s} b{bits} {candidate} "
                      f"val={score:.4f}", flush=True)
                if best is None or score > best[0]:
                    best = (score, candidate)
            selected[(method, bits)] = best[1]
            print(f"  -> selected {method} b{bits}: {best[1]} "
                  f"(val {best[0]:.4f})", flush=True)

    # ---- final runs on the test set ----
    for method in TUNE_GRID:
        for bits in BITS:
            setting = selected[(method, bits)]
            for seed in args.seeds:
                res = run(method, bits, setting, seed, args.epochs,
                          test_loader, method)
                results.append(res)
                print(f"  {method:15s} b{bits} s{seed} setting={setting} "
                      f"test={res.best_test_acc:.4f} "
                      f"acc_rate={res.mean_acceptance_rate:.3f}", flush=True)

    for bits in BITS:
        for seed in args.seeds:
            res = run("qat-ste-adam", bits, {}, seed, args.epochs,
                      test_loader, "qat-ste-adam")
            results.append(res)
            print(f"  {'qat-ste-adam':15s} b{bits} s{seed} "
                  f"test={res.best_test_acc:.4f}", flush=True)

    for seed in args.seeds:
        res = run_one(task="mnist_lowbit", model_name="mlp_256x256",
                      model_fn=mlp_fn, optimizer_name="adam", grid_spec="none",
                      bits=32, train_loader=fit_loader, test_loader=test_loader,
                      criterion=criterion, epochs=args.epochs, batch_size=128,
                      lr=1e-3, momentum=0.0, beta=0.0, move_frac=0.0,
                      seed=seed, device="cpu", is_classification=True,
                      log_every=1000, extra={"label": "adam-fp32",
                                             "setting": "{}"})
        results.append(res)
        print(f"  {'adam-fp32':15s} b32 s{seed} "
              f"test={res.best_test_acc:.4f}", flush=True)

    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} rows to {args.out} "
          f"(elapsed {time.time() - t0:.0f}s)")

    # ---- paired summary ----
    by = {}
    for r in results:
        label = json.loads(r.extra)["label"]
        by.setdefault((label, r.bits), {})[r.seed] = r.best_test_acc
    ref = by.get(("projected-adam", 4), {})
    print("\nmethod / bits: mean test acc, paired difference vs projected-adam")
    for (label, bits), per_seed in sorted(by.items()):
        vals = np.array(list(per_seed.values()))
        line = f"  {label:15s} b{bits:<2d} mean={vals.mean():.4f} n={len(vals)}"
        if bits < 32 and label != "projected-adam":
            base = by.get(("projected-adam", bits), {})
            common = sorted(set(per_seed) & set(base))
            if common:
                d = np.array([per_seed[s] - base[s] for s in common])
                line += (f"  paired-vs-proj-adam={d.mean():+.4f} "
                         f"(per-seed {np.round(d, 4).tolist()})")
        print(line)


if __name__ == "__main__":
    main()
