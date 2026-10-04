"""Gradient-noise study (the feedback's central experiment, now first-class).

Injects norm-scaled Gaussian noise into the *proposal* gradient only, and
measures validation quality vs noise strength for:

* FP32 Adam (noise applied to the optimizer gradient in the training loop);
* GDMC v2 at 8 and 4 bits (noise applied inside the proposal construction);
* GDMC v2 at 8 bits with beta=0, i.e. identical proposals but always accept
  (the acceptance ablation);
* GDMC v2 at 8 bits with a separate Metropolis acceptance batch.

Noise is eps = rho * RMS(g) * N(0, 1) per tensor; the Metropolis loss
evaluation is unaffected. Because each optimizer owns its torch.Generator,
all runs at a given seed see the same data order (common random numbers).

Writes results/raw/noise_study.csv.

Usage::

    .venv/bin/python experiments/14_noise_study.py --quick
    .venv/bin/python experiments/14_noise_study.py
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
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rhos", type=float, nargs="+",
                   default=[0.0, 0.1, 0.3, 1.0, 3.0])
    p.add_argument("--out", type=str, default="results/raw/noise_study.csv")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.epochs = 1
        args.seeds = [0]
        args.rhos = [0.0, 1.0]
        if "--out" not in sys.argv:
            args.out = args.out.replace(".csv", "_quick.csv")

    torch.set_num_threads(4)
    train_ds, test_ds = make_mnist(root="data")
    train_ds = torch.utils.data.TensorDataset(
        train_ds.tensors[0].view(-1, 784), train_ds.tensors[1])
    test_ds = torch.utils.data.TensorDataset(
        test_ds.tensors[0].view(-1, 784), test_ds.tensors[1])
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def model_fn():
        return MLP(in_dim=784, hidden=(256, 256), out_dim=10)

    # (label, optimizer, bits, kwargs)
    methods = [
        ("adam", "adam", 32, dict(lr=1e-3)),
        ("gdmc-b8-accept", "gdmc", 8, dict(beta1=0.9, k=1)),
        ("gdmc-b8-always-accept", "gdmc", 8, dict(beta1=0.9, k=1, beta=0.0)),
        ("gdmc-b4-accept", "gdmc", 4, dict(beta1=0.9, k=1)),
    ]
    extra_cells = [("gdmc-b8-accept-separate", "gdmc", 8,
                    dict(beta1=0.9, k=1, accept_on="separate", accept_split=0.5))]

    results = []
    t_start = time.time()
    cells = [(lab, opt, bits, kw, rho)
             for rho in args.rhos for (lab, opt, bits, kw) in methods]
    # The separate-acceptance cell only matters where noise is strong.
    for rho in [r for r in args.rhos if r >= 1.0]:
        for (lab, opt, bits, kw) in extra_cells:
            cells.append((lab, opt, bits, kw, rho))

    for seed in args.seeds:
        for label, opt, bits, kw, rho in cells:
            quantized = bits < 32
            ts = time.time()
            res = run_one(
                task="mnist_mlp_noise",
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
                lr=kw.get("lr", 0.0),
                momentum=0.0,
                beta=kw.get("beta", 2.0),
                move_frac=0.01,
                seed=seed,
                device="cpu",
                is_classification=True,
                log_every=1000,
                beta1=kw.get("beta1", 0.0),
                k=kw.get("k", 1),
                accept_on=kw.get("accept_on", "minibatch"),
                accept_split=kw.get("accept_split", 0.0),
                grad_noise_rho=rho,
                eval_grid_spec="uniform" if quantized else None,
                eval_bits=bits if quantized else None,
                extra={"rho": rho, "label": label},
            )
            res.wall_time_sec = time.time() - ts
            results.append(res)
            print(f"  rho={rho:<4g} {label:26s} s{seed}: "
                  f"acc={res.best_test_acc:.4f} "
                  f"acc_rate={res.mean_acceptance_rate:.3f} "
                  f"({res.wall_time_sec:.0f}s)", flush=True)
    write_results_csv(results, args.out)
    print(f"Wrote {len(results)} rows to {args.out} "
          f"(elapsed {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
