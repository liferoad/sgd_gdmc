"""Review probe 3: is Projected-GD's low-bit failure a step-size artifact?

ProjectedGD takes an Adam step of size ~lr and then snaps to the grid.
With grid spacing delta, if lr < delta/2 every step is snapped back and
the weights never move.  At 4 bits delta = 2/15 = 0.133, but the project
runs Projected-GD at lr = 1e-2 (and 1e-2 everywhere).  This sweep varies
lr across delta for the same task/model/horizon.

MNIST MLP, 10K train subset, 5 epochs, batch 128, 3 seeds.

Usage: .venv/bin/python analysis/review/03_projected_lr_check.py
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.mnist import make_mnist          # noqa: E402
from src.models.mlp import MLP                 # noqa: E402
from src.runner import run_one, write_results_csv  # noqa: E402

torch.set_num_threads(4)

tr, te = make_mnist(root="data")
tr = torch.utils.data.TensorDataset(tr.tensors[0][:10000].view(-1, 784), tr.tensors[1][:10000])
te = torch.utils.data.TensorDataset(te.tensors[0].view(-1, 784), te.tensors[1])
tr_loader = DataLoader(tr, batch_size=128, shuffle=True)
te_loader = DataLoader(te, batch_size=512, shuffle=False)
crit = nn.CrossEntropyLoss()

def model_fn():
    return MLP(in_dim=784, hidden=(256, 256), out_dim=10)

results = []
for bits, lrs in [(4, [1e-2, 5e-2, 1e-1, 2e-1, 5e-1]),
                  (2, [1e-2, 3e-1, 6e-1, 1.0])]:
    delta = 2.0 / (2**bits - 1)
    for lr in lrs:
        for seed in [0, 1, 2]:
            r = run_one(task="projgd_lr_check", model_name="mlp_256x256",
                        model_fn=model_fn, optimizer_name="projected-gd",
                        grid_spec="uniform", bits=bits, train_loader=tr_loader,
                        test_loader=te_loader, criterion=crit, epochs=5,
                        batch_size=128, lr=lr, momentum=0.0, beta=0.0,
                        move_frac=0.0, seed=seed, device="cpu",
                        is_classification=True, log_every=1000,
                        eval_grid_spec="uniform", eval_bits=bits,
                        extra={"project_base": "adam", "delta": delta})
            results.append(r)
            print(f"projgd b{bits} delta={delta:.3f} lr={lr:<5g} seed{seed}: "
                  f"acc={r.best_test_acc:.4f}", flush=True)
    # GDMC v2 reference at the same budget
    for seed in [0, 1, 2]:
        r = run_one(task="projgd_lr_check", model_name="mlp_256x256",
                    model_fn=model_fn, optimizer_name="gdmc", grid_spec="uniform",
                    bits=bits, train_loader=tr_loader, test_loader=te_loader,
                    criterion=crit, epochs=5, batch_size=128, lr=0.0,
                    momentum=0.0, beta=2.0, move_frac=0.01, seed=seed,
                    device="cpu", is_classification=True, log_every=1000,
                    beta1=0.9, k=1, eval_grid_spec="uniform", eval_bits=bits)
        results.append(r)
        print(f"gdmc  b{bits} (lr n/a)     seed{seed}: acc={r.best_test_acc:.4f}", flush=True)

write_results_csv(results, "results/review/projected_lr_check.csv")
print("wrote results/review/projected_lr_check.csv")
