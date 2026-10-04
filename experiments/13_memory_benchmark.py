"""Measured peak-memory benchmark: FP32 Adam vs 8-bit Adam vs GDMC.

Each method runs in its own subprocess so that ru_maxrss reflects that
method's peak. The model is a 21M-parameter MLP with small activations, so
the measured difference is dominated by optimizer state and proposal
buffers, not by activations.

Run without --method to benchmark every method and write
results/raw/memory_benchmark.csv; with --method NAME it prints one JSON row.

Usage::

    .venv/bin/python experiments/13_memory_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METHODS = ["adam", "adam8bit", "gdmc-v1", "gdmc-v2", "gdmc-v2-mf1"]


def optimizer_state_bytes(opt) -> int:
    """Persistent optimizer state (moments / buffers) in bytes.

    Parameters and gradients are NOT optimizer state and are reported
    separately; for GDMC without momentum the optimizer state is empty.
    """
    import torch
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def build(n_in=1024, d=4096, batch=128):
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
    x = torch.randn(batch, n_in)
    y = torch.randn(batch, d)
    n = sum(p.numel() for p in model.parameters())
    return model, x, y, n


def run_one(method, steps=3):
    import torch
    import resource
    import sys as _sys
    from src.gdmc import GDMCOptimizer, UniformGrid
    from src.baselines.adam8bit import Adam8Bit

    def peak_mb():
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if _sys.platform == "darwin" else v / 1024

    model, x, y, n = build()
    crit = torch.nn.MSELoss()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    if method == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif method == "adam8bit":
        opt = Adam8Bit(model.parameters(), lr=1e-3)
    else:
        move_frac = 1.0 if method.endswith("mf1") else 0.01
        beta1 = 0.0 if method == "gdmc-v1" else 0.9
        opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4),
                            beta=2.0, move_frac=move_frac, beta1=beta1, k=1,
                            rng=torch.Generator().manual_seed(0))
    # Baseline is taken after the model/data are allocated but before the
    # first optimizer step, so the growth below is optimizer state plus the
    # per-step proposal/rollback buffers -- not the model itself.
    base = peak_mb()
    for _ in range(steps):
        if getattr(opt, "requires_closure", False):
            opt.step(closure)
        else:
            closure()
            opt.step()
    peak = peak_mb()
    state_bytes = optimizer_state_bytes(opt)
    grad_bytes = sum(p.grad.numel() * p.grad.element_size()
                     for p in model.parameters() if p.grad is not None)
    row = dict(method=method, params=n, steps=steps, base_mb=base, peak_mb=peak,
               growth_mb=peak - base,
               growth_bytes_per_param=(peak - base) * 1024 * 1024 / n,
               state_bytes_per_param=state_bytes / n,
               grad_bytes_per_param=grad_bytes / n,
               weight_bytes_per_param=4.0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None, choices=METHODS)
    ap.add_argument("--out", default="results/raw/memory_benchmark.csv")
    args = ap.parse_args()
    if args.method:
        print(json.dumps(run_one(args.method)))
        return
    rows = []
    for m in METHODS:
        out = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--method", m],
            capture_output=True, text=True, cwd=str(ROOT))
        if out.returncode != 0:
            print(f"{m}: FAILED\n{out.stderr[-500:]}")
            continue
        row = json.loads(out.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"{m:12s} peak={row['peak_mb']:7.1f}MB growth={row['growth_mb']:7.1f}MB "
              f"({row['growth_bytes_per_param']:6.1f} B/param) "
              f"state={row['state_bytes_per_param']:.2f} grad="
              f"{row['grad_bytes_per_param']:.2f} B/param", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
