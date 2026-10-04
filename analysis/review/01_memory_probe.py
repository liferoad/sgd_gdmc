"""Review probe 1: parameter-related memory of GDMC vs Adam.

Runs one method per subprocess so ru_maxrss reflects that method's peak.
Reports measured peak RSS growth during the optimizer step, plus the
theoretical persistent per-parameter storage.

Usage: .venv/bin/python analysis/review/01_memory_probe.py
"""
from __future__ import annotations
import argparse, json, resource, subprocess, sys, os
from pathlib import Path
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]


def build(n_params_target=20_000_000, batch=128, in_dim=1024):
    # Linear stack with ~21M params; activations are tiny next to weights.
    d = 4096
    layers = [nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU()]
    model = nn.Sequential(*layers)
    n = sum(p.numel() for p in model.parameters())
    x = torch.randn(batch, in_dim)
    y = torch.randn(batch, d)
    return model, x, y, n


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run(method, move_frac=0.01, steps=3):
    sys.path.insert(0, str(ROOT))
    from src.gdmc import GDMCOptimizer, UniformGrid
    torch.manual_seed(0)
    model, x, y, n = build()
    crit = nn.MSELoss()
    base = rss_mb()
    if method == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
        peak = rss_mb()
        persistent = n * (4 + 4 + 4 + 4)  # w, g, m, v
    else:
        beta1 = 0.9 if method == "gdmc-v2" else 0.0
        opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4),
                            beta=2.0, move_frac=move_frac, beta1=beta1, k=1)
        for _ in range(steps):
            def closure():
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                loss.backward()
                return loss
            opt.step(closure)
        peak = rss_mb()
        persistent = n * (4 + 4 + (4 if beta1 > 0 else 0))
    print(json.dumps({"method": method, "move_frac": move_frac, "params": n,
                      "base_mb": base, "peak_mb": peak, "growth_mb": peak - base,
                      "growth_bytes_per_param": (peak - base) * 1024 * 1024 / n,
                      "persistent_bytes_per_param": persistent / n}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True)
    ap.add_argument("--move-frac", type=float, default=0.01)
    a = ap.parse_args(); run(a.method, a.move_frac)
