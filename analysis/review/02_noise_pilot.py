"""Review probe 2: gradient-noise stress pilot.

Pilot for the feedback's central experiment: inject norm-scaled Gaussian
noise into the *proposal* gradient only, keeping the acceptance
evaluation clean, and compare FP32 Adam against GDMC at 4/8 bits with and
without Metropolis acceptance.

Protocol
--------
* Task: MNIST MLP 784-256-256-10, full 60K train, batch 128, 8 epochs.
* Fixed initialization per seed (state_dict reused across all cells).
* Deterministic batch order (shuffle=False) so cells differ only in
  optimizer and noise level.
* Noise: eps = rho * RMS(g_tensor) * N(0,1), rho in {0, .1, .3, 1, 3}.
  Applied to the proposal gradient only; the Metropolis loss comparison
  is a clean forward pass on the same minibatch.
* GDMC config: move_frac=0.01, k=1, beta1=0.9, beta=2.0
  ("-noacc" sets beta=0.0, i.e. always accept).

This is a pilot, not a powered result: 3 seeds, one task, Gaussian noise.

Usage: .venv/bin/python analysis/review/02_noise_pilot.py [--epochs N] [--seeds 0 1 2]
"""
from __future__ import annotations
import argparse, copy, csv, sys, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.mnist import make_mnist            # noqa: E402
from src.models.mlp import MLP                   # noqa: E402
from src.gdmc import GDMCOptimizer, UniformGrid  # noqa: E402


def rms(t):
    return float(t.pow(2).mean().sqrt().item())


def perturb_(params, rho, gen):
    """Add norm-scaled Gaussian noise to .grad in place (proposal only)."""
    if rho == 0.0:
        return
    for p in params:
        g = p.grad
        if g is None:
            continue
        s = rho * rms(g)
        if s > 0:
            g.add_(torch.randn(g.shape, generator=gen, dtype=g.dtype) * s)


def evaluate(model, loader, snap=None):
    model.eval()
    if snap is not None:
        with torch.no_grad():
            for p in model.parameters():
                p.data.copy_(snap.snap(p.data))
    tot = corr = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            corr += int((out.argmax(1) == y).sum().item())
            tot += y.numel()
    return corr / tot


def train_cell(model0, train_loader, test_loader, method, bits, rho, seed, epochs):
    model = copy.deepcopy(model0)
    params = list(model.parameters())
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(1000 * seed + int(rho * 100) + bits)
    grid = None
    if method == "adam":
        opt = torch.optim.Adam(params, lr=1e-3)
    else:
        grid = UniformGrid(bits=bits)
        beta = 0.0 if method.endswith("noacc") else 2.0
        # The optimizer now has a first-class noise hook, so no monkey-patching
        # is needed. (Superseded by experiments/14_noise_study.py, which runs
        # this design through the standard runner.)
        opt = GDMCOptimizer(params, grid=grid, beta=beta, move_frac=0.01,
                            beta1=0.9, k=1, rng=gen, noise_rho=rho,
                            noise_generator=gen)

    best = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            if method == "adam":
                opt.zero_grad(set_to_none=True)
                crit(model(x), y).backward()
                perturb_(params, rho, gen)
                opt.step()
            else:
                def closure():
                    opt.zero_grad(set_to_none=True)
                    loss = crit(model(x), y)
                    loss.backward()
                    return loss
                opt.step(closure)
        acc = evaluate(model, test_loader, snap=grid)
        best = max(best, acc)
    return {"method": method, "bits": bits, "rho": rho, "seed": seed,
            "best_test_acc": best, "final_test_acc": acc,
            "acceptance": float(getattr(opt, "last_acceptance_rate", float("nan"))),
            "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--out", type=str, default="results/review/noise_pilot.csv")
    a = ap.parse_args()

    torch.set_num_threads(8)
    tr, te = make_mnist(root="data")
    tr = TensorDataset(tr.tensors[0].view(-1, 784), tr.tensors[1])
    te = TensorDataset(te.tensors[0].view(-1, 784), te.tensors[1])
    tr_loader = DataLoader(tr, batch_size=128, shuffle=False)
    te_loader = DataLoader(te, batch_size=512, shuffle=False)

    methods = [("adam", 0), ("gdmc", 4), ("gdmc", 8), ("gdmc-noacc", 8)]
    rows = []
    total = len(a.seeds) * len(a.rhos) * len(methods)
    i = 0
    for seed in a.seeds:
        torch.manual_seed(seed)
        model0 = MLP(in_dim=784, hidden=(256, 256), out_dim=10)
        init = copy.deepcopy(model0.state_dict())
        for rho in a.rhos:
            for method, bits in methods:
                model0.load_state_dict(init)
                r = train_cell(model0, tr_loader, te_loader, method, bits, rho,
                               seed, a.epochs)
                rows.append(r); i += 1
                print(f"[{i}/{total}] seed={seed} rho={rho} {method}-b{bits} "
                      f"best={r['best_test_acc']:.4f} final={r['final_test_acc']:.4f} "
                      f"acc_rate={r['acceptance']:.3f} ({r['wall_sec']:.0f}s)",
                      flush=True)
                out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
    print("done ->", a.out, flush=True)


if __name__ == "__main__":
    main()
