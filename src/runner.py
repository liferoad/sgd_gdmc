"""Shared experiment runner.

Given a configuration dict describing one (task, model, optimizer, grid,
seed), this module builds the model, data loaders, optimizer, and
training loop, runs the training, and writes a single CSV row recording
the configuration plus final test metrics.

Reproducibility note: each run gets its own torch.Generator, which is
handed to the optimizer.  Optimizers therefore no longer perturb the
global RNG stream, so two optimizers run with the same seed see the same
minibatch order (common random numbers).
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import StepRecord, TrainConfig, train


def peak_rss_mb() -> float:
    """Process peak resident set size in MiB (macOS bytes, Linux KiB)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


@dataclass
class RunResult:
    """One row of per-run output (config + final metrics)."""
    task: str
    model: str
    optimizer: str
    grid_spec: str
    bits: int
    seed: int
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    beta: float
    move_frac: float
    beta1: float = 0.0
    k: int = 1
    extra: str = ""  # JSON-encoded extra config (e.g. adaptive alpha)
    final_train_loss: float = float("nan")
    final_test_loss: float = float("nan")
    final_test_acc: float = float("nan")
    best_test_acc: float = float("nan")
    best_test_loss: float = float("nan")
    final_acceptance_rate: float = float("nan")
    mean_acceptance_rate: float = float("nan")
    final_delta_loss: float = float("nan")
    final_num_moved: int = 0
    final_train_loss_steps: int = 0
    wall_time_sec: float = float("nan")
    peak_rss_mb: float = float("nan")
    rss_growth_mb: float = float("nan")
    # Extra columns for sweeps:
    weight_mse_vs_continuous: float = float("nan")


def _build_optimizer(name, params, cfg, grid):
    name = name.lower()
    rng = cfg.get("rng")
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.0)
    if name in ("momentum", "momentum-sgd"):
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9))
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    if name in ("adam8bit", "adam-8bit", "adam8"):
        from .baselines.adam8bit import Adam8Bit
        return Adam8Bit(params, lr=cfg["lr"])
    if name in ("signsgd", "sign-sgd", "momentum-sign"):
        from .baselines.signsgd import SignSGD
        return SignSGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9) or 0.9)
    if name == "sgld":
        from .baselines.sgld import SGLD
        return SGLD(params, lr=cfg["lr"], beta=cfg.get("beta", 1.0), rng=rng)
    if name in ("projected-gd", "projected_gd", "projectedgd",
                "projected-adam", "projected-momentum", "projected-msgd"):
        from .baselines.projected_gd import ProjectedGD
        base = "momentum" if name in ("projected-momentum", "projected-msgd") \
            else cfg.get("project_base", "adam")
        return ProjectedGD(params, base=base, grid=grid, lr=cfg["lr"],
                           lr_scale=cfg.get("project_lr_scale"))
    if name in ("gdmc", "gdmc-uniform", "gdmc-auto", "gdmc-v3",
                "gdmc-v1", "gdmc-v2"):
        from .gdmc import GDMCOptimizer
        # "gdmc-auto" / "gdmc-v3" select the magnitude-scaled step mode.
        kmode = "auto" if name in ("gdmc-auto", "gdmc-v3") else cfg.get("k_mode", "fixed")
        return GDMCOptimizer(params, grid=grid,
                             beta=cfg.get("beta", 2.0),
                             move_frac=cfg.get("move_frac", 0.01),
                             accept_on=cfg.get("accept_on", "minibatch"),
                             beta1=cfg.get("beta1", 0.0),
                             k=cfg.get("k", 1),
                             k_mode=kmode,
                             step_scale=cfg.get("step_scale", 1e-3),
                             k_max=cfg.get("k_max", 1 << 22),
                             grad_ema=cfg.get("grad_ema", 0.99),
                             rng=rng,
                             select_mode=cfg.get("select_mode", "bernoulli"),
                             noise_rho=cfg.get("grad_noise_rho", 0.0))
    if name in ("gdmc-adaptive",):
        from .gdmc import GDMCOptimizer, AdaptiveGrid, make_grid
        return GDMCOptimizer(params,
                             grid=make_grid("uniform-pt", bits=cfg.get("bits", 4)),
                             beta=cfg.get("beta", 2.0),
                             move_frac=cfg.get("move_frac", 0.01),
                             accept_on=cfg.get("accept_on", "minibatch"),
                             beta1=cfg.get("beta1", 0.0),
                             k=cfg.get("k", 1),
                             k_mode=cfg.get("k_mode", "fixed"),
                             step_scale=cfg.get("step_scale", 1e-3),
                             k_max=cfg.get("k_max", 1 << 22),
                             grad_ema=cfg.get("grad_ema", 0.99),
                             rng=rng,
                             select_mode=cfg.get("select_mode", "bernoulli"),
                             noise_rho=cfg.get("grad_noise_rho", 0.0))
    raise ValueError(f"unknown optimizer: {name!r}")


def _grid_for(spec, bits, alpha=1.05):
    from .gdmc import make_grid, AdaptiveGrid
    spec = spec.lower()
    if spec == "none":
        return None
    if spec == "adaptive":
        return AdaptiveGrid(bits=bits, alpha=alpha)
    return make_grid(spec, bits=bits)


def _set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(
    task,
    model_name,
    model_fn,
    optimizer_name,
    grid_spec,
    bits,
    train_loader,
    test_loader,
    criterion,
    epochs,
    batch_size,
    lr,
    momentum,
    beta,
    move_frac,
    seed,
    device="cpu",
    is_classification=True,
    log_every=200,
    extra=None,
    eval_grid_spec=None,
    eval_bits=None,
    beta1=0.0,
    k=1,
    k_mode="fixed",
    step_scale=1e-3,
    k_max=1 << 22,
    grad_ema=0.99,
    curves_dir=None,
    # --- new controls (all default to the historical behaviour) ---
    project_lr_scale=None,
    grad_noise_rho=0.0,
    accept_on="minibatch",
    accept_split=0.0,
    accept_loader=None,
    select_mode="bernoulli",
):
    """Run a single (model, optimizer, grid, seed) configuration.

    project_lr_scale : optional float
        For projected-gd: set lr = project_lr_scale * grid spacing.
    grad_noise_rho : float
        Norm-scaled Gaussian noise on the proposal gradient.
    accept_on / accept_split / accept_loader :
        Metropolis acceptance source; see src/gdmc/optimizer.py.
    select_mode : {"bernoulli", "randperm"}
        Move-set selection mode for GDMC.
    """
    _set_seed(seed)
    rng = torch.Generator().manual_seed(int(seed))
    model = model_fn()
    model.to(device)
    grid = _grid_for(grid_spec, bits) if grid_spec else None
    cfg = dict(lr=lr, momentum=momentum, beta=beta, move_frac=move_frac,
               project_base=(extra or {}).get("project_base", "adam"),
               project_lr_scale=project_lr_scale,
               beta1=beta1, k=k, k_mode=k_mode, step_scale=step_scale,
               k_max=k_max, grad_ema=grad_ema, bits=bits,
               grad_noise_rho=grad_noise_rho, accept_on=accept_on,
               select_mode=select_mode, rng=rng)
    optimizer = _build_optimizer(optimizer_name, model.parameters(), cfg, grid)
    is_closure = bool(getattr(optimizer, "requires_closure", False))

    cfg_train = TrainConfig(
        epochs=epochs, log_every=log_every, device=device,
        is_classification=is_classification,
        accept_split=accept_split,
        accept_loader=accept_loader,
        # GDMC applies the noise itself; the loop must not add it twice.
        grad_noise_rho=0.0 if is_closure else grad_noise_rho,
        grad_noise_generator=None if is_closure else rng,
    )

    snap_fn = None
    if eval_grid_spec is not None and eval_bits is not None:
        eval_grid = _grid_for(eval_grid_spec, eval_bits)
        def snap_fn(m):
            with torch.no_grad():
                for p in m.parameters():
                    p.data.copy_(eval_grid.snap(p.data))
    elif grid is not None and optimizer_name in ("sgd", "momentum", "momentum-sgd", "adam", "sgld"):
        def snap_fn(m):
            with torch.no_grad():
                for p in m.parameters():
                    p.data.copy_(grid.snap(p.data))

    rss_before = peak_rss_mb()
    t0 = time.time()
    records = train(model, optimizer, train_loader, test_loader, criterion, cfg_train,
                    snap_fn=snap_fn)
    wall = time.time() - t0
    rss_after = peak_rss_mb()

    if curves_dir is not None:
        cdir = Path(curves_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        fname = (f"{task}__{optimizer_name}__{grid_spec or 'none'}__b{bits}"
                 f"__seed{seed}")
        if beta1 or k != 1:
            fname += f"__b1{beta1}__k{k}"
        if k_mode == "auto":
            fname += f"__auto__ss{step_scale:g}"
        if grad_noise_rho:
            fname += f"__noise{grad_noise_rho:g}"
        if accept_on != "minibatch":
            fname += f"__acc{accept_on}"
        fname += ".csv"
        write_curve_csv(records, cdir / fname)

    final = records[-1]
    test_losses = [r.test_loss for r in records if r.test_loss is not None]
    test_accs = [r.test_acc for r in records if r.test_acc is not None]
    best_loss = min(test_losses) if test_losses else float("nan")
    best_acc = max(test_accs) if test_accs else float("nan")
    acc_rates = [r.acceptance_rate for r in records if r.acceptance_rate is not None]

    return RunResult(
        task=task,
        model=model_name,
        optimizer=optimizer_name,
        grid_spec=grid_spec or "none",
        bits=bits,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        momentum=momentum,
        beta=beta,
        move_frac=move_frac,
        beta1=beta1,
        k=k,
        extra=json.dumps({**(extra or {}), "k_mode": k_mode,
                          "step_scale": step_scale,
                          "project_lr_scale": project_lr_scale,
                          "grad_noise_rho": grad_noise_rho,
                          "accept_on": accept_on,
                          "select_mode": select_mode}),
        final_train_loss=float(final.loss),
        final_test_loss=float(final.test_loss) if final.test_loss is not None else float("nan"),
        final_test_acc=float(final.test_acc) if final.test_acc is not None else float("nan"),
        best_test_loss=best_loss,
        best_test_acc=best_acc,
        final_acceptance_rate=float(getattr(optimizer, "last_acceptance_rate", float("nan"))),
        mean_acceptance_rate=float(np.mean(acc_rates)) if acc_rates else float("nan"),
        final_delta_loss=(float("nan") if getattr(optimizer, "last_delta_loss", None) is None
                          else float(optimizer.last_delta_loss)),
        final_num_moved=int(getattr(optimizer, "last_num_moved", 0) or 0),
        final_train_loss_steps=len(records),
        wall_time_sec=wall,
        peak_rss_mb=rss_after,
        rss_growth_mb=max(0.0, rss_after - rss_before),
    )


def write_results_csv(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


CURVE_FIELDS = ["step", "epoch", "loss", "test_loss", "test_acc",
                "accepted", "acceptance_rate", "delta_loss", "num_moved",
                "accepted_step_size"]


def write_curve_csv(records, path):
    """Write the per-step training curve for a single run to a CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k, "") for k in CURVE_FIELDS})


def read_curves(curves_dir, pattern="*.csv"):
    """Load all per-run curve CSVs in curves_dir into one DataFrame.

    Parses the run config out of the filename (run_one format). Requires pandas.
    """
    import pandas as pd
    rows = []
    for p in sorted(Path(curves_dir).glob(pattern)):
        stem = p.stem
        parts = stem.split("__")
        meta = {"file": stem}
        if len(parts) >= 5:
            meta["task"] = parts[0]
            meta["optimizer"] = parts[1]
            meta["grid_spec"] = parts[2]
            meta["bits"] = int(parts[3].lstrip("b"))
            meta["seed"] = int(parts[4].lstrip("seed"))
            for extra in parts[5:]:
                if extra.startswith("b1"):
                    meta["beta1"] = float(extra[2:])
                elif extra.startswith("ss"):
                    meta["step_scale"] = float(extra[2:])
                elif extra.startswith("k"):
                    meta["k"] = int(extra[1:])
                elif extra.startswith("noise"):
                    meta["grad_noise_rho"] = float(extra[5:])
        df = pd.read_csv(p)
        for k, v in meta.items():
            df[k] = v
        rows.append(df)
    if not rows:
        raise SystemExit(f"no curve CSVs in {curves_dir}")
    return pd.concat(rows, ignore_index=True)
