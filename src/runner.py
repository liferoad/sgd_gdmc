"""Shared experiment runner.

Given a configuration dict describing one (task, model, optimizer, grid,
seed), this module builds the model, data loaders, optimizer, and
training loop, runs the training, and writes a single CSV row
recording the configuration plus final test metrics.

The output of a single run is a row in a per-experiment CSV. Each
experiment script invokes ``run_one`` (or ``run_many``) for a grid of
configurations and aggregates the resulting CSVs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import StepRecord, TrainConfig, train


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
    extra: str  # JSON-encoded extra config (e.g. adaptive alpha)
    final_train_loss: float
    final_test_loss: float
    final_test_acc: float
    best_test_acc: float
    best_test_loss: float
    final_acceptance_rate: float
    final_train_loss_steps: int
    wall_time_sec: float
    # Extra columns for sweeps:
    weight_mse_vs_continuous: float = float("nan")


def _build_optimizer(name, params, cfg, grid):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.0)
    if name in ("momentum", "momentum-sgd"):
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9))
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"])
    if name == "sgld":
        from .baselines.sgld import SGLD
        return SGLD(params, lr=cfg["lr"], beta=cfg.get("beta", 1.0))
    if name in ("projected-gd", "projected_gd", "projectedgd"):
        from .baselines.projected_gd import ProjectedGD
        return ProjectedGD(params, base=cfg.get("project_base", "adam"), grid=grid, lr=cfg["lr"])
    if name in ("gdmc", "gdmc-uniform"):
        from .gdmc import GDMCOptimizer
        return GDMCOptimizer(params, grid=grid,
                             beta=cfg.get("beta", 2.0),
                             move_frac=cfg.get("move_frac", 0.01))
    if name in ("gdmc-adaptive",):
        from .gdmc import GDMCOptimizer, AdaptiveGrid, make_grid
        # ``gdmc-adaptive`` historically meant a per-tensor-uniform grid
        # (range tracks weight magnitude, fixed shape). The fully adaptive
        # grid was retired because it leads to runaway in some cases.
        return GDMCOptimizer(params,
                             grid=make_grid("uniform-pt", bits=cfg.get("bits", 4)),
                             beta=cfg.get("beta", 2.0),
                             move_frac=cfg.get("move_frac", 0.01))
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
):
    """Run a single (model, optimizer, grid, seed) configuration and return a result row.

    ``eval_grid_spec`` / ``eval_bits`` allow you to train with one
    grid and *evaluate* under a different (typically coarser) grid, to
    measure how well the trained weights transfer to low-bit
    deployment. If either is None, evaluation uses the training grid
    (or no snap if no grid was used at training time).
    """
    _set_seed(seed)
    model = model_fn()
    model.to(device)
    grid = _grid_for(grid_spec, bits) if grid_spec else None
    cfg = dict(lr=lr, momentum=momentum, beta=beta, move_frac=move_frac,
               project_base=(extra or {}).get("project_base", "adam"))
    optimizer = _build_optimizer(optimizer_name, model.parameters(), cfg, grid)

    cfg_train = TrainConfig(epochs=epochs, log_every=log_every, device=device,
                            is_classification=is_classification)

    snap_fn = None
    if eval_grid_spec is not None and eval_bits is not None:
        eval_grid = _grid_for(eval_grid_spec, eval_bits)
        def snap_fn(m):
            with torch.no_grad():
                for p in m.parameters():
                    p.data.copy_(eval_grid.snap(p.data))
    elif grid is not None and optimizer_name in ("sgd", "momentum", "momentum-sgd", "adam", "sgld"):
        # For non-quantized optimizers we still want to evaluate under
        # the same grid to make the comparison fair.
        def snap_fn(m):
            with torch.no_grad():
                for p in m.parameters():
                    p.data.copy_(grid.snap(p.data))

    t0 = time.time()
    records = train(model, optimizer, train_loader, test_loader, criterion, cfg_train,
                    snap_fn=snap_fn)
    wall = time.time() - t0

    final = records[-1]
    test_losses = [r.test_loss for r in records if r.test_loss is not None]
    test_accs = [r.test_acc for r in records if r.test_acc is not None]
    best_loss = min(test_losses) if test_losses else float("nan")
    best_acc = max(test_accs) if test_accs else float("nan")

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
        extra=json.dumps(extra or {}),
        final_train_loss=float(final.loss) if not math.isnan(float(final.loss)) else float("nan"),
        final_test_loss=float(final.test_loss) if final.test_loss is not None else float("nan"),
        final_test_acc=float(final.test_acc) if final.test_acc is not None else float("nan"),
        best_test_loss=best_loss,
        best_test_acc=best_acc,
        final_acceptance_rate=float(getattr(optimizer, "last_acceptance_rate", float("nan"))),
        final_train_loss_steps=len(records),
        wall_time_sec=wall,
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
