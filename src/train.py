"""Generic training loop that supports any optimizer with a uniform step API.

The training loop handles two styles of ``optimizer.step()``:

* Standard: ``optimizer.zero_grad(); loss.backward(); optimizer.step()``
* Closure (used by GDMC and SGLD): ``optimizer.step(closure)`` where
  ``closure()`` zeroes grads, computes loss, and backprops.

We detect the style by the presence of the ``requires_closure`` attribute
on the optimizer (set by ``GDMCOptimizer`` and ``SGLD``). If True, the
loop uses the closure interface; otherwise the standard interface.

The loop records per-epoch and per-step metrics in a list of dicts
that the experiment scripts then write to CSV.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 5
    log_every: int = 50  # steps
    device: str = "cpu"
    target_loss: Optional[float] = None  # early-stop if reached
    target_acc: Optional[float] = None  # early-stop if reached (test set)
    is_classification: bool = True  # if False, treat y as a float (regression)


@dataclass
class StepRecord:
    step: int
    epoch: int
    loss: float
    test_loss: Optional[float] = None
    test_acc: Optional[float] = None
    accepted: Optional[bool] = None
    acceptance_rate: Optional[float] = None


def _eval(model, loader, device, criterion, is_classification: bool = True):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # For MSE-style loss, cast target to float.
            if not is_classification:
                y_eval = y.float()
            else:
                y_eval = y
            total_loss += float(criterion(out, y_eval).item()) * x.size(0)
            if is_classification:
                total_correct += int((out.argmax(1) == y).sum().item())
            total += x.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def _requires_closure(opt) -> bool:
    return getattr(opt, "requires_closure", False)


def train(
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    criterion,
    cfg: TrainConfig,
    eval_fn: Optional[Callable[[nn.Module], tuple]] = None,
    step_records: Optional[List[StepRecord]] = None,
    snap_fn: Optional[Callable[[nn.Module], None]] = None,
) -> List[StepRecord]:
    """Train ``model`` with ``optimizer`` for ``cfg.epochs`` epochs.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer or ProjectedGD or GDMCOptimizer or SGLD
    train_loader : DataLoader
    test_loader : DataLoader or None
    criterion : loss function
    cfg : TrainConfig
    eval_fn : optional callable(model) -> (test_loss, test_acc). If None,
        uses the default ``_eval``.
    snap_fn : optional callable(model) to snap weights to a grid (e.g. for
        evaluation under quantization). If None, no snap.
    step_records : list to append to. Created if None.
    """
    if step_records is None:
        step_records = []
    if eval_fn is None and test_loader is not None:
        def eval_fn(m):
            return _eval(m, test_loader, cfg.device, criterion, cfg.is_classification)

    model.to(cfg.device)
    use_closure = _requires_closure(optimizer)
    step_idx = 0
    t0 = time.time()
    early_stop = False
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            if not cfg.is_classification:
                y = y.float()
            if use_closure:
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                loss_val = float(loss.item())
            else:
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())
            # Optionally snap weights for evaluation (e.g. grid quantization).
            test_loss = test_acc = None
            if (step_idx % cfg.log_every == 0 or step_idx == 0) and eval_fn is not None:
                if snap_fn is not None:
                    # Temporarily snap to grid for the eval, then snap back.
                    with torch.no_grad():
                        snaps = [p.data.clone() for p in model.parameters()]
                        snap_fn(model)
                    test_loss, test_acc = eval_fn(model)
                    with torch.no_grad():
                        for p, s in zip(model.parameters(), snaps):
                            p.data.copy_(s)
                else:
                    test_loss, test_acc = eval_fn(model)
            rec = StepRecord(
                step=step_idx, epoch=epoch, loss=loss_val,
                test_loss=test_loss, test_acc=test_acc,
                accepted=getattr(optimizer, "last_accepted", None),
                acceptance_rate=getattr(optimizer, "last_acceptance_rate", None),
            )
            step_records.append(rec)
            step_idx += 1
            if cfg.target_loss is not None and loss_val <= cfg.target_loss:
                early_stop = True
                break
            if cfg.target_acc is not None and test_acc is not None and test_acc >= cfg.target_acc:
                early_stop = True
                break
        if early_stop:
            break
    # Final evaluation under the model's current (possibly snapped) weights.
    if eval_fn is not None:
        if snap_fn is not None:
            with torch.no_grad():
                snap_fn(model)
        test_loss, test_acc = eval_fn(model)
        step_records.append(StepRecord(
            step=step_idx, epoch=cfg.epochs, loss=float("nan"),
            test_loss=test_loss, test_acc=test_acc,
            accepted=getattr(optimizer, "last_accepted", None),
            acceptance_rate=getattr(optimizer, "last_acceptance_rate", None),
        ))
    return step_records
