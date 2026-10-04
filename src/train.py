"""Generic training loop that supports any optimizer with a uniform step API.

The training loop handles two styles of optimizer.step():

* Standard: optimizer.zero_grad(); loss.backward(); optimizer.step()
* Closure (used by GDMC and SGLD): optimizer.step(closure) where
  closure() zeroes grads, computes loss, and backprops.

We detect the style by the presence of the requires_closure attribute
on the optimizer (set by GDMCOptimizer and SGLD). If True, the loop uses
the closure interface; otherwise the standard interface.

Two features that the GDMC study needs are wired in here:

* separate acceptance batches (TrainConfig.accept_split / accept_loader)
  so the Metropolis test is not evaluated on the same minibatch that
  produced the proposal gradient;
* norm-scaled gradient noise (TrainConfig.grad_noise_rho) applied to the
  proposal gradient of standard optimizers. GDMCOptimizer applies the
  same noise internally, so a run should set it in exactly one place.

The loop records per-epoch and per-step metrics in a list of dicts
that the experiment scripts then write to CSV.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional

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
    # --- acceptance-batch controls (GDMC with accept_on="separate") ---
    accept_split: float = 0.0      # >0: hold out this fraction of each batch
    accept_loader: Optional[DataLoader] = None  # alternative source, cycled
    # --- gradient noise for non-closure optimizers ---
    grad_noise_rho: float = 0.0
    grad_noise_generator: Optional[torch.Generator] = None


@dataclass
class StepRecord:
    step: int
    epoch: int
    loss: float
    test_loss: Optional[float] = None
    test_acc: Optional[float] = None
    accepted: Optional[bool] = None
    acceptance_rate: Optional[float] = None
    delta_loss: Optional[float] = None
    num_moved: Optional[int] = None
    accepted_step_size: Optional[float] = None


def _eval(model, loader, device, criterion, is_classification: bool = True):
    was_training = model.training
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
    # Restore the mode the caller was in: leaving the model in eval() mode
    # silently disables dropout for the rest of the epoch.
    model.train(was_training)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def _bn_buffers(model):
    """Snapshot the running statistics of every BatchNorm module."""
    out = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            out.append((m,
                        m.running_mean.clone() if m.running_mean is not None else None,
                        m.running_var.clone() if m.running_var is not None else None,
                        m.num_batches_tracked.clone()
                        if getattr(m, "num_batches_tracked", None) is not None else None))
    return out


def _accept_loss(model, x, y, criterion):
    """Deterministic acceptance loss.

    The Metropolis comparison must not be contaminated by dropout sampling, and
    it must not permanently advance BatchNorm running statistics: the proposal
    forward pass would otherwise move them even when the move is rejected. We
    therefore evaluate in eval() mode (no dropout, running stats used) and
    restore both the mode and the BatchNorm buffers afterwards.
    """
    if x is None or y is None:
        raise ValueError("acceptance batch is missing")
    was_training = model.training
    saved = _bn_buffers(model)
    model.eval()
    try:
        with torch.no_grad():
            return criterion(model(x), y)
    finally:
        for m, mean, var, nbt in saved:
            if mean is not None:
                m.running_mean.copy_(mean)
            if var is not None:
                m.running_var.copy_(var)
            if nbt is not None and getattr(m, "num_batches_tracked", None) is not None:
                m.num_batches_tracked.copy_(nbt)
        model.train(was_training)


def _requires_closure(opt) -> bool:
    return getattr(opt, "requires_closure", False)


def _accept_on(opt) -> str:
    try:
        return str(opt.param_groups[0].get("accept_on", "minibatch"))
    except (AttributeError, IndexError, KeyError):
        return "minibatch"


def _device_gen(gen: Optional[torch.Generator], device, cache: dict):
    """Return a generator whose device matches the tensor's (torch requires it)."""
    if gen is None:
        return None
    device = torch.device(device)
    if gen.device == device:
        return gen
    key = str(device)
    if key not in cache:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen).item())
        cache[key] = torch.Generator(device=device).manual_seed(seed)
    return cache[key]


def _add_grad_noise(params: Iterable[torch.nn.Parameter], rho: float,
                    gen: Optional[torch.Generator]) -> None:
    """Add rho * RMS(g) * N(0,1) to every populated .grad, in place."""
    if rho <= 0.0:
        return
    cache: dict = {}
    for p in params:
        if p.grad is None:
            continue
        s = rho * float(p.grad.detach().pow(2).mean().sqrt().item())
        if s <= 0.0:
            continue
        dgen = _device_gen(gen, p.grad.device, cache)
        if dgen is None:
            p.grad.add_(torch.randn_like(p.grad) * s)
        else:
            p.grad.add_(torch.randn(p.grad.shape, generator=dgen,
                                    dtype=p.grad.dtype, device=p.grad.device) * s)


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


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
    """Train model with optimizer for cfg.epochs epochs.

    Parameters
    ----------
    eval_fn : optional callable(model) -> (test_loss, test_acc).
    snap_fn : optional callable(model) to snap weights to a grid for
        evaluation. If None, no snap.
    step_records : list to append to. Created if None.
    """
    if step_records is None:
        step_records = []
    if eval_fn is None and test_loader is not None:
        def eval_fn(m):
            return _eval(m, test_loader, cfg.device, criterion, cfg.is_classification)

    model.to(cfg.device)
    use_closure = _requires_closure(optimizer)
    requires_accept = use_closure and getattr(optimizer, "requires_accept_closure", False)
    separate_accept = requires_accept and _accept_on(optimizer) == "separate"
    if separate_accept and cfg.accept_split <= 0.0 and cfg.accept_loader is None:
        raise ValueError(
            "optimizer uses accept_on='separate' but neither accept_split "
            "nor accept_loader was configured"
        )
    accept_iter = _cycle(cfg.accept_loader) if cfg.accept_loader is not None else None

    params = [p for p in model.parameters()]
    step_idx = 0
    last_train_loss = float("nan")
    early_stop = False
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            if not cfg.is_classification:
                y = y.float()
            if use_closure:
                xa = ya = None
                if separate_accept:
                    if accept_iter is not None:
                        xa, ya = next(accept_iter)
                        xa, ya = xa.to(cfg.device), ya.to(cfg.device)
                        if not cfg.is_classification:
                            ya = ya.float()
                    else:
                        n = x.size(0)
                        k = min(max(int(round(n * cfg.accept_split)), 1), n - 1)
                        xg, yg = x[:n - k], y[:n - k]
                        xa, ya = x[n - k:], y[n - k:]
                        x, y = xg, yg
                elif requires_accept:
                    # Same-batch acceptance, but still a deterministic,
                    # BatchNorm-preserving evaluation.
                    xa, ya = x, y
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    return loss
                if requires_accept:
                    def accept_closure():
                        return _accept_loss(model, xa, ya, criterion)
                    loss = optimizer.step(closure, accept_closure)
                else:
                    loss = optimizer.step(closure)
                loss_val = float(loss.item())
            else:
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                _add_grad_noise(params, cfg.grad_noise_rho,
                                cfg.grad_noise_generator)
                optimizer.step()
                loss_val = float(loss.item())
            last_train_loss = loss_val
            test_loss = test_acc = None
            if (step_idx % cfg.log_every == 0 or step_idx == 0) and eval_fn is not None:
                if snap_fn is not None:
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
                delta_loss=getattr(optimizer, "last_delta_loss", None),
                num_moved=getattr(optimizer, "last_num_moved", None),
                accepted_step_size=getattr(optimizer, "last_accepted_step_size", None),
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
    # Final evaluation. The synthetic final record keeps the last real
    # training loss so that final_train_loss is not NaN.
    if eval_fn is not None:
        if snap_fn is not None:
            with torch.no_grad():
                snap_fn(model)
        test_loss, test_acc = eval_fn(model)
        step_records.append(StepRecord(
            step=step_idx, epoch=cfg.epochs, loss=last_train_loss,
            test_loss=test_loss, test_acc=test_acc,
            accepted=getattr(optimizer, "last_accepted", None),
            acceptance_rate=getattr(optimizer, "last_acceptance_rate", None),
            delta_loss=getattr(optimizer, "last_delta_loss", None),
            num_moved=getattr(optimizer, "last_num_moved", None),
            accepted_step_size=getattr(optimizer, "last_accepted_step_size", None),
        ))
    return step_records
