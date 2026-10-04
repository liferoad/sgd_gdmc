"""Tests for the training loop and runner (regressions for reported defects)."""
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLP
from src.runner import run_one


def tiny_loaders(n=256, d=16, classes=3, batch=32):
    x = torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, d)
    x = x + 0.01 * torch.randn(n, d)
    y = torch.randint(0, classes, (n,))
    tr = DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)
    te = DataLoader(TensorDataset(x[:64], y[:64]), batch_size=64, shuffle=False)
    return tr, te


def model_fn():
    return MLP(in_dim=16, hidden=(16,), out_dim=3)


def base_kwargs(tr, te, crit, **over):
    kw = dict(task="t", model_name="mlp", model_fn=model_fn,
              train_loader=tr, test_loader=te, criterion=crit,
              epochs=2, batch_size=32, momentum=0.0, beta=2.0, move_frac=0.1,
              device="cpu", is_classification=True, log_every=5)
    kw.update(over)
    return kw


def test_final_train_loss_is_not_nan():
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    r = run_one(optimizer_name="gdmc", lr=0.0, seed=0, grid_spec="uniform",
                bits=4, eval_grid_spec="uniform", eval_bits=4,
                beta1=0.9, k=1, **base_kwargs(tr, te, crit))
    assert not math.isnan(r.final_train_loss)
    assert r.final_train_loss > 0.0


def test_adam_final_train_loss_is_not_nan():
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    r = run_one(optimizer_name="adam", lr=1e-3, seed=0, grid_spec="none",
                bits=32, eval_grid_spec=None, eval_bits=None,
                **base_kwargs(tr, te, crit))
    assert not math.isnan(r.final_train_loss)


def test_separate_acceptance_batch_runs_and_is_recorded():
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    r = run_one(optimizer_name="gdmc", lr=0.0, seed=0, grid_spec="uniform",
                bits=4, eval_grid_spec="uniform", eval_bits=4, beta1=0.9, k=1,
                accept_on="separate", accept_split=0.5,
                **base_kwargs(tr, te, crit))
    assert 0.0 <= r.mean_acceptance_rate <= 1.0
    assert not math.isnan(r.final_delta_loss)
    assert r.final_num_moved > 0


def test_separate_acceptance_without_source_raises():
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    with pytest.raises(ValueError):
        run_one(optimizer_name="gdmc", lr=0.0, seed=0, grid_spec="uniform",
                bits=4, eval_grid_spec="uniform", eval_bits=4, beta1=0.9, k=1,
                accept_on="separate", **base_kwargs(tr, te, crit))


def test_grad_noise_path_runs_for_standard_optimizer():
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    r = run_one(optimizer_name="adam", lr=1e-3, seed=0, grid_spec="none",
                bits=32, eval_grid_spec=None, eval_bits=None,
                grad_noise_rho=1.0, **base_kwargs(tr, te, crit))
    assert not math.isnan(r.final_train_loss)


@pytest.mark.parametrize("optimizer_name", ["adam8bit", "signsgd"])
def test_new_baselines_run_through_the_runner(optimizer_name):
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    r = run_one(optimizer_name=optimizer_name, lr=1e-3, seed=0,
                grid_spec="none", bits=32, eval_grid_spec=None, eval_bits=None,
                **base_kwargs(tr, te, crit))
    assert not math.isnan(r.final_train_loss)
    assert r.best_test_acc > 0.0


def test_peak_memory_is_reported():
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    r = run_one(optimizer_name="gdmc", lr=0.0, seed=0, grid_spec="uniform",
                bits=4, eval_grid_spec="uniform", eval_bits=4, beta1=0.9, k=1,
                **base_kwargs(tr, te, crit))
    assert r.peak_rss_mb > 0.0
    assert r.rss_growth_mb >= 0.0


def test_eval_restores_training_mode():
    from src.train import _eval
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(),
                          nn.Linear(8, 1))
    model.train()
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    _eval(model, loader, "cpu", nn.MSELoss(), is_classification=False)
    assert model.training is True, "eval left the model in eval mode"


def test_accept_loss_is_deterministic_and_preserves_batchnorm():
    from src.train import _accept_loss
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(8, 1))
    x = torch.randn(32, 8)
    y = torch.randn(32, 1)
    crit = nn.MSELoss()
    model.train()
    bn = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)][0]
    mean0, var0 = bn.running_mean.clone(), bn.running_var.clone()
    nbt0 = bn.num_batches_tracked.clone()
    loss1 = float(_accept_loss(model, x, y, crit).item())
    # BatchNorm buffers must not advance and the mode must be restored.
    assert torch.equal(bn.running_mean, mean0)
    assert torch.equal(bn.running_var, var0)
    assert torch.equal(bn.num_batches_tracked, nbt0)
    assert model.training is True
    # dropout is disabled in eval mode, so the acceptance loss is deterministic
    loss2 = float(_accept_loss(model, x, y, crit).item())
    assert abs(loss1 - loss2) < 1e-9


def test_gdmc_step_through_runner_uses_accept_closure():
    """The runner must supply an accept closure, and both compared losses must
    come from it (same batch by default)."""
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    r = run_one(optimizer_name="gdmc", lr=0.0, seed=0, grid_spec="uniform",
                bits=4, eval_grid_spec="uniform", eval_bits=4, beta1=0.9, k=1,
                **base_kwargs(tr, te, crit))
    assert not math.isnan(r.final_delta_loss)
    assert 0.0 <= r.mean_acceptance_rate <= 1.0


def test_curves_include_new_diagnostic_columns(tmp_path):
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    run_one(optimizer_name="gdmc", lr=0.0, seed=0, grid_spec="uniform", bits=4,
            eval_grid_spec="uniform", eval_bits=4, beta1=0.9, k=1,
            curves_dir=str(tmp_path), **base_kwargs(tr, te, crit))
    files = list(tmp_path.glob("*.csv"))
    assert len(files) == 1
    header = files[0].read_text().splitlines()[0]
    for col in ["delta_loss", "num_moved", "accepted_step_size"]:
        assert col in header


class RecordingLoader:
    """Wraps a DataLoader and records the first feature of each batch."""

    def __init__(self, loader):
        self.loader = loader
        self.order = []

    def __iter__(self):
        for x, y in self.loader:
            # x[:, 0] encodes the sample index (plus tiny noise).
            self.order.append(round(float(x[0, 0].item())))
            yield x, y

    def __len__(self):
        return len(self.loader)


def _order_for(optimizer_name, tr, te, crit):
    rec = RecordingLoader(tr)
    run_one(optimizer_name=optimizer_name, seed=0,
            grid_spec="none" if optimizer_name == "adam" else "uniform",
            bits=32 if optimizer_name == "adam" else 4,
            lr=1e-3 if optimizer_name == "adam" else 0.0,
            eval_grid_spec=None if optimizer_name == "adam" else "uniform",
            eval_bits=None if optimizer_name == "adam" else 4,
            beta1=0.0 if optimizer_name == "adam" else 0.9, k=1,
            **base_kwargs(rec, te, crit, epochs=1))
    return rec.order


def test_same_seed_gives_same_data_order_across_optimizers():
    """Common random numbers: GDMC must not perturb the global RNG stream."""
    tr, te = tiny_loaders()
    crit = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    adam_order = _order_for("adam", tr, te, crit)
    tr2, te2 = tiny_loaders()
    torch.manual_seed(0)
    gdmc_order = _order_for("gdmc", tr2, te2, crit)
    assert adam_order == gdmc_order
