"""Tests for GDMCOptimizer.

The important regression here is test_proposal_changes_weights_at_every_bit_width:
the original code silently produced bit-identical weights at 16/32 bits (the
single grid step was below float32 precision) and nothing caught it.
"""
import pytest
import torch
import torch.nn as nn

from src.gdmc import GDMCOptimizer, AdaptiveGrid, UniformGrid, make_grid


def make_model(seed=0, dim=8):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))


def make_data(n=16, d=8):
    torch.manual_seed(1)
    return torch.randn(n, d), torch.randn(n, 1)


def on_grid(model, grid, tol=1e-5):
    delta = (grid.vmax - grid.vmin) / (grid.levels - 1)
    for p in model.parameters():
        idx = (p.data - grid.vmin) / delta
        err = float((idx - idx.round()).abs().max().item()) * delta
        assert err < tol


def total_change(model, before):
    return sum(float((p.data - b).abs().sum().item())
               for p, b in zip(model.parameters(), before))


@pytest.mark.parametrize("bits", [2, 4, 8, 16, 32])
def test_proposal_changes_weights_at_every_bit_width(bits):
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    grid = UniformGrid(bits=bits)
    opt = GDMCOptimizer(model.parameters(), grid=grid, beta=2.0, move_frac=0.1,
                        rng=torch.Generator().manual_seed(0))
    before = [p.data.clone() for p in model.parameters()]

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    assert opt.last_num_moved > 0
    assert total_change(model, before) > 0.0, "proposal was a no-op"
    on_grid(model, grid)


@pytest.mark.parametrize("bits", [16, 32])
def test_auto_k_moves_at_fine_grids(bits):
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=bits), beta=2.0,
                        move_frac=0.1, beta1=0.9, k_mode="auto", step_scale=1e-2,
                        rng=torch.Generator().manual_seed(0))
    before = [p.data.clone() for p in model.parameters()]

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    assert opt.last_mean_k is not None and opt.last_mean_k >= 1.0
    assert total_change(model, before) > 0.0


def test_rejected_move_rolls_back_exactly():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    grid = UniformGrid(bits=4)
    opt = GDMCOptimizer(model.parameters(), grid=grid, beta=2.0, move_frac=0.5,
                        rng=torch.Generator().manual_seed(0))
    opt._accept = lambda lo, ln, beta: False  # force the rejection branch
    snapped = [grid.snap(p.data).clone() for p in model.parameters()]

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    assert opt.last_accepted is False
    assert all(torch.equal(p.data, s)
               for p, s in zip(model.parameters(), snapped))


def test_zero_beta_always_accepts():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4), beta=0.0,
                        move_frac=0.5, rng=torch.Generator().manual_seed(0))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    assert opt.last_accepted is True


def test_separate_acceptance_uses_the_accept_closure():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4), beta=1e6,
                        move_frac=0.5, accept_on="separate",
                        rng=torch.Generator().manual_seed(0))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    calls = {"n": 0}

    def rising_accept_closure():
        # small before the proposal, slightly larger after -> reject at beta=1e6
        calls["n"] += 1
        return torch.tensor(100.0 if calls["n"] == 1 else 100.1)

    opt.step(closure, rising_accept_closure)
    assert calls["n"] == 2, "accept_closure must be evaluated before AND after"
    assert opt.last_accepted is False
    assert abs(opt.last_delta_loss - 0.1) < 1e-5


def test_separate_acceptance_compares_both_losses_on_the_accept_batch():
    """Regression: loss_old must come from the accept batch, not the gradient
    batch, otherwise the batch offset dominates the Metropolis decision."""
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4), beta=1e6,
                        move_frac=0.5, accept_on="separate",
                        rng=torch.Generator().manual_seed(0))

    def gradient_closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss * 0.0 + 1000.0  # gradient-batch loss is 1000

    calls = {"n": 0}

    def accept_closure():
        calls["n"] += 1
        return torch.tensor(100.0 if calls["n"] == 1 else 100.1)

    opt.step(gradient_closure, accept_closure)
    # The move slightly *raises* the accept-batch loss, so with beta=1e6 it
    # must be rejected. The old code compared 100.1 against the gradient-batch
    # loss (1000) and accepted.
    assert opt.last_grad_loss == 1000.0
    assert abs(opt.last_loss_old - 100.0) < 1e-5
    assert abs(opt.last_loss_new - 100.1) < 1e-5
    assert opt.last_accepted is False


def test_separate_acceptance_accepts_a_genuine_improvement():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4), beta=1e6,
                        move_frac=0.2, accept_on="separate",
                        rng=torch.Generator().manual_seed(0))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    calls = {"n": 0}

    def falling_accept_closure():
        calls["n"] += 1
        return torch.tensor(100.0 if calls["n"] == 1 else 99.0)

    opt.step(closure, falling_accept_closure)
    assert opt.last_accepted is True


def test_per_tensor_uniform_grid_does_not_rescale_grid_points():
    """UniformGrid(per_tensor=True) must use the tensor's own range for both
    snapping and proposals, not the raw vmin/vmax multipliers."""
    grid = make_grid("uniform-pt", bits=2)
    w = torch.tensor([-0.6, -0.2, 0.2, 0.6])
    vmin, vmax = grid.range_for(w)
    assert abs(vmin + 0.6) < 1e-6 and abs(vmax - 0.6) < 1e-6
    snapped = grid.snap_with_range(w, vmin, vmax)
    assert torch.allclose(snapped, w, atol=1e-6), snapped

    p = nn.Parameter(w.clone())
    opt = GDMCOptimizer([p], grid=grid, beta=2.0, move_frac=0.1,
                        rng=torch.Generator().manual_seed(0))
    opt._snap_inplace(p, grid)
    assert torch.allclose(p.data, w, atol=1e-6), p.data


def test_separate_acceptance_requires_closure():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4),
                        accept_on="separate",
                        rng=torch.Generator().manual_seed(0))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    with pytest.raises(ValueError):
        opt.step(closure)


def test_gradient_noise_changes_proposal_but_not_acceptance_loss():
    crit = nn.MSELoss()
    x, y = make_data()

    def run(rho):
        model = make_model(seed=3)
        opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4), beta=2.0,
                            move_frac=0.3, noise_rho=rho,
                            rng=torch.Generator().manual_seed(5),
                            noise_generator=torch.Generator().manual_seed(9))

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            return loss

        opt.step(closure)
        return [p.data.clone() for p in model.parameters()], opt.last_loss_old

    clean, loss_clean = run(0.0)
    noisy, loss_noisy = run(3.0)
    assert any(not torch.equal(a, b) for a, b in zip(clean, noisy))
    assert abs(loss_clean - loss_noisy) < 1e-9


def test_optimizer_does_not_consume_the_global_rng():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=8), beta=2.0,
                        move_frac=0.2, beta1=0.9,
                        rng=torch.Generator().manual_seed(0))
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["bernoulli", "randperm"])
def test_select_modes_run(mode):
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4), beta=2.0,
                        move_frac=0.05, select_mode=mode,
                        rng=torch.Generator().manual_seed(0))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    on_grid(model, UniformGrid(bits=4))


def test_momentum_state_is_created():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4), beta=2.0,
                        move_frac=0.1, beta1=0.9,
                        rng=torch.Generator().manual_seed(0))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    assert any("m" in st for st in opt.state.values())


@pytest.mark.parametrize("grid", [AdaptiveGrid(bits=4),
                                     make_grid("uniform-pt", bits=4)])
def test_adaptive_grids_run_and_move(grid):
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=grid, beta=2.0, move_frac=0.2,
                        beta1=0.9, rng=torch.Generator().manual_seed(0))
    before = [p.data.clone() for p in model.parameters()]

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    assert total_change(model, before) > 0.0


def test_auto_k_with_adaptive_grid():
    model = make_model()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=AdaptiveGrid(bits=16), beta=2.0,
                        move_frac=0.2, beta1=0.9, k_mode="auto", step_scale=1e-2,
                        rng=torch.Generator().manual_seed(0))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss

    opt.step(closure)
    assert opt.last_mean_k is not None and opt.last_mean_k >= 1.0


def test_unimplemented_accept_on_is_rejected():
    model = make_model()
    with pytest.raises(ValueError):
        GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=4),
                      accept_on="full")
