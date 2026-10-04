"""Tests for GDMCOptimizer.

Two regressions matter most here:

* the proposal must actually move *proposal* weights - comparing against the
  unquantized initialization is not enough, because the initial snap alone can
  make the comparison pass. Movement is therefore measured from the post-snap
  state, and the fixed-step float32 no-op at 32 bits is tested separately from
  the auto-k fix.
* for adaptive/per-tensor grids the range must be captured once per step, so a
  rejected move restores the exact pre-proposal weights instead of re-snapping
  them on a different grid.
"""
import pytest
import torch
import torch.nn as nn

from src.gdmc import GDMCOptimizer, AdaptiveGrid, UniformGrid, make_grid


def make_model(seed=0, dim=8):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))


def make_linear(seed=0, dim=8):
    """A single linear layer: every coordinate gets a non-zero gradient, so a
    no-op proposal can only come from the grid arithmetic itself."""
    torch.manual_seed(seed)
    return nn.Linear(dim, 1)


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


def step_capturing_post_snap(model, opt, closure, accept=True):
    """Run one step and return each parameter's state right after the snap.

    This is the state the proposal starts from; comparing against anything
    earlier (e.g. the unquantized init) would count the snap itself as
    movement.
    """
    captured = {}
    orig_snap = opt._snap_inplace

    def spy(p, grid, *a, **k):
        orig_snap(p, grid, *a, **k)
        captured[id(p)] = p.data.clone()

    opt._snap_inplace = spy
    if not accept:
        opt._accept = lambda *a, **k: False
    try:
        opt.step(closure)
    finally:
        opt._snap_inplace = orig_snap
    return captured


def make_closure(opt, model, x, y, crit):
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        return loss
    return closure


def post_snap_change(model, captured):
    return sum(float((p.data - captured[id(p)]).abs().sum().item())
               for p in model.parameters())


@pytest.mark.parametrize("bits", [2, 4, 8, 16])
def test_fixed_step_proposal_actually_moves_weights(bits):
    model = make_linear()
    x, y = make_data()
    crit = nn.MSELoss()
    grid = UniformGrid(bits=bits)
    opt = GDMCOptimizer(model.parameters(), grid=grid, beta=0.0, move_frac=0.5,
                        k=1, rng=torch.Generator().manual_seed(0))
    captured = step_capturing_post_snap(model, opt,
                                        make_closure(opt, model, x, y, crit))
    assert opt.last_num_selected > 0
    assert post_snap_change(model, captured) > 0.0, "the proposal was a no-op"
    assert opt.last_num_moved > 0
    on_grid(model, grid)


def test_fixed_step_k1_cannot_move_a_32_bit_grid():
    """Documents the limitation v3 exists to fix: one 32-bit grid step
    (4.7e-10) is below the float32 spacing at 0.5 (6e-8), so the proposal is a
    no-op even though coordinates were selected."""
    grid = UniformGrid(bits=32)
    p = nn.Parameter(torch.full((8,), 0.5))
    opt = GDMCOptimizer([p], grid=grid, beta=0.0, move_frac=1.0, k=1,
                        rng=torch.Generator().manual_seed(0))
    before = p.data.clone()

    def closure():
        opt.zero_grad(set_to_none=True)
        p.grad = torch.ones_like(p)
        return torch.tensor(1.0)

    opt.step(closure)
    assert torch.equal(p.data, before)
    assert opt.last_num_selected == 8
    assert opt.last_num_moved == 0, "diagnostics must count real changes"


@pytest.mark.parametrize("bits", [16, 32])
def test_auto_k_moves_at_fine_grids(bits):
    model = make_linear()
    x, y = make_data()
    crit = nn.MSELoss()
    opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=bits), beta=0.0,
                        move_frac=0.5, beta1=0.9, k_mode="auto", step_scale=1e-2,
                        rng=torch.Generator().manual_seed(0))
    captured = step_capturing_post_snap(model, opt,
                                        make_closure(opt, model, x, y, crit))
    assert opt.last_mean_k is not None and opt.last_mean_k >= 1.0
    assert post_snap_change(model, captured) > 0.0


def test_num_moved_counts_actual_changes_not_selections():
    for bits in (4, 16, 32):
        model = make_model()
        x, y = make_data()
        crit = nn.MSELoss()
        opt = GDMCOptimizer(model.parameters(), grid=UniformGrid(bits=bits),
                            beta=0.0, move_frac=0.5,
                            rng=torch.Generator().manual_seed(0))
        seen = {}
        orig_apply = opt._apply

        def spy_apply(records, _seen=seen, _orig=orig_apply):
            _seen["records"] = records
            _orig(records)

        opt._apply = spy_apply
        opt.step(make_closure(opt, model, x, y, crit))
        records = seen["records"]
        expected_moved = int(sum(int((r.new_values != r.old_values).sum())
                                 for r in records))
        expected_selected = int(sum(int(r.index.numel()) for r in records))
        assert opt.last_num_moved == expected_moved, bits
        assert opt.last_num_selected == expected_selected, bits
        assert opt.last_num_moved <= opt.last_num_selected


def test_rejected_adaptive_move_restores_exact_pre_proposal_weights():
    """Regression: the adaptive range used to be recomputed after the snap, so
    a rejected move re-snapped the weights onto a different grid."""
    grid = AdaptiveGrid(bits=4, alpha=1.05)
    original = torch.tensor([-0.63, -0.21, 0.21, 0.63])
    p = nn.Parameter(original.clone())
    opt = GDMCOptimizer([p], grid=grid, beta=2.0, move_frac=1.0, k=1,
                        rng=torch.Generator().manual_seed(0))

    captured = step_capturing_post_snap(
        [p], opt,
        lambda: (opt.zero_grad(set_to_none=True),
                 setattr(p, "grad", torch.ones_like(p)),
                 torch.tensor(1.0))[2],
        accept=False)

    after_snap = captured[id(p)]
    # The snap legitimately moves the weights onto the adaptive grid ...
    assert not torch.equal(after_snap, original)
    # ... and the rejected move must restore exactly those values.
    assert torch.equal(p.data, after_snap)
    # The old (buggy) behaviour re-snapped on the range derived from the
    # post-snap weights, which is a different grid.
    vmin2, vmax2 = grid.range_for(after_snap)
    wrong = grid.snap_with_range(after_snap, vmin2, vmax2)
    assert not torch.equal(wrong, after_snap)
    assert not torch.equal(p.data, wrong)


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
