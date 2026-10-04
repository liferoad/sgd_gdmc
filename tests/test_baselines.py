"""Tests for the baseline optimizers (including the new low-bit ones)."""
import math

import pytest
import torch
import torch.nn as nn

from src.baselines import Adam8Bit, ProjectedGD, SignSGD, grid_delta
from src.gdmc import UniformGrid


def toy_problem(seed=0, n=64, d=8):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
    x = torch.randn(n, d)
    y = torch.randn(n, 1)
    return model, x, y, nn.MSELoss()


def test_grid_delta_matches_spacing():
    assert abs(grid_delta(UniformGrid(bits=4)) - 2.0 / 15.0) < 1e-12
    assert grid_delta(UniformGrid(bits=4, per_tensor=True)) is None


def test_projected_gd_lr_scale_matches_delta():
    model, _, _, _ = toy_problem()
    grid = UniformGrid(bits=4)
    opt = ProjectedGD(model.parameters(), base="adam", grid=grid, lr_scale=0.5)
    assert abs(opt.lr - 0.5 * grid_delta(grid)) < 1e-12


def test_adam8bit_state_is_about_two_bytes_per_parameter():
    # Use a large tensor so the per-block float32 scales are negligible.
    torch.manual_seed(0)
    model = nn.Linear(1000, 100)
    n = sum(p.numel() for p in model.parameters())
    opt = Adam8Bit(model.parameters(), lr=1e-3)
    x = torch.randn(8, 1000)
    opt.zero_grad(set_to_none=True)
    model(x).pow(2).mean().backward()
    opt.step()
    per_param = opt.state_bytes() / n
    assert 1.99 < per_param < 2.02, per_param


def test_adam8bit_reduces_loss():
    model, x, y, crit = toy_problem(seed=1)
    opt = Adam8Bit(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        if first is None:
            first = float(loss.item())
        loss.backward()
        opt.step()
    last = float(crit(model(x), y).item())
    assert last < first * 0.9, (first, last)


def test_adam8bit_quantization_roundtrip():
    from src.baselines.adam8bit import quantize_blockwise, dequantize_blockwise
    t = torch.randn(5000)
    codes, scale = quantize_blockwise(t, block_size=2048)
    assert codes.dtype == torch.int8
    back = dequantize_blockwise(codes, scale, t.shape, block_size=2048)
    rel = float((back - t).norm() / t.norm())
    assert rel < 0.05, rel


def test_signsgd_steps_are_bounded_by_lr():
    model, x, y, crit = toy_problem(seed=2)
    opt = SignSGD(model.parameters(), lr=1e-2, momentum=0.0)
    before = [p.data.clone() for p in model.parameters()]
    opt.zero_grad(set_to_none=True)
    crit(model(x), y).backward()
    opt.step()
    for p, b in zip(model.parameters(), before):
        assert torch.allclose((p.data - b).abs(),
                              torch.full_like(p.data, 1e-2), atol=1e-6)


def test_signsgd_reduces_loss():
    model, x, y, crit = toy_problem(seed=3)
    opt = SignSGD(model.parameters(), lr=1e-2, momentum=0.9)
    first = None
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        if first is None:
            first = float(loss.item())
        loss.backward()
        opt.step()
    last = float(crit(model(x), y).item())
    assert last < first
