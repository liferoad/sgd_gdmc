"""Tests for the quantization grids."""
import pytest
import torch

from src.gdmc import AdaptiveGrid, UniformGrid, bits_to_levels, make_grid


def test_bits_to_levels():
    assert bits_to_levels(2) == 4
    assert bits_to_levels(8) == 256
    with pytest.raises(ValueError):
        bits_to_levels(0)


@pytest.mark.parametrize("bits", [2, 3, 4, 8, 16])
def test_snap_lands_on_grid(bits):
    grid = UniformGrid(bits=bits)
    w = torch.randn(1000) * 0.5
    snapped = grid.snap(w)
    delta = (grid.vmax - grid.vmin) / (grid.levels - 1)
    idx = (snapped - grid.vmin) / delta
    # idx is large at high bit-widths, so compare in *value* units.
    assert float((idx - idx.round()).abs().max()) * delta < 1e-6
    assert float(snapped.min()) >= grid.vmin - 1e-6
    assert float(snapped.max()) <= grid.vmax + 1e-6


def test_step_is_one_level_in_the_sign_direction():
    grid = UniformGrid(bits=4)
    delta = (grid.vmax - grid.vmin) / (grid.levels - 1)
    base = grid.snap(torch.tensor([-0.2, -0.2]))
    up = grid.step(base, torch.ones(2), k=1)
    down = grid.step(base, -torch.ones(2), k=1)
    assert torch.allclose(up - base, torch.full((2,), delta), atol=1e-6)
    assert torch.allclose(down - base, torch.full((2,), -delta), atol=1e-6)


def test_step_k_is_k_levels():
    grid = UniformGrid(bits=8)
    delta = (grid.vmax - grid.vmin) / (grid.levels - 1)
    base = grid.snap(torch.tensor([-0.2, -0.2]))
    out = grid.step(base, torch.ones(2), k=5)
    assert torch.allclose(out - base, torch.full((2,), 5 * delta), atol=1e-6)


def test_step_multi_vectorised():
    grid = UniformGrid(bits=8)
    delta = (grid.vmax - grid.vmin) / (grid.levels - 1)
    base = grid.snap(torch.tensor([-0.2, -0.2]))
    k = torch.tensor([1.0, 3.0])
    out = grid.step_multi(base, torch.ones(2), k)
    assert torch.allclose(out - base, torch.tensor([delta, 3 * delta]), atol=1e-6)


def test_adaptive_grid_tracks_range():
    grid = AdaptiveGrid(bits=4)
    w = torch.tensor([0.0, 0.5, -0.25])
    vmin, vmax = grid.range_for(w)
    assert vmin < 0 < vmax
    snapped = grid.snap(w)
    delta = (vmax - vmin) / (grid.levels - 1)
    idx = (snapped - vmin) / delta
    assert float((idx - idx.round()).abs().max()) < 1e-3


def test_make_grid_specs():
    assert make_grid("uniform", 4).levels == 16
    assert make_grid("uniform-pt", 4).per_tensor is True
    assert isinstance(make_grid("adaptive", 4), AdaptiveGrid)
    with pytest.raises(ValueError):
        make_grid("nope", 4)
