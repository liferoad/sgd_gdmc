"""Tests for the latent-weight QAT (straight-through) model."""
import torch
import torch.nn as nn

from src.gdmc import UniformGrid
from src.models.quant import QuantLinearSTE, QuantMLP


def test_forward_uses_quantized_weights():
    torch.manual_seed(0)
    grid = UniformGrid(bits=4)
    layer = QuantLinearSTE(6, 3, grid)
    x = torch.randn(4, 6)
    out = layer(x)
    wq = grid.snap(layer.weight)
    expected = nn.functional.linear(x, wq, layer.bias)
    assert torch.allclose(out, expected, atol=1e-6)


def test_gradients_reach_the_latent_weights():
    torch.manual_seed(0)
    layer = QuantLinearSTE(6, 3, UniformGrid(bits=4))
    x = torch.randn(4, 6)
    layer(x).pow(2).mean().backward()
    assert layer.weight.grad is not None
    assert float(layer.weight.grad.abs().sum()) > 0.0


def test_qat_mlp_trains():
    torch.manual_seed(0)
    model = QuantMLP(in_dim=16, hidden=(16,), out_dim=3, bits=4)
    x = torch.randn(64, 16)
    y = torch.randint(0, 3, (64,))
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        if first is None:
            first = float(loss.item())
        loss.backward()
        opt.step()
    assert float(crit(model(x), y).item()) < first
