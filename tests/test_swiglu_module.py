import pytest
import torch
from torch.nn import functional as F

from cs336_basics.Transformer import Linear, SwiGLU, silu


@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_swiglu_reference_gradients_and_independent_positions(shape, dtype):
    torch.manual_seed(42)
    layer = SwiGLU(4, d_ff=7, device=torch.device("cpu"), dtype=dtype)
    assert all(isinstance(child, Linear) for child in layer.children())
    assert set(dict(layer.named_parameters())) == {"w1.weight", "w2.weight", "w3.weight"}
    assert layer.d_ff == 7
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = [p.detach().clone().requires_grad_() for p in layer.parameters()]
    w1, w2, w3 = weights
    expected = F.linear(F.silu(F.linear(reference_x, w1)) * F.linear(reference_x, w3), w2)
    actual = layer(x)
    assert actual.shape == x.shape and actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    independent = torch.stack([layer(token) for token in x.reshape(-1, 4)]).reshape(shape)
    torch.testing.assert_close(actual, independent)
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, *layer.parameters()), grad_output)
    expected_grads = torch.autograd.grad(expected, (reference_x, *weights), grad_output)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_swiglu_default_width_and_silu_signed_inputs():
    for width in [16, 32, 64, 128]:
        layer = SwiGLU(width)
        assert layer.d_ff % 64 == 0
        assert 8 * width / 3 <= layer.d_ff < 8 * width / 3 + 64
        assert layer.w1.weight.shape == layer.w3.weight.shape == (layer.d_ff, width)
        assert layer.w2.weight.shape == (width, layer.d_ff)
    x = torch.tensor([-100.0, -3.0, 0.0, 2.0, 100.0], requires_grad=True)
    torch.testing.assert_close(silu(x), F.silu(x))
