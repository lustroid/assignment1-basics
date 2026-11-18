import pytest
import torch
from torch.nn import functional as F

from cs336_basics.Transformer import RMSNorm


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
def test_rmsnorm_matches_fp32_reference_and_gradients(dtype, shape):
    torch.manual_seed(42)
    layer = RMSNorm(4, eps=0.01, device=torch.device("cpu"), dtype=dtype)
    assert set(dict(layer.named_parameters())) == {"weight"}
    torch.testing.assert_close(layer.weight, torch.ones(4, dtype=dtype))
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, 2.0, -1.0, 1.5], dtype=dtype))
    x = torch.randn(shape, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = layer.weight.detach().clone().requires_grad_()
    expected = F.rms_norm(reference_x.float(), (4,), reference_weight.float(), eps=0.01).to(dtype)
    actual = layer(x)
    assert actual.dtype == dtype
    assert actual.shape == shape
    torch.testing.assert_close(actual, expected)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(layer.weight.grad, reference_weight.grad)


def test_rmsnorm_fp16_overflow_zero_input_and_token_independence():
    layer = RMSNorm(4)
    x = torch.tensor([[[60000, -60000, 30000, -30000], [0, 0, 0, 0]]], dtype=torch.float16)
    assert not torch.isfinite(x.square()).all()
    actual = layer(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0, 1], torch.zeros(4, dtype=x.dtype))
    expected = F.rms_norm(x.float(), (4,), eps=layer.eps).to(x.dtype)
    torch.testing.assert_close(actual, expected)
    for token in range(x.shape[1]):
        torch.testing.assert_close(actual[0, token], layer(x[0, token]))
