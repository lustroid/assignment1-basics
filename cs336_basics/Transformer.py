import math

import torch
from einops import einsum, reduce
from torch import nn


class Linear(nn.Module):
    """Bias-free linear transformation of the input's final dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (..., in_features) to (..., out_features)."""
        # return x @ self.weight.T
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Learned token vectors selected by integer token IDs."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs of shape (...) to vectors of shape (..., embedding_dim)."""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize each token's feature vector, then apply a learned gain."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize (..., d_model) in FP32 and preserve the input dtype."""
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = reduce(x.square(), "... d_model -> ... 1", "mean")
        result = x * torch.rsqrt(mean_square + self.eps) * self.weight.to(torch.float32)
        return result.to(in_dtype)
