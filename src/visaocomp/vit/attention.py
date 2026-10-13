"""Atenção feita à mão: scaled dot-product e multi-head com uma projeção Q, K, V por cabeça."""

from __future__ import annotations

import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_k)) V; devolve a saída e os pesos de atenção."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (..., Nq, d_k), k: (..., Nk, d_k), v: (..., Nk, d_v); mask True = pode olhar."""
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.dropout(weights) @ v, weights


class AttentionHead(nn.Module):
    """Uma cabeça: projeções Q, K e V próprias (dim -> head_dim) e a atenção."""

    def __init__(self, dim: int, head_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q = nn.Linear(dim, head_dim)
        self.k = nn.Linear(dim, head_dim)
        self.v = nn.Linear(dim, head_dim)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, dim) -> saída (B, N, head_dim) e pesos (B, N, N)."""
        return self.attention(self.q(x), self.k(x), self.v(x), mask)


class MultiHeadAttention(nn.Module):
    """Várias cabeças independentes em paralelo, saídas concatenadas e a projeção de saída W_O."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} não divide por {num_heads} cabeças")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.heads = nn.ModuleList(
            AttentionHead(dim, self.head_dim, dropout) for _ in range(num_heads)
        )
        self.out = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def concat_heads(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A concatenação das cabeças, antes de W_O: (B, N, dim), e os pesos (B, H, N, N)."""
        outs, weights = zip(*(head(x, mask) for head in self.heads), strict=True)
        return torch.cat(outs, dim=-1), torch.stack(weights, dim=1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, dim) -> (B, N, dim) e os pesos de atenção de cada cabeça (B, H, N, N)."""
        cat, weights = self.concat_heads(x, mask)
        return self.out_dropout(self.out(cat)), weights
