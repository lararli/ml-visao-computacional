"""Bloco encoder do Transformer: pré-LayerNorm, atenção, FFN de duas camadas e residual."""

from __future__ import annotations

import torch
from torch import nn

from visaocomp.vit.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Duas camadas lineares com GELU no meio: dim -> hidden -> dim."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, dim) -> (B, N, dim)."""
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class TransformerEncoderBlock(nn.Module):
    """x + MHA(LN(x)), depois x + FFN(LN(x)). Devolve a saída e os pesos de atenção."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, N, dim) -> (B, N, dim) e os pesos (B, H, N, N)."""
        attn_out, weights = self.attn(self.norm1(x), mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights
