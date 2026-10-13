"""PatchEmbedding e VisionTransformer: imagem -> patches -> CLS + posição -> encoder -> logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from visaocomp.vit.encoder import TransformerEncoderBlock


@dataclass(frozen=True)
class ViTConfig:
    """Formato do ViT: imagem, patch, dimensão, blocos, cabeças e saídas."""

    image_size: int = 96
    patch: int = 8
    in_channels: int = 3
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_outputs: int = 2
    use_pos: bool = True

    @property
    def grid(self) -> int:
        """Patches por lado (96 / 8 = 12)."""
        return self.image_size // self.patch

    @property
    def num_patches(self) -> int:
        """Número de patches (12 x 12 = 144)."""
        return self.grid**2

    def to_dict(self) -> dict:
        """Dicionário simples para o metrics.json."""
        return asdict(self)


class PatchEmbedding(nn.Module):
    """Convolução com kernel e stride do tamanho do patch: cada patch vira um vetor de `dim`."""

    def __init__(self, image_size: int, patch: int, in_channels: int, dim: int) -> None:
        super().__init__()
        if image_size % patch != 0:
            raise ValueError(f"imagem {image_size} não divide em patches de {patch}")
        self.patch = patch
        self.num_patches = (image_size // patch) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, dim), patches em ordem de leitura (linha a linha)."""
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """ViT do zero: patches, CLS aprendível, posição aprendível, encoder e cabeça multirrótulo."""

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg.image_size, cfg.patch, cfg.in_channels, cfg.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, cfg.dim))
        self.pos_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_outputs)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Imagem -> sequência de entrada do encoder: [CLS, patch_1..patch_N] (+ posição)."""
        patches = self.patch_embed(x)
        cls = self.cls_token.expand(patches.size(0), -1, -1)
        seq = torch.cat([cls, patches], dim=1)
        if self.cfg.use_pos:
            seq = seq + self.pos_embed
        return self.pos_drop(seq)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Imagem -> todos os tokens de saída (B, N+1, dim) e os pesos de cada bloco."""
        seq = self.tokens(x)
        weights = []
        for block in self.blocks:
            seq, w = block(seq)
            weights.append(w)
        return self.norm(seq), weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> logits (B, num_outputs), lidos do token CLS."""
        seq, _ = self.encode(x)
        return self.head(seq[:, 0])

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Logits e os pesos de atenção de cada bloco, cada um (B, H, N+1, N+1)."""
        seq, weights = self.encode(x)
        return self.head(seq[:, 0]), weights

    def param_count(self) -> int:
        """Número de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def shuffle_patches(images: torch.Tensor, patch: int, perm: torch.Tensor) -> torch.Tensor:
    """Embaralha os patches da imagem na ordem `perm` (patch i da saída = patch perm[i])."""
    b, c, h, w = images.shape
    g_h, g_w = h // patch, w // patch
    blocks = images.reshape(b, c, g_h, patch, g_w, patch).permute(0, 2, 4, 1, 3, 5)
    blocks = blocks.reshape(b, g_h * g_w, c, patch, patch)[:, perm]
    blocks = blocks.reshape(b, g_h, g_w, c, patch, patch).permute(0, 3, 1, 4, 2, 5)
    return blocks.reshape(b, c, h, w)
