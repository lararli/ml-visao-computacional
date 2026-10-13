"""Configuração do ViT do zero, lida de configs/a1_scratch.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from visaocomp.paths import repo_root
from visaocomp.vit import ViTConfig


@dataclass(frozen=True)
class ScratchConfig:
    """Dados, modelo, treino e avaliação do ViT do zero."""

    attributes: tuple[str, ...] = ("pool", "solar")
    image_size: int = 96
    pixel_m: float = 0.6
    patch: int = 8
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    epochs: int = 75
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    augment: bool = True
    pos_weight: bool = True
    examples: tuple[str, ...] = field(default=("pool", "solar", "erro"))

    def vit(self) -> ViTConfig:
        """O ViTConfig correspondente (2 saídas: uma por atributo)."""
        return ViTConfig(
            image_size=self.image_size,
            patch=self.patch,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            num_outputs=len(self.attributes),
        )

    def with_overrides(self, **kwargs: Any) -> ScratchConfig:
        """Cópia com alguns campos trocados (épocas pela linha de comando, testes)."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Dicionário simples para o metrics.json e o manifesto."""
        out = asdict(self)
        out["attributes"] = list(self.attributes)
        out["examples"] = list(self.examples)
        return out

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> ScratchConfig:
        """Lê configs/a1_scratch.yaml (ou outro arquivo)."""
        path = Path(path) if path else repo_root() / "configs" / "a1_scratch.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        image, model, train = doc.get("image", {}), doc.get("model", {}), doc.get("train", {})
        cfg = cls(
            attributes=tuple(doc.get("attributes", cls.attributes)),
            image_size=int(image.get("size", cls.image_size)),
            pixel_m=float(image.get("pixel_m", cls.pixel_m)),
            patch=int(model.get("patch", cls.patch)),
            dim=int(model.get("dim", cls.dim)),
            depth=int(model.get("depth", cls.depth)),
            heads=int(model.get("heads", cls.heads)),
            mlp_ratio=float(model.get("mlp_ratio", cls.mlp_ratio)),
            dropout=float(model.get("dropout", cls.dropout)),
            epochs=int(train.get("epochs", cls.epochs)),
            batch_size=int(train.get("batch_size", cls.batch_size)),
            lr=float(train.get("lr", cls.lr)),
            weight_decay=float(train.get("weight_decay", cls.weight_decay)),
            warmup_epochs=int(train.get("warmup_epochs", cls.warmup_epochs)),
            grad_clip=float(train.get("grad_clip", cls.grad_clip)),
            seed=int(train.get("seed", cls.seed)),
            augment=bool(train.get("augment", cls.augment)),
            pos_weight=bool(train.get("pos_weight", cls.pos_weight)),
            examples=tuple(doc.get("attention", {}).get("examples", cls.examples)),
        )
        if cfg.image_size % cfg.patch:
            raise ValueError(f"imagem {cfg.image_size} não divide em patches de {cfg.patch}")
        return cfg
