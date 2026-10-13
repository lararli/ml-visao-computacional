"""Configuração da A3, lida de configs/a3.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from visaocomp.paths import repo_root


@dataclass(frozen=True)
class A3Config:
    """Todos os parâmetros da A3: dados, modelo, split e treino."""

    source: str = "images_dataset"
    subdir: str = "data"
    extensions: tuple[str, ...] = (".bmp", ".jpg", ".jpeg", ".png")
    dedupe: bool = True
    arch: str = "efficientnet_b0"
    weights: str | None = "IMAGENET1K_V1"
    dropout: float = 0.2
    resize: int = 256
    crop: int = 224
    train_frac: float = 0.70
    val_frac: float = 0.15
    seed: int = 42
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "adam"
    num_workers: int = 0

    @property
    def test_frac(self) -> float:
        """Fração do teste: o que sobra depois do treino e da validação."""
        return round(1.0 - self.train_frac - self.val_frac, 10)

    def with_overrides(self, **kwargs: Any) -> A3Config:
        """Cópia com alguns campos trocados (usado nos testes)."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Dicionário simples para gravar no manifesto."""
        out = asdict(self)
        out["extensions"] = list(self.extensions)
        out["test_frac"] = self.test_frac
        return out

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> A3Config:
        """Lê configs/a3.yaml (ou outro arquivo) e valida as frações do split."""
        path = Path(path) if path else repo_root() / "configs" / "a3.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        model, image = doc.get("model", {}), doc.get("image", {})
        split, train = doc.get("split", {}), doc.get("train", {})
        cfg = cls(
            source=doc.get("source", cls.source),
            subdir=doc.get("subdir", cls.subdir),
            extensions=tuple(str(e).lower() for e in doc.get("extensions", cls.extensions)),
            dedupe=bool(doc.get("dedupe", True)),
            arch=model.get("arch", cls.arch),
            weights=model.get("weights", cls.weights),
            dropout=float(model.get("dropout", cls.dropout)),
            resize=int(image.get("resize", cls.resize)),
            crop=int(image.get("crop", cls.crop)),
            train_frac=float(split.get("train", cls.train_frac)),
            val_frac=float(split.get("val", cls.val_frac)),
            seed=int(split.get("seed", cls.seed)),
            epochs=int(train.get("epochs", cls.epochs)),
            batch_size=int(train.get("batch_size", cls.batch_size)),
            lr=float(train.get("lr", cls.lr)),
            optimizer=str(train.get("optimizer", cls.optimizer)),
            num_workers=int(train.get("num_workers", cls.num_workers)),
        )
        if not (0 < cfg.train_frac < 1 and 0 < cfg.val_frac < 1 and cfg.test_frac > 0):
            raise ValueError(
                f"frações do split inválidas em {path}: train={cfg.train_frac}, "
                f"val={cfg.val_frac}, sobra para o teste {cfg.test_frac}"
            )
        return cfg
