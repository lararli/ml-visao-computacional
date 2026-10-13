"""A1: ViT pré-treinado e ResNet-18 nos mesmos recortes, split e métricas do ViT do zero."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from visaocomp.a1.chips import ChipCatalogReader, check_matrix, label_matrix, load_images
from visaocomp.a1.scratch_config import ScratchConfig
from visaocomp.a1.train import (
    MultiLabelTrainer,
    best_f1_threshold,
    multilabel_metrics,
    positive_weights,
)
from visaocomp.device import seed_everything
from visaocomp.paths import repo_root
from visaocomp.run import RunDir

log = logging.getLogger(__name__)

STAGE = "a1_pretrained"
CONFIG = Path("configs") / "a1_pretrained.yaml"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PretrainedChipModel(nn.Module):
    """Corpo pré-treinado no ImageNet com cabeça nova; amplia o recorte e normaliza dentro."""

    def __init__(self, backbone: nn.Module, input_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) em [0, 1] -> logits (B, A)."""
        if x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
        return self.backbone((x - self.mean) / self.std)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PretrainedVit(PretrainedChipModel):
    """`vit_small_patch16_224` do timm (ImageNet-21k, ajustado no 1k); cabeça trocada."""

    def __init__(self, num_outputs: int, pretrained: bool = True, name: str = "") -> None:
        import timm

        backbone = timm.create_model(
            name or "vit_small_patch16_224", pretrained=pretrained, num_classes=num_outputs
        )
        super().__init__(backbone, 224)

    @property
    def head(self) -> nn.Linear:
        return self.backbone.head


class ResnetBaseline(PretrainedChipModel):
    """ResNet-18 do torchvision (ImageNet-1k); a camada `fc` vira a cabeça de 2 saídas."""

    def __init__(self, num_outputs: int, pretrained: bool = True, name: str = "") -> None:
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_outputs)
        super().__init__(backbone, 224)

    @property
    def head(self) -> nn.Linear:
        return self.backbone.fc


MODELS: dict[str, type[PretrainedChipModel]] = {
    "vit_pretrained": PretrainedVit,
    "resnet18": ResnetBaseline,
}


@dataclass(frozen=True)
class ModelSpec:
    """Um modelo pré-treinado: chave, classe e o treino curto dele."""

    key: str
    label: str
    epochs: int
    batch_size: int
    lr: float
    warmup_epochs: int

    def train_cfg(self, base: ScratchConfig) -> ScratchConfig:
        return base.with_overrides(
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            warmup_epochs=self.warmup_epochs,
        )


def load_specs(path: Path | None = None) -> tuple[ScratchConfig, list[ModelSpec]]:
    """Lê `configs/a1_pretrained.yaml`; o resto do treino é o do ViT do zero (mesmo split)."""
    raw = yaml.safe_load((path or repo_root() / CONFIG).read_text(encoding="utf-8"))
    base = ScratchConfig.from_yaml()
    specs = [ModelSpec(key=k, **v) for k, v in raw["models"].items()]
    return base, specs


def file_hash(path: Path) -> str:
    """Primeiros 12 caracteres do sha256 do checkpoint: entra no `model_version`."""
    side = Path(str(path) + ".sha256")
    if not Path(path).exists() and side.exists():  # clone sem os pesos: o hash vem gravado ao lado
        return side.read_text(encoding="utf-8").strip()[:12]
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


class PretrainedPipeline:
    """Treina cada modelo do config, escolhe limiar na validação e avalia no teste uma vez."""

    def __init__(
        self,
        base: ScratchConfig,
        specs: list[ModelSpec],
        catalog_path: Path,
        data_root: Path,
        outputs_root: Path,
        device: torch.device,
        pretrained: bool = True,
    ) -> None:
        self.base = base
        self.specs = specs
        self.catalog_path = Path(catalog_path)
        self.data_root = Path(data_root)
        self.outputs_root = Path(outputs_root)
        self.device = device
        self.pretrained = pretrained

    def run(self) -> Path:
        """Executa o treino dos pré-treinados e devolve o caminho do manifest.json."""
        t_start = time.perf_counter()
        attrs = self.base.attributes
        chips, _ = ChipCatalogReader(self.catalog_path, attrs).read()
        images = load_images(list(chips["path"]), self.data_root, self.base.image_size)
        labels = label_matrix(chips, attrs)
        checks = check_matrix(chips, attrs)
        idx = {s: np.flatnonzero(chips["split"].to_numpy() == s) for s in ("train", "val", "test")}
        test = chips.iloc[idx["test"]].reset_index(drop=True)
        reviewed = test[[f"{a}_check" for a in attrs]].notna().any(axis=1).to_numpy()
        t_lab, t_chk = labels[idx["test"]], checks[idx["test"]]
        pos_weight = positive_weights(labels[idx["train"]]) if self.base.pos_weight else None

        run = RunDir(self.outputs_root, STAGE)
        out_dir = run.path / STAGE
        out_dir.mkdir()
        models: dict[str, Any] = {}
        pred_rows: list[pd.DataFrame] = []
        for spec in self.specs:
            seed_everything(self.base.seed)
            cfg = spec.train_cfg(self.base)
            model = MODELS[spec.key](len(attrs), pretrained=self.pretrained)
            ckpt = out_dir / f"{spec.key}.pt"
            trainer = MultiLabelTrainer(model, cfg, self.device, pos_weight, ckpt)
            t0 = time.perf_counter()
            hist = trainer.fit(
                images[idx["train"]], labels[idx["train"]], images[idx["val"]], labels[idx["val"]]
            )
            seconds = time.perf_counter() - t0
            val = trainer.predict(images[idx["val"]], labels[idx["val"]])
            thresholds = [
                best_f1_threshold(
                    labels[idx["val"], j][~np.isnan(labels[idx["val"], j])],
                    val.probs[~np.isnan(labels[idx["val"], j]), j],
                )
                for j in range(len(attrs))
            ]
            pred = trainer.predict(images[idx["test"]], t_lab)
            version = f"{spec.key}-{file_hash(ckpt)}"
            models[spec.key] = {
                "label": spec.label,
                "model_version": version,
                "params": model.param_count(),
                "head": {"in": model.head.in_features, "out": model.head.out_features},
                "train": {
                    **cfg.to_dict(),
                    "seconds": seconds,
                    "best_epoch": hist.best_epoch,
                    "best_val_pr_auc_mean": hist.best_val_pr_auc_mean,
                    "device": str(self.device),
                },
                "thresholds": dict(zip(attrs, thresholds, strict=True)),
                "test": {
                    "n_chips": int(len(test)),
                    "per_attribute": multilabel_metrics(t_lab, pred.probs, attrs, thresholds),
                },
                "test_reviewed": {
                    "n_chips": int(reviewed.sum()),
                    "per_attribute": multilabel_metrics(
                        t_chk[reviewed], pred.probs[reviewed], attrs, thresholds
                    ),
                },
                "history": hist.to_dict(),
            }
            for split, probs in (("val", val.probs), ("test", pred.probs)):
                part = chips.iloc[idx[split]][["chip_id", "house_id"]].reset_index(drop=True)
                part["split"], part["model"], part["model_version"] = split, spec.key, version
                for j, a in enumerate(attrs):
                    part[f"prob_{a}"] = probs[:, j]
                pred_rows.append(part)
            log.info("%s: fim em %.0f s, versão %s", spec.key, seconds, version)

        run.write_json(
            f"{STAGE}/metrics.json",
            {"run_id": run.run_id, "attributes": list(attrs), "models": models},
        )
        pd.concat(pred_rows).to_csv(out_dir / "predictions.csv", index=False)
        return run.write_manifest(
            [
                {
                    "name": "catalog.duckdb",
                    "path": str(self.catalog_path),
                    "tables": ["silver.chip", "silver.label", "silver.house"],
                }
            ],
            {"recortes": int(len(chips)), **{s: int(len(i)) for s, i in idx.items()}},
            extra={
                "device": str(self.device),
                "timings_seconds": {"total": round(time.perf_counter() - t_start, 1)},
            },
        )
