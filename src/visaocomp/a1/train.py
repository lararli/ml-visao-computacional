"""Treino multirrótulo do ViT (piscina e solar), métricas por atributo e mapas de atenção."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from visaocomp.a1.chips import augment
from visaocomp.a1.scratch_config import ScratchConfig
from visaocomp.vit import VisionTransformer

log = logging.getLogger(__name__)


# -- modelo com o pré-processamento ---------------------------------------------------------------
class ChipClassifier(nn.Module):
    """O ViT com a normalização do treino dentro: entra RGB em [0, 1], sai um logit por atributo."""

    def __init__(self, vit: VisionTransformer, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.vit = vit
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) em [0, 1] -> logits (B, A)."""
        return self.vit((x - self.mean) / self.std)

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Logits e os pesos de atenção de cada bloco."""
        return self.vit.forward_with_attention((x - self.mean) / self.std)


def channel_stats(images: torch.Tensor) -> tuple[list[float], list[float]]:
    """Média e desvio por canal (imagens uint8, só do treino)."""
    x = images.float() / 255.0
    return x.mean(dim=(0, 2, 3)).tolist(), x.std(dim=(0, 2, 3)).tolist()


# -- perda ----------------------------------------------------------------------------------------
class MaskedBCELoss(nn.Module):
    """BCE por atributo; rótulo vazio (NaN) fica fora; peso nos positivos contra o desequilíbrio."""

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Média sobre os rótulos válidos; sem nenhum válido, devolve 0 (sem NaN)."""
        mask = ~torch.isnan(targets)
        filled = torch.where(mask, targets, torch.zeros_like(targets))
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, filled, pos_weight=self.pos_weight, reduction="none"
        )
        n = mask.sum()
        if n == 0:
            return logits.sum() * 0.0
        return (loss * mask).sum() / n


def positive_weights(labels: np.ndarray) -> np.ndarray:
    """Negativos / positivos por atributo, contando só os rótulos válidos."""
    pos = np.nansum(labels == 1, axis=0).astype(np.float64)
    neg = np.nansum(labels == 0, axis=0).astype(np.float64)
    return (neg / np.maximum(pos, 1.0)).astype(np.float32)


# -- métricas -------------------------------------------------------------------------------------
def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Limiar que maximiza o F1 (rótulos vazios já removidos). Sem positivo: 0,5."""
    from sklearn.metrics import precision_recall_curve

    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))])


def attribute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    """precision, recall, F1 e PR-AUC de um atributo; NaN em y = vazio, fica fora."""
    from sklearn.metrics import average_precision_score

    keep = ~np.isnan(y)
    y, p = y[keep].astype(int), p[keep]
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    has_both = 0 < y.sum() < len(y)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(y, p)) if has_both else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def multilabel_metrics(
    labels: np.ndarray, probs: np.ndarray, attributes: tuple[str, ...], thresholds: list[float]
) -> dict[str, dict[str, Any]]:
    """Métricas de cada atributo, cada um com o seu limiar."""
    return {
        a: attribute_metrics(labels[:, j], probs[:, j], thresholds[j])
        for j, a in enumerate(attributes)
    }


def pr_auc_per_attribute(labels: np.ndarray, probs: np.ndarray) -> list[float | None]:
    """PR-AUC de cada atributo (None quando não há positivo e negativo)."""
    return [
        attribute_metrics(labels[:, j], probs[:, j], 0.5)["pr_auc"] for j in range(labels.shape[1])
    ]


# -- treino ---------------------------------------------------------------------------------------
@dataclass
class History:
    """Perda de treino e validação, PR-AUC de validação por atributo, lr e tempo, por época."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: dict[str, list[float | None]] = field(default_factory=dict)
    lr: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = 0  # 1 = primeira época
    best_val_pr_auc_mean: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Dicionário simples para o metrics.json."""
        return asdict(self)


@dataclass
class Prediction:
    """Probabilidades (N, A) e, se pedido, a linha do CLS por cabeça na camada escolhida."""

    probs: np.ndarray
    loss: float
    cls_attention: np.ndarray | None = None  # (N, H, grid, grid)


class MultiLabelTrainer:
    """AdamW com aquecimento e cosseno; guarda a época de maior PR-AUC médio na validação."""

    def __init__(
        self,
        model: nn.Module,
        cfg: ScratchConfig,
        device: torch.device,
        pos_weight: np.ndarray | None = None,
        checkpoint: Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.checkpoint = checkpoint
        pw = torch.tensor(pos_weight, device=device) if pos_weight is not None else None
        self.loss_fn = MaskedBCELoss(pw).to(device)
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
        no_decay = [
            p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
        )
        self.gen = torch.Generator().manual_seed(cfg.seed)

    def lr_at(self, step: int, total: int, warmup: int) -> float:
        """lr do passo: sobe linear no aquecimento, depois cai em cosseno até 0."""
        if step < warmup:
            return self.cfg.lr * (step + 1) / warmup
        frac = (step - warmup) / max(1, total - warmup)
        return self.cfg.lr * 0.5 * (1 + math.cos(math.pi * frac))

    def _batches(self, n: int, shuffle: bool) -> list[torch.Tensor]:
        idx = torch.randperm(n, generator=self.gen) if shuffle else torch.arange(n)
        return list(idx.split(self.cfg.batch_size))

    def _to_input(self, images: torch.Tensor, idx: torch.Tensor, train: bool) -> torch.Tensor:
        x = images[idx.to(images.device)].to(self.device).float() / 255.0
        return augment(x, self.gen) if train and self.cfg.augment else x

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: np.ndarray,
        val_x: torch.Tensor,
        val_y: np.ndarray,
    ) -> History:
        """Treina as épocas do config e restaura a melhor (grava o checkpoint)."""
        hist = History(val_pr_auc={a: [] for a in self.cfg.attributes})
        train_x, val_x = train_x.to(self.device), val_x.to(self.device)  # uint8, cabe na GPU
        ty = torch.tensor(train_y)
        steps_per_epoch = math.ceil(len(train_x) / self.cfg.batch_size)
        total = steps_per_epoch * self.cfg.epochs
        warmup = steps_per_epoch * self.cfg.warmup_epochs
        best_state: dict[str, torch.Tensor] = {}
        best = -1.0
        step = 0
        log.info(
            "treino: início, %d épocas, %d recortes, lote %d, dispositivo %s",
            self.cfg.epochs, len(train_x), self.cfg.batch_size, self.device,
        )  # fmt: skip
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            self.model.train()
            total_loss, n_seen = 0.0, 0
            for idx in self._batches(len(train_x), shuffle=True):
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr_at(step, total, warmup)
                x = self._to_input(train_x, idx, train=True)
                loss = self.loss_fn(self.model(x), ty[idx].to(self.device))
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
                total_loss += loss.item() * len(idx)
                n_seen += len(idx)
                step += 1
            val = self.predict(val_x, val_y)
            aucs = pr_auc_per_attribute(val_y, val.probs)
            valid = [a for a in aucs if a is not None]
            mean_auc = float(np.mean(valid)) if valid else 0.0
            hist.train_loss.append(total_loss / max(n_seen, 1))
            hist.val_loss.append(val.loss)
            for a, v in zip(self.cfg.attributes, aucs, strict=True):
                hist.val_pr_auc[a].append(v)
            hist.lr.append(self.optimizer.param_groups[0]["lr"])
            hist.epoch_seconds.append(time.perf_counter() - t0)
            if mean_auc > best:
                best, hist.best_epoch = mean_auc, epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            log.info(
                "época %d/%d: loss %.4f/%.4f (treino/validação), PR-AUC val %s, %.1fs",
                epoch, self.cfg.epochs, hist.train_loss[-1], val.loss,
                " ".join(f"{a}={v:.3f}" if v is not None else f"{a}=-"
                         for a, v in zip(self.cfg.attributes, aucs, strict=True)),
                hist.epoch_seconds[-1],
            )  # fmt: skip
        hist.best_val_pr_auc_mean = best
        self.model.load_state_dict(best_state)
        if self.checkpoint is not None:
            torch.save(best_state, self.checkpoint)
        log.info(
            "treino: fim, melhor época %d (PR-AUC médio de validação %.4f)", hist.best_epoch, best
        )
        return hist

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        labels: np.ndarray | None = None,
        attention_layer: int | None = None,
    ) -> Prediction:
        """Probabilidades sem augmentation; com `attention_layer`, guarda a linha do CLS."""
        self.model.eval()
        probs, maps = [], []
        total_loss, n_valid = 0.0, 0
        for idx in self._batches(len(images), shuffle=False):
            x = self._to_input(images, idx, train=False)
            if attention_layer is None:
                logits = self.model(x)
            else:
                logits, weights = self.model.forward_with_attention(x)
                grid = self.model.vit.cfg.grid
                cls = weights[attention_layer][:, :, 0, 1:]
                maps.append(cls.reshape(cls.size(0), cls.size(1), grid, grid).cpu().numpy())
            if labels is not None:
                y = torch.tensor(labels[idx.numpy()], device=self.device)
                k = int((~torch.isnan(y)).sum())
                total_loss += self.loss_fn(logits, y).item() * k
                n_valid += k
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return Prediction(
            probs=np.concatenate(probs),
            loss=total_loss / max(n_valid, 1),
            cls_attention=np.concatenate(maps) if maps else None,
        )


# -- mapas de atenção -----------------------------------------------------------------------------
class AttentionMapper:
    """Mapas da linha do CLS (quanto o CLS olha para cada patch), por cabeça, sobre a imagem."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def upsample(self, maps: np.ndarray) -> np.ndarray:
        """(..., g, g) -> (..., size, size), interpolação bilinear."""
        t = torch.from_numpy(np.ascontiguousarray(maps, dtype=np.float32))
        lead = t.shape[:-2]
        t = t.reshape(-1, 1, *t.shape[-2:])
        up = nn.functional.interpolate(
            t, size=(self.image_size,) * 2, mode="bilinear", align_corners=False
        )
        return up.reshape(*lead, self.image_size, self.image_size).numpy()

    @staticmethod
    def region_lift(maps: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
        """Atenção na região / área da região (1 = igual ao acaso), por cabeça e na média.

        maps: (N, H, g, g), cada mapa soma 1 sobre os patches; masks: (N, g, g) em [0, 1].
        Só entram recortes com região (área > 0).
        """
        area = masks.reshape(len(masks), -1).mean(axis=1)
        keep = area > 0
        if not keep.any():
            return {
                "n": 0,
                "per_head": [],
                "mean_heads": None,
                "attention_share": None,
                "area_share": None,
            }
        m = maps[keep]
        mk = masks[keep][:, None]
        share = (m * mk).sum(axis=(2, 3)) / m.sum(axis=(2, 3))  # (n, H)
        lift = share / area[keep][:, None]
        return {
            "n": int(keep.sum()),
            "per_head": [float(v) for v in lift.mean(axis=0)],
            "mean_heads": float(lift.mean()),
            "attention_share": float(share.mean()),
            "area_share": float(area[keep].mean()),
        }

    def figure(
        self,
        image: np.ndarray,
        maps: np.ndarray,
        title: str,
        path: Path,
        highlight: int | None = None,
    ) -> Path:
        """Recorte, média das cabeças e cada cabeça sobre ele; `highlight` marca a cabeça usada."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        heads = maps.shape[0]
        up = self.upsample(maps)
        rgb = np.transpose(image, (1, 2, 0))
        fig, axes = plt.subplots(1, heads + 2, figsize=(2.1 * (heads + 2), 2.6))
        axes[0].imshow(rgb)
        axes[0].set_title("recorte", fontsize=8)
        panels = [("média das cabeças", up.mean(axis=0))]
        panels += [
            (f"cabeça {h + 1}" + (" (usada)" if h == highlight else ""), up[h])
            for h in range(heads)
        ]
        for ax, (name, m) in zip(axes[1:], panels, strict=True):
            ax.imshow(rgb)
            ax.imshow(m, cmap="inferno", alpha=0.55)
            ax.set_title(name, fontsize=8)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
