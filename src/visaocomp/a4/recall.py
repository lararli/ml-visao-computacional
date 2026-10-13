"""Classificador de raio-X e o experimento do recall da COVID com e sem imagens sintéticas."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sstats  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from visaocomp.a4.config import ClassifierConfig  # noqa: E402
from visaocomp.a4.fid import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

log = logging.getLogger(__name__)


class GpuAugment:
    """Augmentation leve por amostra, no dispositivo: rotação, escala, translação, brilho."""

    def __init__(
        self,
        rotation_deg: float = 10,
        scale: float = 0.1,
        translate: float = 0.05,
        brightness: float = 0.1,
        contrast: float = 0.1,
    ) -> None:
        self.rotation = math.radians(rotation_deg)
        self.scale = scale
        self.translate = translate
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """x [N, C, H, W] em [0, 1] -> mesma forma, transformada."""
        n = x.shape[0]

        def u(lo: float, hi: float) -> torch.Tensor:
            return (torch.rand(n, generator=gen) * (hi - lo) + lo).to(x.device)

        ang = u(-self.rotation, self.rotation)
        sc = u(1 - self.scale, 1 + self.scale)
        tx, ty = u(-self.translate, self.translate) * 2, u(-self.translate, self.translate) * 2
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack(
            [torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1
        ).to(x.dtype)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        b = u(-self.brightness, self.brightness).view(n, 1, 1, 1)
        c = u(1 - self.contrast, 1 + self.contrast).view(n, 1, 1, 1)
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        return ((out - mean) * c + mean + b).clamp(0, 1)


class XrayClassifier(nn.Module):
    """ResNet-18 com o pré-processamento junto: cinza em [0, 1] -> 3 canais normalizados."""

    def __init__(self, n_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [N, 1, H, W] em [0, 1] -> logits [N, classes]."""
        x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        return self.net((x - self.mean) / self.std)


def class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Peso por classe = N / (K * n_c): a classe rara pesa mais na perda."""
    counts = torch.bincount(labels.cpu(), minlength=n_classes).float()
    if (counts == 0).any():
        raise ValueError(f"classe sem exemplo no treino: contagens {counts.tolist()}")
    return counts.sum() / (n_classes * counts)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Matriz de confusão: linha = verdade, coluna = previsão."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics(
    cm: np.ndarray, classes: tuple[str, ...], positive: int, prevalence: np.ndarray
) -> dict[str, float]:
    """Recall e precisão por classe, accuracy, recall médio e precisão da COVID na prevalência."""
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    recall = np.divide(np.diag(cm), rows, out=np.zeros(len(classes)), where=rows > 0)
    precision = np.divide(np.diag(cm), cols, out=np.zeros(len(classes)), where=cols > 0)
    rate = cm / np.maximum(rows[:, None], 1)  # P(previsto | verdade)
    denom = float((rate[:, positive] * prevalence).sum())
    ppv_prev = float(rate[positive, positive] * prevalence[positive] / denom) if denom > 0 else 0.0
    out: dict[str, float] = {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "balanced_accuracy": float(recall.mean()),
        "ppv_covid_prevalencia_caso": ppv_prev,
    }
    for i, c in enumerate(classes):
        out[f"recall_{c.lower()}"] = float(recall[i])
        out[f"precision_{c.lower()}"] = float(precision[i])
    return out


def mean_ci(values: list[float] | np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Média, desvio, mínimo, máximo e intervalo t de Student (n pequeno)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(sstats.t.ppf((1 + level) / 2, n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": float(v.min()),
        "max": float(v.max()),
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


@dataclass
class TrainResult:
    """Um treino do classificador: o modelo da melhor época e as curvas de validação."""

    model: XrayClassifier
    best_epoch: int
    curves: list[dict[str, float]]
    seconds: float


class ClassifierTrainer:
    """Treina a ResNet-18 com class weights e augmentation leve; escolhe a época pela validação."""

    def __init__(
        self,
        cfg: ClassifierConfig,
        classes: tuple[str, ...],
        positive: int,
        prevalence: np.ndarray,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.classes = classes
        self.positive = positive
        self.prevalence = prevalence
        self.device = device
        self.augment = GpuAugment(**cfg.augment)

    @torch.no_grad()
    def predict(self, model: nn.Module, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Classe prevista e probabilidades. images uint8 [N, 1, H, W]."""
        model.eval()
        probs = []
        for i in range(0, len(images), 128):
            x = images[i : i + 128].to(self.device).float() / 255.0
            probs.append(torch.softmax(model(x), dim=1).float().cpu())
        p = torch.cat(probs).numpy()
        return p.argmax(axis=1), p

    def evaluate(
        self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, Any]:
        """Métricas e matriz de confusão num conjunto."""
        pred, probs = self.predict(model, images)
        cm = confusion(labels.numpy(), pred, len(self.classes))
        return {
            **classification_metrics(cm, self.classes, self.positive, self.prevalence),
            "confusion": cm.tolist(),
            "pred": pred,
            "probs": probs,
        }

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        seed: int,
    ) -> TrainResult:
        """Treina por `epochs`; guarda a época de maior recall médio na validação."""
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        model = XrayClassifier(len(self.classes), self.cfg.pretrained).to(self.device)
        weights = class_weights(train_y, len(self.classes)).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        steps = self.cfg.epochs * math.ceil(len(train_x) / self.cfg.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.cfg.lr, total_steps=steps)
        best, best_epoch, best_state, curves = -1.0, 0, None, []
        start = time.perf_counter()
        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            perm = torch.randperm(len(train_x), generator=gen)
            losses = []
            for i in range(0, len(train_x), self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                x = train_x[idx].to(self.device).float() / 255.0
                y = train_y[idx].to(self.device)
                loss = loss_fn(model(self.augment(x, gen)), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
            val = self.evaluate(model, val_x, val_y)
            pos = self.classes[self.positive].lower()
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_balanced_accuracy": val["balanced_accuracy"],
                    f"val_recall_{pos}": val[f"recall_{pos}"],
                }
            )
            if val["balanced_accuracy"] >= best:
                best, best_epoch = val["balanced_accuracy"], epoch
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
        assert best_state is not None
        model.load_state_dict(best_state)
        return TrainResult(model, best_epoch, curves, time.perf_counter() - start)


@dataclass
class Condition:
    """Uma condição de treino: as imagens e rótulos que entram no classificador."""

    name: str
    images: torch.Tensor
    labels: torch.Tensor
    n_synthetic: int = 0


class RecallExperiment:
    """Mesmo classificador, mesmo teste real, várias seeds: recall da COVID com e sem sintético."""

    def __init__(
        self,
        trainer: ClassifierTrainer,
        seeds: tuple[int, ...],
        out_dir: Path,
    ) -> None:
        self.trainer = trainer
        self.seeds = seeds
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        conditions: list[Condition],
        val: tuple[torch.Tensor, torch.Tensor],
        test: tuple[torch.Tensor, torch.Tensor],
        test_paths: list[str],
    ) -> pd.DataFrame:
        """Treina cada condição em cada seed, avalia no teste real e grava recall.csv."""
        tr = self.trainer
        pos = tr.classes[tr.positive].lower()
        rows, confusions, curves, preds = [], {}, {}, []
        for cond in conditions:
            for seed in self.seeds:
                res = tr.fit(cond.images, cond.labels, val[0], val[1], seed)
                ev = tr.evaluate(res.model, test[0], test[1])
                key = f"{cond.name}/seed{seed}"
                confusions[key] = ev["confusion"]
                curves[key] = res.curves
                n_pos = int((cond.labels == tr.positive).sum())
                rows.append(
                    {
                        "condition": cond.name,
                        "seed": seed,
                        f"recall_{pos}": ev[f"recall_{pos}"],
                        **{
                            k: v
                            for k, v in ev.items()
                            if k not in ("confusion", "pred", "probs", f"recall_{pos}")
                        },
                        "best_epoch": res.best_epoch,
                        "n_train": len(cond.labels),
                        f"n_train_{pos}": n_pos,
                        "n_synthetic": cond.n_synthetic,
                        "n_test": len(test[1]),
                        f"n_test_{pos}": int((test[1] == tr.positive).sum()),
                        "test_real_only": True,
                        "train_seconds": round(res.seconds, 1),
                    }
                )
                for path, t, p, pr in zip(
                    test_paths, test[1].tolist(), ev["pred"], ev["probs"], strict=True
                ):
                    preds.append(
                        {
                            "condition": cond.name,
                            "seed": seed,
                            "path": path,
                            "true": t,
                            "pred": int(p),
                            f"prob_{pos}": float(pr[tr.positive]),
                        }
                    )
                log.info(
                    "classificador %s seed %d: recall %s %.3f, bal. acc %.3f (época %d, %.0f s)",
                    cond.name,
                    seed,
                    pos,
                    ev[f"recall_{pos}"],
                    ev["balanced_accuracy"],
                    res.best_epoch,
                    res.seconds,
                )
                del res
        df = pd.DataFrame(rows)
        df.to_csv(self.out_dir / "recall.csv", index=False)
        pd.DataFrame(preds).to_csv(self.out_dir / "test_predictions.csv", index=False)
        (self.out_dir / "confusion.json").write_text(
            json.dumps(confusions, indent=2), encoding="utf-8"
        )
        (self.out_dir / "classifier_curves.json").write_text(
            json.dumps(curves, indent=2), encoding="utf-8"
        )
        summary = self.summarize(df, [c.name for c in conditions], f"recall_{pos}")
        (self.out_dir / "recall_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.plot(df, curves, f"recall_{pos}")
        return df

    @staticmethod
    def summarize(df: pd.DataFrame, conditions: list[str], metric: str) -> dict[str, Any]:
        """Média e intervalo por condição; diferença pareada por seed (última menos primeira)."""
        out: dict[str, Any] = {"metric": metric, "conditions": {}}
        for cond in conditions:
            sub = df[df["condition"] == cond].sort_values("seed")
            out["conditions"][cond] = {
                m: mean_ci(sub[m].to_numpy())
                for m in (metric, "balanced_accuracy", "accuracy", "ppv_covid_prevalencia_caso")
            }
        if len(conditions) == 2:
            a = df[df["condition"] == conditions[0]].sort_values("seed")
            b = df[df["condition"] == conditions[1]].sort_values("seed")
            diff = b[metric].to_numpy() - a[metric].to_numpy()
            ci = mean_ci(diff)
            if ci["ci_low"] > 0:
                verdict = "melhora"
            elif ci["ci_high"] < 0:
                verdict = "piora"
            else:
                verdict = "inconclusivo"
            out["paired_difference"] = {
                "of": f"{conditions[1]} - {conditions[0]}",
                "per_seed": diff.tolist(),
                **ci,
                "verdict": verdict,
            }
        return out

    def plot(
        self, df: pd.DataFrame, curves: dict[str, list[dict[str, float]]], metric: str
    ) -> None:
        """Figura do recall por seed e curvas de validação."""
        conds = list(dict.fromkeys(df["condition"]))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for i, cond in enumerate(conds):
            vals = df[df["condition"] == cond][metric].to_numpy()
            ax[0].scatter(np.full(len(vals), i), vals, s=40, zorder=3)
            ax[0].errorbar(
                i,
                vals.mean(),
                yerr=vals.std(ddof=1) if len(vals) > 1 else 0,
                fmt="_",
                ms=30,
                color="black",
                capsize=6,
            )
        ax[0].set_xticks(range(len(conds)), conds)
        ax[0].set(title=f"{metric} no teste real (cada ponto = 1 seed)", ylim=(0, 1.02))
        for key, c in curves.items():
            ax[1].plot(
                [r["epoch"] for r in c],
                [r["val_balanced_accuracy"] for r in c],
                label=key,
                alpha=0.8,
            )
        ax[1].set(title="Validação: recall médio por época", xlabel="época")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self.out_dir / "recall.png", dpi=110)
        plt.close(fig)
