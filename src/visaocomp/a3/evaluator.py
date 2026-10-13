"""Métricas por classe, matriz de confusão e baseline; o Evaluator roda o modelo num DataLoader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Predictions:
    """Classes reais, classes previstas, confiança da previsão e a loss média."""

    labels: np.ndarray
    preds: np.ndarray
    confidence: np.ndarray
    loss: float


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """Matriz n x n: linha = classe real, coluna = classe prevista."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (labels, preds), 1)
    return cm


def compute_metrics(
    labels: np.ndarray, preds: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Accuracy global e por classe, precision, F1, matriz de confusão e as trocas mais comuns.

    Uma classe sem imagens no conjunto aparece com `n = 0` e accuracy `None`: não some da lista.
    """
    n = len(class_names)
    cm = confusion_matrix(np.asarray(labels), np.asarray(preds), n)
    support, correct, predicted = cm.sum(axis=1), np.diag(cm), cm.sum(axis=0)
    per_class: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(class_names):
        recall = _ratio(correct[i], support[i])
        precision = _ratio(correct[i], predicted[i])
        f1 = (
            _ratio(2 * precision * recall, precision + recall)
            if precision and recall
            else (0.0 if support[i] else None)
        )
        per_class[name] = {
            "n": int(support[i]),
            "correct": int(correct[i]),
            "accuracy": recall,
            "precision": precision,
            "f1": f1,
        }
    with_support = [v for v in per_class.values() if v["n"]]
    confusions = sorted(
        (
            {"true": class_names[i], "pred": class_names[j], "count": int(cm[i, j])}
            for i in range(n)
            for j in range(n)
            if i != j and cm[i, j]
        ),
        key=lambda d: (-d["count"], d["true"], d["pred"]),
    )
    return {
        "n_samples": int(cm.sum()),
        "accuracy_global": _ratio(correct.sum(), cm.sum()),
        "balanced_accuracy": _ratio(sum(v["accuracy"] for v in with_support), len(with_support)),
        "macro_f1": _ratio(sum(v["f1"] for v in with_support), len(with_support)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": confusions[:5],
    }


def majority_baseline(
    train_labels: list[int], test_labels: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Baseline: prever sempre a classe mais comum do treino. Mostra o piso a superar."""
    counts = np.bincount(np.asarray(train_labels), minlength=len(class_names))
    major = int(counts.argmax())
    acc = _ratio(int((np.asarray(test_labels) == major).sum()), len(test_labels))
    return {"strategy": "classe mais comum do treino", "class": class_names[major], "accuracy": acc}


class Evaluator:
    """Roda um modelo num DataLoader e devolve as previsões."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Predictions:
        """Previsões, confiança e loss média do conjunto inteiro."""
        self.model.eval()
        labels, preds, conf = [], [], []
        total_loss, n = 0.0, 0
        for x, y in loader:
            logits = self.model(x.to(self.device))
            total_loss += nn.functional.cross_entropy(
                logits, y.to(self.device), reduction="sum"
            ).item()
            prob = logits.softmax(dim=1)
            top, idx = prob.max(dim=1)
            labels.append(y.numpy())
            preds.append(idx.cpu().numpy())
            conf.append(top.cpu().numpy())
            n += len(y)
        if n == 0:
            raise ValueError("conjunto vazio: nada para avaliar")
        return Predictions(
            np.concatenate(labels), np.concatenate(preds), np.concatenate(conf), total_loss / n
        )
