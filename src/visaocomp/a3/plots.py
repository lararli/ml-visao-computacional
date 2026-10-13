"""Figuras da A3: curvas de treino e matriz de confusão."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_curves(history: dict[str, Any], path: Path) -> Path:
    """Loss e accuracy por época, treino contra validação, com a melhor época marcada."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], marker="o", label="treino")
        ax.plot(epochs, history[f"val_{key}"], marker="s", label="validação")
        ax.axvline(history["best_epoch"], color="gray", linestyle=":", label="melhor época")
        ax.set_xlabel("época")
        ax.set_ylabel(title)
        ax.set_title(f"{title} por época")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion(cm: list[list[int]], classes: list[str], path: Path) -> Path:
    """Matriz de confusão do teste: linha = classe real, coluna = classe prevista."""
    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe real")
    ax.set_title("Matriz de confusão (teste)")
    top = arr.max() or 1
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                    color="white" if arr[i, j] > top / 2 else "black")  # fmt: skip
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
