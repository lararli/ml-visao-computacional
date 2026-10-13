"""Trainer: um treino, com histórico por época e o melhor checkpoint pela validação."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from visaocomp.a3.config import A3Config
from visaocomp.a3.model import CnnClassifier

log = logging.getLogger(__name__)


@dataclass
class History:
    """Loss e accuracy de treino e validação, por época."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    best_epoch: int = 0  # 1 = primeira época

    def to_dict(self) -> dict[str, Any]:
        """Dicionário simples para o metrics.json."""
        return asdict(self)


class Trainer:
    """Treina só a cabeça (Adam), guarda a melhor época pela accuracy de validação."""

    def __init__(
        self,
        model: CnnClassifier,
        cfg: A3Config,
        device: torch.device,
        checkpoint: Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.checkpoint = checkpoint
        params = [p for p in model.parameters() if p.requires_grad]
        if cfg.optimizer != "adam":
            raise ValueError(f"otimizador não suportado: {cfg.optimizer}")
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def _epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(dim=1) == y).sum().item()
                n += len(y)
        if n == 0:
            raise ValueError("DataLoader vazio")
        return total_loss / n, correct / n

    def fit(self, train: DataLoader, val: DataLoader) -> History:
        """Roda as épocas, restaura a melhor (menor loss de validação desempata) e a grava."""
        hist = History()
        best: tuple[float, float] | None = None
        best_state: dict[str, torch.Tensor] = {}
        log.info(
            "treino: início, %d épocas, lr %g, dispositivo %s",
            self.cfg.epochs,
            self.cfg.lr,
            self.device,
        )
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = self._epoch(train, train=True)
            va_loss, va_acc = self._epoch(val, train=False)
            for lst, v in zip(
                (hist.train_loss, hist.train_acc, hist.val_loss, hist.val_acc, hist.epoch_seconds),
                (tr_loss, tr_acc, va_loss, va_acc, time.perf_counter() - t0),
                strict=True,
            ):
                lst.append(v)
            if best is None or (va_acc, -va_loss) > best:
                best, hist.best_epoch = (va_acc, -va_loss), epoch
                best_state = self.model.trainable_state()
            log.info(
                "época %d/%d: loss %.4f/%.4f, acc %.4f/%.4f (treino/validação), %.1fs",
                epoch, self.cfg.epochs, tr_loss, va_loss, tr_acc, va_acc, hist.epoch_seconds[-1],
            )  # fmt: skip
        self.model.load_state_dict(best_state, strict=False)
        if self.checkpoint is not None:
            torch.save(best_state, self.checkpoint)
        log.info("treino: fim, melhor época %d (acc de validação %.4f)", hist.best_epoch, best[0])
        return hist
