"""Escolha do dispositivo e da seed, iguais em todas as atividades."""

from __future__ import annotations

import random

import numpy as np
import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Devolve `prefer` quando dado; senão cuda, depois mps, depois cpu."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Fixa as seeds de random, numpy e torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
