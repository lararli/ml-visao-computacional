"""Configuração de log: início e fim de cada etapa, com contagens."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Liga o log no formato `hora nível nome: mensagem` (idempotente)."""
    root = logging.getLogger()
    if any(getattr(h, "_visaocomp", False) for h in root.handlers):
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    handler._visaocomp = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
