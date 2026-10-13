"""Caminhos do projeto: raiz do repositório e pasta de dados."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "VISAOCOMP_DATA_DIR"


def repo_root() -> Path:
    """Raiz do repositório (a pasta que tem o pyproject.toml)."""
    return Path(__file__).resolve().parents[2]


def data_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Pasta de dados: argumento, depois VISAOCOMP_DATA_DIR, depois data.nosync/ na raiz."""
    raw = override or os.environ.get(DATA_DIR_ENV) or "data.nosync"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else repo_root() / path
