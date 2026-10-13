"""Ponto de entrada da A3: lê o config, confere os dados e roda o A3Pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from visaocomp.a3.config import A3Config
from visaocomp.a3.pipeline import A3Pipeline
from visaocomp.config import load_data_config
from visaocomp.data.sources import build_source
from visaocomp.device import pick_device
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir, repo_root

log = logging.getLogger(__name__)


def run_a3(
    cfg: A3Config | None = None,
    outputs_root: Path | None = None,
    section_path: Path | None = None,
    device: str | None = None,
) -> Path:
    """Verifica a fonte pelo DataSource, roda a A3 e devolve o caminho do manifest.json."""
    cfg = cfg or A3Config.from_yaml()
    root = repo_root()
    source = build_source(load_data_config().source(cfg.source), data_dir())
    info = source.describe()
    if not info["verified"]:
        raise FileNotFoundError(f"fonte {cfg.source} não verificada. Rode `make data` antes.")
    pipeline = A3Pipeline(
        cfg,
        source.dest_dir / cfg.subdir,
        info,
        outputs_root or root / "outputs",
        pick_device(device),
        section_path or root / "reports" / "sections" / "a3.md",
    )
    return pipeline.run()


def main(argv: list[str] | None = None) -> int:
    """Linha de comando de `scripts/run_a3.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="outro configs/a3.yaml")
    parser.add_argument("--device", default=None, help="cuda, mps ou cpu (padrão: automático)")
    parser.add_argument("--epochs", type=int, default=None, help="troca as épocas do config")
    parser.add_argument("--batch-size", type=int, default=None, help="troca o lote do config")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = A3Config.from_yaml(args.config)
    overrides = {
        k: v for k, v in (("epochs", args.epochs), ("batch_size", args.batch_size)) if v is not None
    }
    try:
        run_a3(cfg.with_overrides(**overrides), device=args.device)
    except FileNotFoundError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0
