"""Entrada do ViT do zero: treina e avalia (padrão) ou só regera o a1.md (--report-only)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from visaocomp.a1.scratch import STAGE, ScratchPipeline
from visaocomp.a1.scratch_config import ScratchConfig
from visaocomp.a1.scratch_report import ScratchReport, latest_run
from visaocomp.device import pick_device
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir, repo_root

log = logging.getLogger(__name__)

SECTION = Path("reports") / "sections" / "a1.md"
NARRATIVE = Path("reports") / "a1_scratch_narrativa.yaml"
FIGURES = Path("reports") / "figures" / "a1"


def run_scratch(
    cfg: ScratchConfig | None = None,
    outputs_root: Path | None = None,
    device: str | None = None,
    root: Path | None = None,
) -> Path:
    """Treina, avalia e gera o a1.md; devolve o caminho do manifest.json."""
    cfg = cfg or ScratchConfig.from_yaml()
    root = root or repo_root()
    data = data_dir()
    outputs = outputs_root or root / "outputs"
    pipeline = ScratchPipeline(
        cfg, data / "catalog.duckdb", data, outputs, pick_device(device), root / FIGURES
    )
    manifest = pipeline.run()
    ScratchReport(manifest.parent, root / NARRATIVE).write(root / SECTION)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Linha de comando de `scripts/run_a1_scratch.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="outro configs/a1_scratch.yaml")
    parser.add_argument("--device", default=None, help="cuda, mps ou cpu (padrão: automático)")
    parser.add_argument("--epochs", type=int, default=None, help="troca as épocas do config")
    parser.add_argument(
        "--report-only", action="store_true", help="só regera reports/sections/a1.md do último run"
    )
    args = parser.parse_args(argv)
    configure_logging()
    root = repo_root()
    try:
        if args.report_only:
            run = latest_run(root / "outputs")
            if run is None:
                raise FileNotFoundError(
                    f"nenhum run {STAGE} em outputs/: rode `make run-a1-scratch`"
                )
            ScratchReport(run, root / NARRATIVE).write(root / SECTION)
            return 0
        cfg = ScratchConfig.from_yaml(args.config)
        if args.epochs is not None:
            cfg = cfg.with_overrides(epochs=args.epochs)
        run_scratch(cfg, device=args.device)
    except FileNotFoundError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0
