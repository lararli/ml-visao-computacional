"""Ponto de entrada do `make run-a4`: caso de raio-X, GAN, recall com e sem sintético e a4.md."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from visaocomp.a4.config import load_a4_config
from visaocomp.a4.pipeline import STAGE, A4Pipeline, open_run, pick_device
from visaocomp.a4.report import A4Report, find_complete_run
from visaocomp.config import load_data_config
from visaocomp.data.sources import KaggleSource
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir, repo_root
from visaocomp.run import RunDir

log = logging.getLogger("run_a4")


def main() -> int:
    """Roda a A4 (ou pula, se já existe uma execução completa com o mesmo config)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="roda de novo, numa pasta nova")
    parser.add_argument("--resume", metavar="RUN_ID", help="continua uma execução interrompida")
    parser.add_argument(
        "--report-only", action="store_true", help="só regera reports/sections/a4.md"
    )
    parser.add_argument("--device", help="cuda, mps ou cpu (padrão: automático)")
    args = parser.parse_args()
    configure_logging()
    cfg = load_a4_config()
    outputs = repo_root() / "outputs"
    report_path = repo_root() / "reports" / "sections" / "a4.md"
    figures = repo_root() / "reports" / "figures" / "a4"

    done = find_complete_run(outputs, cfg.sha256)
    if done and not (args.force or args.resume):
        log.info(
            "a4: execução completa %s já existe com o mesmo config; pulando o treino", done.name
        )
        A4Report(done, cfg).write(report_path, figures)
        return 0
    if args.report_only:
        print("ERRO: nenhuma execução completa da A4 com este config em outputs/", file=sys.stderr)
        return 2

    src_cfg = load_data_config().source(cfg.source)
    source = KaggleSource(src_cfg, data_dir())
    try:
        raw_dir = source.load()  # verifica sha256 e contagem antes de usar
    except FileNotFoundError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    check = source.verify()
    info = {
        "name": src_cfg.name,
        "source": f"{src_cfg.owner}/{src_cfg.dataset}",
        "version": src_cfg.version,
        "sha256": check.sha256,
        "n_files": check.n_files,
    }
    try:
        run = open_run(outputs, args.resume) if args.resume else RunDir(outputs, STAGE)
    except FileNotFoundError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    pipeline = A4Pipeline(
        cfg,
        raw_dir=raw_dir,
        run=run,
        device=pick_device(args.device),
        cache_dir=data_dir() / "interim" / "a4",
        source_info=info,
    )
    pipeline.run()
    A4Report(Path(run.path), cfg).write(report_path, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
