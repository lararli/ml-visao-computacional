"""Ponto de entrada do `make run-a2`: CLIP no ADS-16, embeddings no DuckDB, ranking e busca."""

from __future__ import annotations

import argparse
import logging
import sys

from visaocomp.a2.config import load_a2_config
from visaocomp.a2.pipeline import ThresholdCheckMissingError, build_pipeline
from visaocomp.a2.report import A2Report
from visaocomp.config import default_config_path
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir, repo_root
from visaocomp.run import latest_manifest


def main() -> int:
    """Roda a atividade 2 e devolve o código de saída."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="só sorteia os pares para conferir o limiar a olho (folhas em outputs/calibracao/)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenera reports/sections/a2.md a partir da última execução, sem recalcular",
    )
    args = parser.parse_args()
    configure_logging()
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root = repo_root()
    cfg = load_a2_config()
    outputs = root / "outputs"
    check = root / cfg.threshold_check_file
    report_path = root / "reports" / "sections" / "a2.md"
    narrative = root / "reports" / "a2_narrativa.yaml"

    if args.report_only:
        manifest = latest_manifest(outputs, "a2")
        if manifest is None:
            print("ERRO: nenhuma execução da A2 em outputs/; rode `make run-a2`", file=sys.stderr)
            return 2
        A2Report(outputs / manifest["run_id"], cfg, narrative).write(report_path)
        return 0

    pipe = build_pipeline(
        cfg,
        data_dir(),
        outputs,
        check,
        data_cfg_path=default_config_path(),
        with_text_comparison=not args.calibrate,
    )
    try:
        if args.calibrate:
            pipe.calibrate(outputs / "calibracao")
            return 0
        run = pipe.run()
    except ThresholdCheckMissingError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    A2Report(run.path, cfg, narrative).write(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
