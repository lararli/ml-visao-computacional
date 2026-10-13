"""Ponto de entrada da A1 no `make data`: consultas públicas, catálogo, recortes e manifesto."""

from __future__ import annotations

import argparse
import sys

from visaocomp.a1.pipeline import A1Pipeline, find_review_dir
from visaocomp.a1.sources import load_a1_config, make_sources
from visaocomp.config import load_data_config
from visaocomp.data import DownloadFailedError
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir, repo_root


def main() -> int:
    """Lê o configs/a1.yaml, roda o A1Pipeline e devolve o código de saída."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refaz as consultas")
    args = parser.parse_args()
    configure_logging()
    cfg = load_a1_config()
    ddir = data_dir()
    pipeline = A1Pipeline(
        cfg,
        make_sources(cfg, ddir),
        ddir,
        repo_root() / "outputs",
        ddir / load_data_config().catalog,
        review_dir=find_review_dir(),
    )
    try:
        pipeline.run(force=args.force)
    except DownloadFailedError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
