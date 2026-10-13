"""Ponto de entrada do `make data`: baixa, verifica e grava o manifesto."""

from __future__ import annotations

import argparse
import sys

from visaocomp.config import load_data_config
from visaocomp.data import DataIngestor, DownloadFailedError, SourceChangedError
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir, repo_root


def main() -> int:
    """Lê o config, roda o DataIngestor e devolve o código de saída."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="baixa de novo, mesmo se já passa")
    args = parser.parse_args()
    configure_logging()
    ingestor = DataIngestor(load_data_config(), data_dir(), repo_root() / "outputs")
    try:
        ingestor.run(force=args.force)
    except (SourceChangedError, DownloadFailedError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
