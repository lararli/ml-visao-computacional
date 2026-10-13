"""Gera a página de conferência da A1 (revisao.html) a partir da amostra e da conferência."""

from __future__ import annotations

import argparse
from pathlib import Path

from visaocomp.a1.pipeline import SAMPLE_FILE, find_review_dir
from visaocomp.a1.review import ReviewPage
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir


def main() -> int:
    """Lê processed/a1_review_sample.csv e <conferencia>/conferencia_inicial.csv; grava revisao.html."""  # noqa: E501
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=None, help="pasta da conferência")
    args = parser.parse_args()
    configure_logging()
    folder = args.dir or find_review_dir()
    if folder is None:
        print("ERRO: não achei notas/conferencia; use --dir", flush=True)
        return 2
    out = ReviewPage(data_dir()).write(
        data_dir() / SAMPLE_FILE, folder / "conferencia_inicial.csv", folder / "revisao.html"
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
