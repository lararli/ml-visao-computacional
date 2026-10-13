"""Ponto de entrada do `make pipeline`: para cada dataset, treina se precisa e publica a gold."""

from __future__ import annotations

import argparse
import json
import sys

from visaocomp.log import configure_logging
from visaocomp.pipeline import DATASETS, PipelineRunner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only", default="", help=f"datasets separados por vírgula: {', '.join(DATASETS)}"
    )
    ap.add_argument(
        "--force", action="store_true", help="treina de novo mesmo com resultado em outputs/"
    )
    ap.add_argument("--gold-only", action="store_true", help="não treina; só publica a gold")
    args = ap.parse_args(argv)
    configure_logging()
    names = [n for n in args.only.split(",") if n] or None
    summary = PipelineRunner(names).run(force=args.force, gold_only=args.gold_only)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
