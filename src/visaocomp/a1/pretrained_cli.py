"""Entrada da A1 pré-treinada: treina e faz a parte de depois (padrão), ou só a parte de
depois (--post-only): comparação, "não dá" com motivo, gold e reports/sections/a1.md."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from visaocomp.a1.finetune import CONFIG, STAGE, PretrainedPipeline, load_specs
from visaocomp.a1.pretrained_report import PretrainedPost
from visaocomp.a1.scratch import STAGE as SCRATCH_STAGE
from visaocomp.device import pick_device
from visaocomp.log import configure_logging
from visaocomp.paths import data_dir, repo_root

log = logging.getLogger(__name__)

SECTION = Path("reports") / "sections" / "a1.md"
NARRATIVE = Path("reports") / "a1_pretrained_narrativa.yaml"


def latest(outputs: Path, stage: str) -> Path:
    runs = sorted(p for p in outputs.glob(f"*-{stage}") if (p / stage / "metrics.json").exists())
    if not runs:
        raise FileNotFoundError(f"nenhum run de {stage} em {outputs}")
    return runs[-1]


def post(root: Path, outputs: Path, write_gold: bool = True) -> Path:
    """Comparação, abstenção, gold e o bloco do a1.md para o run mais recente."""
    raw = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    p = PretrainedPost(
        latest(outputs, STAGE),
        latest(outputs, SCRATCH_STAGE),
        data_dir() / "catalog.duckdb",
        root / NARRATIVE,
        raw.get("abstain"),
    )
    stats = p.run(write_gold=write_gold)
    return p.write(root / SECTION, stats)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--post-only", action="store_true", help="não treina; usa o último run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    configure_logging()
    root = repo_root()
    outputs = root / "outputs"
    if not args.post_only:
        base, specs = load_specs()
        data = data_dir()
        PretrainedPipeline(
            base, specs, data / "catalog.duckdb", data, outputs, pick_device(args.device)
        ).run()
    path = post(root, outputs)
    log.info("A1 pré-treinado: a1.md atualizado em %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
