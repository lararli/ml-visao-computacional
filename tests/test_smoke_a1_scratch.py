"""Smoke do ViT do zero: catálogo minúsculo -> treino em CPU -> metrics.json, mapas e a1.md."""

from __future__ import annotations

import json

import pytest
import torch

from a1_scratch_world import build_world
from visaocomp.a1.scratch import STAGE, ScratchPipeline
from visaocomp.a1.scratch_config import ScratchConfig
from visaocomp.a1.scratch_report import END, START, ScratchReport, latest_run
from visaocomp.run import latest_manifest

pytestmark = pytest.mark.smoke


def test_pipeline_do_zero_em_catalogo_minusculo(tmp_path):
    cat = build_world(tmp_path / "data")
    cfg = ScratchConfig(patch=16, dim=16, depth=1, heads=2, epochs=2, warmup_epochs=1, batch_size=8)
    outputs, figs = tmp_path / "outputs", tmp_path / "figs"
    pipe = ScratchPipeline(cfg, cat, tmp_path / "data", outputs, torch.device("cpu"), figs)
    manifest = pipe.run()
    assert latest_manifest(outputs, STAGE)["counts"]["recortes"] == 30
    run = latest_run(outputs)
    assert run == manifest.parent
    m = json.loads((run / STAGE / "metrics.json").read_text(encoding="utf-8"))
    assert len(m["history"]["train_loss"]) == 2 and 1 <= m["train"]["best_epoch"] <= 2
    assert set(m["test"]["per_attribute"]) == {"pool", "solar"}
    assert m["test_reviewed"]["n_chips"] == 6
    assert m["test_reviewed"]["per_attribute"]["solar"]["n"] == 5  # um "não dá" fica fora
    assert m["permutation"]["without_pe_max_abs_logit_change"] < 1e-4
    assert {"roof_all", "pool_all"} <= set(m["regions"]) and m["regions"]["roof_all"]["n"] == 6
    for name in m["attention"]["figures"]:
        assert (run / STAGE / name).stat().st_size > 1000 and (figs / name).exists()
    assert (run / STAGE / "best.pt").exists() and (run / STAGE / "curves.png").exists()

    section = tmp_path / "a1.md"
    section.write_text("# A1\n\nparte de outra atividade\n", encoding="utf-8")
    ScratchReport(run).write(section)
    ScratchReport(run).write(section)  # de novo: troca o bloco, não duplica
    text = section.read_text(encoding="utf-8")
    assert text.count(START) == 1 and text.count(END) == 1
    assert "parte de outra atividade" in text and m["run_id"] in text
