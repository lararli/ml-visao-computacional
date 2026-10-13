"""Smoke da A4: caso, GAN sem e com mitigação, FID, sintético, recall e a4.md, em CPU."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from visaocomp.a4.config import load_a4_config
from visaocomp.a4.pipeline import A4Pipeline
from visaocomp.a4.report import A4Report, find_complete_run
from visaocomp.run import RunDir

pytestmark = pytest.mark.smoke

SMOKE = {
    "case": {"test_per_class": 4, "train_counts": {"NORMAL": 10, "PNEUMONIA": 8, "COVID": 6}},
    "gan": {
        "epochs": 2,
        "batch_size": 8,
        "ngf": 8,
        "ndf": 8,
        "sample_every": 1,
        "fid_every": 1,
        "fid_n": 24,
    },
    "synthetic": {"n_covid": 10},
    "classifier": {
        "seeds": [0, 1, 2],
        "image_size": 32,
        "epochs": 1,
        "batch_size": 8,
        "pretrained": False,
    },
}


class TinyFeatures(nn.Module):
    """Extrator minúsculo no lugar da Inception (sem download)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Conv2d(1, 4, 3, 2), nn.AdaptiveAvgPool2d(2), nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_raw(root, n_per_class=20, copies=2):
    """Pasta raw/ falsa: 3 classes de ruído com algumas cópias exatas."""
    rng = np.random.default_rng(0)
    for c in ("NORMAL", "PNEUMONIA", "COVID"):
        (root / c).mkdir(parents=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
            Image.fromarray(arr).save(root / c / f"{c}_{i}.png")
        for i in range(copies):
            Image.open(root / c / f"{c}_{i}.png").save(root / c / f"{c}_copia_{i}.png")
    return root


def test_a4_de_ponta_a_ponta_em_amostra_minuscula(tmp_path):
    cfg = load_a4_config(overrides=SMOKE)
    raw = make_raw(tmp_path / "raw")
    outputs = tmp_path / "outputs"
    run = RunDir(outputs, "a4")
    A4Pipeline(cfg, raw, run, torch.device("cpu"), TinyFeatures(), tmp_path / "interim").run()

    out = run.path
    dedup = json.loads((out / "dedup.json").read_text(encoding="utf-8"))
    assert dedup["copias_removidas"] == 6
    split = pd.read_csv(out / "splits.csv")
    assert set(split["path"][split["split"] == "test"]).isdisjoint(
        split["path"][split["split"] != "test"]
    )
    gan = json.loads((out / "gan_metrics.json").read_text(encoding="utf-8"))
    assert set(gan["runs"]) == {"sem_mitigacao", "com_mitigacao"}
    for r in gan["runs"].values():
        assert r["n_d_steps"] == r["n_g_steps"] > 0
        assert len(r["fid_curve"]) == 2 and r["sample_grids"]
    assert len(list((out / "synthetic").glob("covid_*.png"))) == 10
    rec = pd.read_csv(out / "recall" / "recall.csv")
    assert len(rec) == 6 and set(rec["condition"]) == {"sem_sintetico", "com_sintetico"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completo" and manifest["stage"] == "a4"

    assert find_complete_run(outputs, cfg.sha256) == out
    md = A4Report(out, cfg).write(tmp_path / "a4.md", tmp_path / "figs")
    assert out.name in md.read_text(encoding="utf-8")
