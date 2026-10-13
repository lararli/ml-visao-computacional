"""Smoke: a ingestão inteira (3 fontes, catálogo, manifesto) com .zip minúsculos, sem rede."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

import pytest

from conftest import FakeSession, make_zip
from visaocomp.config import load_data_config
from visaocomp.data import DataIngestor
from visaocomp.data.sources import READY_MSG

pytestmark = pytest.mark.smoke


def test_ingestao_completa_em_amostra_minuscula(tmp_path, caplog):
    cfg_path = tmp_path / "data.yaml"
    shutil.copy(load_data_config().path, cfg_path)
    # a cópia parte sem checksum: o primeiro download do .zip de teste grava o dele
    cfg_path.write_text(
        re.sub(r"sha256: [0-9a-f]{64}", "sha256: null", cfg_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    cfg = load_data_config(cfg_path)
    body = make_zip({"classe/a.png": b"1", "classe/b.png": b"2"})
    data, outputs = tmp_path / "data.nosync", tmp_path / "outputs"

    DataIngestor(cfg, data, outputs, session=FakeSession(body)).run()
    manifest = next(outputs.glob("*/manifest.json"))
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(doc["sources"]) == 3 and doc["counts"]["downloaded"] == 3
    assert all(len(s["sha256"]) == 64 and s["n_files"] == 2 for s in doc["sources"])
    assert doc["counts"]["schemas"] == ["bronze", "silver", "gold"]
    assert (data / "catalog.duckdb").exists()
    assert "sha256: null" not in cfg_path.read_text(encoding="utf-8")

    cfg2 = load_data_config(cfg_path)
    with caplog.at_level(logging.INFO):
        DataIngestor(cfg2, data, outputs, session=FakeSession(None)).run()
    assert caplog.text.count(READY_MSG) == 3
    assert len(list(Path(outputs).glob("*/manifest.json"))) == 2
