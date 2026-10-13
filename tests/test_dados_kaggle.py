"""Dados do Kaggle: `make data` baixou as 3 fontes, verificou e gravou o manifesto.

Precisa de `make data` já executado (por isso o marcador `data`, fora do `make test`).
"""

from __future__ import annotations

import logging
import re

import pytest

from visaocomp.catalog import Catalog
from visaocomp.config import load_data_config
from visaocomp.data import DataIngestor
from visaocomp.data.sources import READY_MSG
from visaocomp.paths import data_dir, repo_root
from visaocomp.run import latest_manifest

pytestmark = pytest.mark.data

EXPECTED = {
    "ads16": "groffo/ads16-dataset",
    "images_dataset": "pavansanagapati/images-dataset",
    "covid_xray": "sachinkumar413/covid-pneumonia-normal-chest-xray-images",
}


@pytest.fixture(scope="module")
def manifest():
    doc = latest_manifest(repo_root() / "outputs", "data")
    assert doc is not None, "nenhum manifesto de dados em outputs/: rode `make data`"
    return doc


def test_manifesto_tem_3_fontes_com_sha256_e_arquivos(manifest):
    sources = {s["name"]: s for s in manifest["sources"]}
    assert set(sources) == set(EXPECTED)
    cfg = load_data_config()
    for name, slug in EXPECTED.items():
        s = sources[name]
        assert s["source"] == slug
        assert re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
        assert s["sha256"] == cfg.source(name).sha256, "o sha256 do manifesto difere do config"
        assert s["version"] == 1 and s["verified"] is True
        assert s["n_files"] > 0 and s["n_bytes"] > 0
    assert manifest["git"].keys() == {"commit", "dirty"}


def test_dados_em_data_nosync_e_fora_do_git():
    raw = data_dir() / "raw"
    for name in EXPECTED:
        assert (raw / name).is_dir() and any((raw / name).iterdir())
    assert ".nosync" in str(data_dir()), "os dados têm de ficar numa pasta .nosync (iCloud)"
    ignore = (repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "data.nosync/" in ignore and ".venv.nosync/" in ignore


def test_catalogo_tem_os_tres_schemas():
    cfg = load_data_config()
    with Catalog(data_dir() / cfg.catalog) as cat:
        assert cat.list_schemas() == ["bronze", "silver", "gold"]


def test_make_data_de_novo_nao_baixa_nada(caplog, tmp_path):
    """Roda a ingestão de novo com uma sessão que proíbe rede: as 3 dizem 'já existe'."""

    class NoNetwork:
        def get(self, *args, **kwargs):
            raise AssertionError("tentou baixar de novo")

    cfg = load_data_config()
    ingestor = DataIngestor(cfg, data_dir(), tmp_path / "outputs", session=NoNetwork())
    with caplog.at_level(logging.INFO):
        ingestor.run()
    assert caplog.text.count(READY_MSG) == 3
    again = latest_manifest(tmp_path / "outputs", "data")
    assert again["counts"]["downloaded"] == 0 and again["counts"]["skipped"] == 3
