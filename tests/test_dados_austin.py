"""Dados da A1: Austin no catalog.duckdb, o manifesto e a conferência a olho.

Precisa de `make data` já executado (por isso o marcador `data`, fora do `make test`).
"""

from __future__ import annotations

import logging
import time

import duckdb
import pandas as pd
import pytest

from visaocomp.a1.pipeline import STAGE, find_review_dir
from visaocomp.a1.review import MARKS
from visaocomp.a1.sources import SKIP_MSG, load_a1_config, make_sources
from visaocomp.config import load_data_config
from visaocomp.paths import data_dir, repo_root
from visaocomp.run import latest_manifest

pytestmark = pytest.mark.data


def _connect() -> duckdb.DuckDBPyConnection:
    """Abre o catálogo só para leitura; espera se outro processo estiver escrevendo."""
    path = data_dir() / load_data_config().catalog
    for _ in range(60):
        try:
            return duckdb.connect(str(path), read_only=True)
        except duckdb.IOException:
            time.sleep(5)
    raise RuntimeError(f"catálogo ocupado: {path}")


@pytest.fixture(scope="module")
def tables():
    con = _connect()
    try:
        q = lambda sql: con.execute(sql).df()  # noqa: E731
        out = {
            "parcel": q("SELECT parcel_id FROM silver.parcel"),
            "address": q("SELECT address_id, parcel_id FROM silver.address"),
            "house": q("SELECT house_id, parcel_id, address_id, address FROM silver.house"),
            "chip": q("SELECT chip_id, house_id, cell_id, split, path, nir_path FROM silver.chip"),
            "label": q("SELECT chip_id, attribute, label, reason, first_check FROM silver.label"),
        }
    finally:
        con.close()
    return out


@pytest.fixture(scope="module")
def manifest():
    doc = latest_manifest(repo_root() / "outputs", STAGE)
    assert doc is not None, "nenhum manifesto da A1 em outputs/: rode `make data`"
    return doc


def test_chaves_unicas(tables):
    assert tables["parcel"]["parcel_id"].is_unique and tables["parcel"]["parcel_id"].notna().all()
    assert tables["house"]["house_id"].is_unique
    assert tables["house"]["parcel_id"].is_unique
    assert tables["chip"]["chip_id"].is_unique
    assert not tables["label"].duplicated(["chip_id", "attribute"]).any()
    assert len(tables["house"]) > 0 and len(tables["chip"]) > 0


def test_toda_casa_tem_endereco_e_lote(tables):
    h = tables["house"]
    assert h["address"].notna().all() and (h["address"].str.strip() != "").all()
    assert h["parcel_id"].notna().all() and h["address_id"].notna().all()
    assert h["parcel_id"].isin(tables["parcel"]["parcel_id"]).all()
    assert h["address_id"].isin(tables["address"]["address_id"]).all()
    a = tables["address"].set_index("address_id")["parcel_id"]
    assert (a.loc[h["address_id"]].to_numpy() == h["parcel_id"].to_numpy()).all()
    assert tables["chip"]["house_id"].isin(h["house_id"]).all()


def test_treino_validacao_e_teste_nao_dividem_celula(tables, manifest):
    c = tables["chip"]
    assert set(c["split"]) == {"train", "val", "test"}
    assert (c.groupby("cell_id")["split"].nunique() == 1).all()
    assert manifest["counts"]["celulas_em_mais_de_um_split"] == 0
    frac = c["split"].value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8


def test_contagem_por_atributo_rotulo_e_motivo_no_manifesto(tables, manifest):
    lab = tables["label"]
    rot, mot = manifest["counts"]["rotulos"], manifest["counts"]["vazios_por_motivo"]
    n_chips = len(tables["chip"])
    for attr in ("pool", "solar"):
        g = lab[lab["attribute"] == attr]
        assert len(g) == n_chips
        expected = {
            "1": int((g["label"] == 1).sum()),
            "0": int((g["label"] == 0).sum()),
            "vazio": int(g["label"].isna().sum()),
        }
        assert rot[attr] == expected
        assert sum(rot[attr].values()) == n_chips
        assert mot[attr] == {k: int(v) for k, v in g["reason"].value_counts().items()}
        assert sum(mot[attr].values()) == expected["vazio"]
        assert g.loc[g["label"].isna(), "reason"].notna().all()  # vazio sempre tem motivo
    assert manifest["counts"]["casas_com_endereco_pct"] == 100.0
    assert manifest["counts"]["casas"] == len(tables["house"])


def test_conferencia_inicial_csv_200_linhas_e_revisao_html(tables):
    folder = find_review_dir()
    assert folder is not None, "não achei notas/conferencia"
    inicial = pd.read_csv(folder / "conferencia_inicial.csv", dtype=str, keep_default_na=False)
    assert len(inicial) == 200 and inicial["chip_id"].is_unique
    test_chips = set(tables["chip"].loc[tables["chip"]["split"] == "test", "chip_id"])
    assert set(inicial["chip_id"]) <= test_chips
    assert set(inicial["inicial_pool"]) <= set(MARKS) and set(inicial["inicial_solar"]) <= set(
        MARKS
    )
    nao_da = inicial[inicial["inicial_pool"] == "não dá"]
    assert (nao_da["inicial_pool_reason"].str.strip() != "").all()  # "não dá" tem motivo
    html = (folder / "revisao.html").read_text(encoding="utf-8")
    assert html.count('class="card"') == 200
    checked = tables["label"].dropna(subset=["first_check"])
    assert len(checked) == 400  # a marcação também está no catálogo


def test_rodar_de_novo_nao_refaz_consultas_nem_recortes(tables, caplog):
    class NoNetwork:
        def request(self, *args, **kwargs):
            raise AssertionError("tentou consultar de novo")

    srcs = make_sources(load_a1_config(), data_dir(), NoNetwork())  # type: ignore[arg-type]
    with caplog.at_level(logging.INFO):
        assert [s.download() for s in srcs.values()] == [False] * 6
    assert caplog.text.count(SKIP_MSG) == 6
    for rel in list(tables["chip"]["path"]) + list(tables["chip"]["nir_path"]):
        assert (data_dir() / rel).is_file()
