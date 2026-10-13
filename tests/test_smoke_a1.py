"""Smoke da A1: o pipeline inteiro (consultas, catálogo, recortes, rótulos, manifesto) no mundo
sintético, sem rede. Rodar de novo não refaz consultas nem recortes."""

from __future__ import annotations

import json

import duckdb
import pytest

from a1_world import WorldSession, world_sources
from visaocomp.a1.pipeline import STAGE, A1Pipeline
from visaocomp.a1.sources import load_a1_config
from visaocomp.run import latest_manifest

pytestmark = pytest.mark.smoke


def test_pipeline_a1_em_mundo_sintetico(tmp_path):
    data, outputs, cat = tmp_path / "data", tmp_path / "outputs", tmp_path / "data" / "c.duckdb"
    session = WorldSession(fail_first=1)  # cada URL falha uma vez antes de responder
    srcs = world_sources(data, session, tmp_path / "naip_src")
    pipe = A1Pipeline(load_a1_config(), srcs, data, outputs, cat, review_dir=None)
    pipe.run()
    doc = latest_manifest(outputs, STAGE)
    c = doc["counts"]
    assert c["consultas_feitas_agora"] == 6 and c["casas"] == 5
    assert c["recortes"]["sem_foto"] == 1 and c["casas_com_endereco_pct"] == 100.0
    assert c["rotulos"]["pool"]["1"] == 1 and c["vazios_por_motivo"]["solar"] == {"arvore": 1}
    assert c["vazios_por_motivo"]["pool"] == {"fora_do_recorte": 1}
    assert c["licencas"]["sem_lote"] == 1 and c["celulas_em_mais_de_um_split"] == 0
    assert all(s["verified"] for s in doc["sources"])
    con = duckdb.connect(str(cat), read_only=True)
    tables = {
        r[0]
        for r in con.execute(
            "SELECT table_schema || '.' || table_name FROM information_schema.tables"
        ).fetchall()
    }
    assert {"silver.parcel", "silver.address", "silver.house", "silver.permit_event",
            "silver.osm_pool", "silver.chip", "silver.label", "bronze.naip_item",
            "bronze.parcel_raw", "bronze.address_raw", "bronze.building_raw",
            "bronze.osm_pool_raw", "bronze.permit_raw"} <= tables  # fmt: skip
    n_label = con.execute("SELECT count(*) FROM silver.label").fetchone()[0]
    con.close()
    assert n_label == 2 * 4

    calls = len(session.calls)
    pipe2 = A1Pipeline(load_a1_config(), world_sources(data, session, tmp_path / "naip_src"),
                       data, outputs, cat)  # fmt: skip
    pipe2.run()
    doc2 = latest_manifest(outputs, STAGE)
    assert doc2["run_id"] != doc["run_id"]
    assert doc2["counts"]["consultas_feitas_agora"] == 0
    assert doc2["counts"]["recortes"]["criados"] == 0
    assert len(session.calls) == calls  # nenhuma chamada nova
    assert doc2["counts"]["catalogo"] == doc["counts"]["catalogo"]
    assert json.dumps(doc2["counts"]["rotulos"]) == json.dumps(doc["counts"]["rotulos"])
