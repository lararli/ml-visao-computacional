"""GoldWriter: chave única, MERGE idempotente e um modelo novo não apaga o antigo."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from visaocomp.a1.abstain import ABSTAIN, NO, YES
from visaocomp.a1.gold import EVIDENCE, HOUSE_ATTRIBUTE, GoldWriter, agreement


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA silver")
    con.execute("""CREATE TABLE silver.house AS SELECT * FROM (VALUES
        ('h1', '0000000001', '1 Main St', -97.7, 30.2),
        ('h2', '0000000002', '2 Main St', -97.7, 30.3))
        t(house_id, parcel_id, address, lon, lat)""")
    con.execute("""CREATE TABLE silver.chip AS SELECT * FROM (VALUES
        ('c1', 'h1', 2022, 0.1, 0.2), ('c2', 'h2', 2022, 0.7, 0.9))
        t(chip_id, house_id, image_year, roof_tree_frac, yard_tree_frac)""")
    con.execute("""CREATE TABLE silver.label AS SELECT * FROM (VALUES
        ('c1', 'pool', 1, 'osm', 'way/1'), ('c1', 'solar', 0, 'permit', NULL),
        ('c2', 'pool', NULL, NULL, NULL), ('c2', 'solar', 1, 'permit', 'P-9'))
        t(chip_id, attribute, label, label_source, evidence_id)""")
    con.close()
    return path


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chip_id": ["c1", "c1", "c2", "c2"],
            "attribute": ["pool", "solar", "pool", "solar"],
            "probability": [0.9, 0.7, 0.5, 0.2],
            "decision": [YES, YES, ABSTAIN, ABSTAIN],
            "reason": [None, None, "árvore", "árvore"],
        }
    )


def _count(path: Path, table: str) -> int:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


def test_merge_idempotente_e_versao_nova_soma(catalog):
    w = GoldWriter(catalog)
    ctx = w.context(["c1", "c2"])
    t = w.tables(_decisions(), ctx, "vit-aaa", "run1")
    w.write(t)
    w.write(t)  # rodar de novo não duplica
    assert _count(catalog, HOUSE_ATTRIBUTE) == 4
    assert _count(catalog, EVIDENCE) == 4
    w.write(w.tables(_decisions(), ctx, "vit-bbb", "run2"))  # modelo novo: linhas novas
    assert _count(catalog, HOUSE_ATTRIBUTE) == 8
    assert _count(catalog, EVIDENCE) == 4  # evidência é lote x atributo: a última versão
    con = duckdb.connect(str(catalog), read_only=True)
    dup = con.execute(f"""SELECT count(*) FROM (SELECT parcel_id, attribute, image_year,
        model_version FROM {HOUSE_ATTRIBUTE} GROUP BY ALL HAVING count(*) > 1)""").fetchone()[0]
    row = con.execute(f"""SELECT address, decision, reason FROM {HOUSE_ATTRIBUTE}
        WHERE parcel_id = '0000000002' AND attribute = 'pool' LIMIT 1""").fetchone()
    con.close()
    assert dup == 0
    assert row == ("2 Main St", ABSTAIN, "árvore")


def test_agreement():
    assert agreement(YES, 1) == "concorda"
    assert agreement(NO, 0) == "concorda"
    assert agreement(YES, 0) == "imagem sem licença"
    assert agreement(NO, 1) == "licença sem imagem"
    assert agreement(ABSTAIN, 1) == ABSTAIN
    assert agreement(YES, None) == ABSTAIN


def test_evidencia_cruza_decisao_e_registro(catalog):
    w = GoldWriter(catalog)
    t = w.tables(_decisions(), w.context(["c1", "c2"]), "vit-aaa", "run1")
    ev = t[EVIDENCE].set_index(["parcel_id", "attribute"])["agreement"]
    assert ev[("0000000001", "pool")] == "concorda"
    assert ev[("0000000001", "solar")] == "imagem sem licença"
    assert ev[("0000000002", "solar")] == ABSTAIN
