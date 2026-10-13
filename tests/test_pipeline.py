"""O pipeline: uma atividade = um dataset = uma gold; todo modelo em gold.model_run."""

import duckdb
import pytest

from visaocomp.a1.pipeline import KEYS
from visaocomp.paths import data_dir
from visaocomp.pipeline import (
    ADS16_OBJECT,
    DATASETS,
    IMAGES_PREDICTION,
    MODEL_RUN,
    XRAY_PREDICTION,
    ModelRun,
)

GOLD = {
    "austin_houses": ["gold.austin_house_attribute", "gold.austin_house_attribute_evidence"],
    "ads16": [ADS16_OBJECT],
    "images": [IMAGES_PREDICTION],
    "chest_xray": [XRAY_PREDICTION],
}


def test_um_dataset_por_atividade():
    acts = {name: cls.activity for name, cls in DATASETS.items()}
    assert acts == {"austin_houses": "A1", "ads16": "A2", "images": "A3", "chest_xray": "A4.1"}
    for cls in DATASETS.values():
        assert cls.scripts, cls.dataset


def test_model_run_row():
    r = ModelRun("A3", "images", "efficientnet_b0", "efficientnet_b0-abc", "run1", "acc", 0.9,
                 {"macro_f1": 0.8}, "outputs/run1")  # fmt: skip
    row = r.row()
    assert row["value"] == 0.9 and '"macro_f1": 0.8' in row["metrics_json"]
    assert set(KEYS[MODEL_RUN]) <= set(row)


@pytest.mark.data
def test_gold_dos_quatro_datasets():
    path = data_dir() / "catalog.duckdb"
    assert path.exists(), "rode `make data` e `make pipeline`"
    con = duckdb.connect(str(path), read_only=True)
    try:
        for tables in GOLD.values():
            for t in tables:
                n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                assert n > 0, t
                key = ", ".join(KEYS[t])
                dup = con.execute(
                    f"SELECT count(*) FROM (SELECT {key} FROM {t} GROUP BY ALL HAVING count(*) > 1)"
                ).fetchone()[0]
                assert dup == 0, f"{t}: chave repetida"
        acts = {r[0] for r in con.execute(f"SELECT DISTINCT activity FROM {MODEL_RUN}").fetchall()}
        assert acts == {"A1", "A2", "A3", "A4.1"}
        nulls = con.execute(
            "SELECT count(*) FROM gold.austin_house_attribute WHERE address IS NULL "
            "OR parcel_id IS NULL"
        ).fetchone()[0]
        assert nulls == 0, "todo registro da gold de Austin tem endereço e lote"
    finally:
        con.close()
