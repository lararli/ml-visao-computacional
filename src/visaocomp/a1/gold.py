"""GoldWriter: a camada gold da A1 no catalog.duckdb.

- `gold.austin_house_attribute`: lote x atributo x ano da foto x versão do modelo. Um modelo
  novo grava linhas novas e não apaga as antigas.
- `gold.austin_house_attribute_evidence`: lote x atributo. Cruza a decisão da imagem com a
  licença e o OpenStreetMap e grava `agreement`.
A escrita é MERGE pela chave (A1Catalog), então rodar de novo não duplica.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from visaocomp.a1.abstain import ABSTAIN, NO, YES
from visaocomp.a1.pipeline import KEYS, A1Catalog

HOUSE_ATTRIBUTE = "gold.austin_house_attribute"
EVIDENCE = "gold.austin_house_attribute_evidence"
KEYS.setdefault(HOUSE_ATTRIBUTE, ["parcel_id", "attribute", "image_year", "model_version"])
KEYS.setdefault(EVIDENCE, ["parcel_id", "attribute"])

AGREE = "concorda"
IMAGE_ONLY = "imagem sem licença"
RECORD_ONLY = "licença sem imagem"


def agreement(decision: str, label: float | None) -> str:
    """Compara a decisão da imagem com o rótulo da licença ou do OSM (1, 0 ou vazio)."""
    if decision == ABSTAIN or label is None or pd.isna(label):
        return ABSTAIN
    if (decision == YES and label == 1) or (decision == NO and label == 0):
        return AGREE
    return IMAGE_ONLY if decision == YES else RECORD_ONLY


class GoldWriter:
    """Monta as duas tabelas gold a partir das decisões e grava por MERGE."""

    SOURCE = "a1_pretrained (decisão da imagem + silver.label)"

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = Path(catalog_path)

    def context(self, chip_ids: list[str]) -> pd.DataFrame:
        """Lote, endereço, ano e NDVI de cada recorte; mais o rótulo de cada atributo."""
        con = duckdb.connect(str(self.catalog_path), read_only=True)
        try:
            con.register("_ids", pd.DataFrame({"chip_id": chip_ids}))
            return con.execute(
                """
                SELECT c.chip_id, c.house_id, h.parcel_id, c.image_year, h.address, h.lon, h.lat,
                       c.roof_tree_frac, c.yard_tree_frac, l.attribute, l.label, l.label_source,
                       l.evidence_id
                FROM _ids JOIN silver.chip c USING (chip_id)
                JOIN silver.house h USING (house_id)
                LEFT JOIN silver.label l USING (chip_id)
                """
            ).df()
        finally:
            con.close()

    def tables(
        self, decisions: pd.DataFrame, context: pd.DataFrame, model_version: str, run_id: str
    ) -> dict[str, pd.DataFrame]:
        """`decisions`: chip_id, attribute, probability, decision, reason."""
        df = decisions.merge(context, on=["chip_id", "attribute"], how="left", validate="1:1")
        if df["parcel_id"].isna().any():
            raise ValueError("recorte sem lote no catálogo: a gold exige parcel_id")
        df["model_version"], df["run_id"], df["_source"] = model_version, run_id, self.SOURCE
        house = df[
            [
                "parcel_id",
                "attribute",
                "image_year",
                "model_version",
                "house_id",
                "address",
                "lon",
                "lat",
                "decision",
                "probability",
                "reason",
                "chip_id",
                "run_id",
                "_source",
            ]
        ].copy()
        ev = df[
            [
                "parcel_id",
                "attribute",
                "model_version",
                "decision",
                "label",
                "label_source",
                "evidence_id",
                "_source",
            ]
        ].rename(columns={"decision": "image_decision", "label": "record_label"})
        ev["agreement"] = [
            agreement(d, lab)
            for d, lab in zip(ev["image_decision"], ev["record_label"], strict=True)
        ]
        return {HOUSE_ATTRIBUTE: house.reset_index(drop=True), EVIDENCE: ev.reset_index(drop=True)}

    def write(self, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
        return A1Catalog(self.catalog_path).merge_all(tables, prune=set())
