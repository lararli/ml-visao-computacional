"""Orquestra a A1: consultas -> bronze -> silver (lote, endereço, casa, licença, piscina, recorte,
rótulo) -> manifesto. Grava no catalog.duckdb com MERGE pela chave (rodar de novo não duplica)."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from visaocomp.a1.build import (
    ATTRIBUTES,
    AreaSplitter,
    ChipMaker,
    HouseBuilder,
    LabelBuilder,
    NaipRaster,
    NdviReasoner,
    PermitTradeClassifier,
    ReviewSampler,
    osm_pool_geometries,
)
from visaocomp.a1.sources import A1Config, A1Source
from visaocomp.paths import repo_root
from visaocomp.run import RunDir

log = logging.getLogger(__name__)

STAGE = "a1_data"
REVIEW_ENV = "VISAOCOMP_REVIEW_DIR"
SAMPLE_FILE = "processed/a1_review_sample.csv"

# chave de cada tabela do catálogo (Modelo de dados do PLANO_TECNICO)
KEYS: dict[str, list[str]] = {
    "bronze.naip_item": ["item_id", "_run_id"],
    "bronze.parcel_raw": ["objectid", "_run_id"],
    "bronze.address_raw": ["objectid", "_run_id"],
    "bronze.building_raw": ["id", "_run_id"],
    "bronze.osm_pool_raw": ["osm_id", "_run_id"],
    "bronze.permit_raw": ["permit_number", "_run_id"],
    "silver.parcel": ["parcel_id"],
    "silver.address": ["address_id"],
    "silver.house": ["house_id"],
    "silver.permit_event": ["permit_number"],
    "silver.osm_pool": ["osm_id"],
    "silver.chip": ["chip_id"],
    "silver.label": ["chip_id", "attribute"],
}


def find_review_dir() -> Path | None:
    """Pasta da conferência: VISAOCOMP_REVIEW_DIR, ou <ancestral>/notas/conferencia."""
    env = os.environ.get(REVIEW_ENV)
    if env:
        return Path(env)
    for parent in repo_root().parents:
        if (parent / "notas").is_dir():
            return parent / "notas" / "conferencia"
    return None


class A1Catalog:
    """Escreve no catalog.duckdb: abre, faz o MERGE e fecha logo (o arquivo é compartilhado)."""

    def __init__(self, path: Path, wait_s: float = 5.0, tries: int = 60) -> None:
        self.path = Path(path)
        self.wait_s = wait_s
        self.tries = tries

    def _connect(self) -> duckdb.DuckDBPyConnection:
        for attempt in range(1, self.tries + 1):
            try:
                return duckdb.connect(str(self.path))
            except duckdb.IOException as exc:  # outro processo está com o arquivo
                if attempt == self.tries:
                    raise
                log.warning("catálogo ocupado (%s); espero %.0f s", exc, self.wait_s)
                time.sleep(self.wait_s)
        raise RuntimeError("inalcançável")

    def merge_all(self, tables: dict[str, pd.DataFrame], prune: set[str]) -> dict[str, int]:
        """MERGE de cada tabela pela chave; em `prune`, apaga as chaves que saíram."""
        con = self._connect()
        counts = {}
        try:
            con.execute("BEGIN")
            for schema in ("bronze", "silver", "gold"):
                con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            for name, df in tables.items():
                counts[name] = self._merge(con, name, df, KEYS[name], name in prune)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()
        return counts

    @staticmethod
    def _merge(
        con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, key: list[str], prune: bool
    ) -> int:
        now = datetime.now(UTC).replace(tzinfo=None)
        df = df.copy()
        df["_created_at"] = now
        df["_updated_at"] = now
        if df.duplicated(key).any():
            raise ValueError(f"{table}: chave {key} repetida nos dados novos")
        con.register("_new", df)
        cols = list(df.columns)
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema || '.' || "
            "table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            con.execute(f"CREATE TABLE {table} AS SELECT * FROM _new LIMIT 0")
            con.execute(f"ALTER TABLE {table} ADD PRIMARY KEY ({', '.join(key)})")
        else:  # o esquema evolui: coluna nova entra, tipo diferente é convertido
            old = dict(
                con.execute(f"SELECT column_name, column_type FROM (DESCRIBE {table})").fetchall()
            )
            new = dict(
                con.execute("SELECT column_name, column_type FROM (DESCRIBE _new)").fetchall()
            )
            for col, typ in new.items():
                if col not in old:
                    con.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {typ}')
                elif old[col] != typ and col not in key and typ != '"NULL"':
                    con.execute(f'ALTER TABLE {table} ALTER "{col}" SET DATA TYPE {typ}')
        keep = {"_created_at"} | set(key)
        sets = ", ".join(f'"{c}" = excluded."{c}"' for c in cols if c not in keep)
        col_list = ", ".join(f'"{c}"' for c in cols)
        con.execute(
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _new "
            f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {sets}"
        )
        if prune:
            on = " AND ".join(f"t.{k} = n.{k}" for k in key)
            con.execute(f"DELETE FROM {table} t WHERE NOT EXISTS (SELECT 1 FROM _new n WHERE {on})")
        con.unregister("_new")
        return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


class A1Pipeline:
    """Roda a A1 de ponta a ponta e grava o manifesto em outputs/<run_id>/."""

    def __init__(
        self,
        cfg: A1Config,
        sources: dict[str, A1Source],
        data_dir: Path,
        outputs_root: Path,
        catalog_path: Path,
        review_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.sources = sources
        self.data_dir = Path(data_dir)
        self.outputs_root = Path(outputs_root)
        self.catalog = A1Catalog(catalog_path)
        self.review_dir = review_dir
        nd = cfg.ndvi
        self.reasoner = NdviReasoner(
            nd["veg_threshold"], nd["roof_tree_frac"], nd["yard_tree_frac"]
        )

    # -- etapas -------------------------------------------------------------------------------
    def fetch(self, force: bool = False) -> int:
        """Faz as consultas que faltam; devolve quantas foram feitas agora."""
        return sum(int(s.download(force=force)) for s in self.sources.values())

    def run(self, force: bool = False) -> Path:
        """Consulta, monta, grava no catálogo e escreve o manifesto; devolve o manifesto."""
        t0 = time.monotonic()
        log.info("A1: início, área %s %s", self.cfg.area.name, self.cfg.area.bbox)
        fetched = self.fetch(force)
        counts: dict[str, Any] = {"consultas_feitas_agora": fetched}
        s = self.sources
        hb = HouseBuilder(self.cfg.house)

        parcels, counts["lotes_fonte"] = hb.parcels(s["parcels"].features())
        addresses, counts["enderecos"] = hb.addresses(s["addresses"].features(), parcels)
        bdf = (
            duckdb.connect()
            .execute(f"SELECT * FROM read_parquet('{s['buildings'].parquet()}') ORDER BY id")
            .df()
        )
        buildings, counts["predios"] = hb.buildings(bdf, parcels)
        houses, counts["lotes"] = hb.houses(parcels, addresses, buildings)

        classifier = PermitTradeClassifier(self.cfg.trades)
        permits, counts["licencas"] = classifier.events(
            s["permits"].rows(), set(parcels["parcel_id"])
        )
        pools = osm_pool_geometries(s["osm_pools"].elements())
        pools["parcel_id"] = hb._containing(
            [g.representative_point() for g in pools["geometry"]], parcels
        )
        house_lots = set(houses["parcel_id"])
        counts["piscinas_osm"] = {
            "total": len(pools),
            "sem_lote": int(pools["parcel_id"].isna().sum()),
            "em_lote_com_casa": int(pools["parcel_id"].isin(house_lots).sum()),
        }
        counts["licencas"]["em_lote_com_casa"] = int(permits["parcel_id"].isin(house_lots).sum())

        rasters = [
            NaipRaster(i["item_id"], int(str(i["datetime"])[:4]), i["path"])
            for i in s["naip"].items()
        ]
        maker = ChipMaker(
            rasters, self.data_dir / "processed", self.reasoner,
            int(self.cfg.chip["size_px"]), float(self.cfg.chip["max_nodata_frac"]),
        )  # fmt: skip
        chips, counts["recortes"] = maker.make(houses, parcels, pools)

        sp = self.cfg.split
        splitter = AreaSplitter(float(sp["cell_m"]), sp["fractions"], int(sp["seed"]))
        chips["cell_id"] = [
            splitter.cell_of(x, y)
            for x, y in zip(chips["x_center"], chips["y_center"], strict=True)
        ]
        cell_split = splitter.assign(chips["cell_id"])
        chips["split"] = chips["cell_id"].map(cell_split)

        labels = LabelBuilder(self.reasoner).build(chips, permits, pools)
        inicial = self._first_checks()
        if inicial:
            key = list(zip(labels["chip_id"], labels["attribute"], strict=True))
            labels["first_check"] = [inicial.get(k) for k in key]
            counts["conferencia_inicial"] = self._agreement(labels)

        sample = self._review_sample(chips, labels, houses)
        counts |= self._label_counts(chips, labels, houses)
        counts["amostra_conferencia"] = dict(Counter(sample["stratum"]))
        counts["catalogo"] = self._write_catalog(
            parcels, addresses, houses, buildings, bdf, permits, pools, chips, labels, rasters
        )
        run = RunDir(self.outputs_root, STAGE)
        sample.to_csv(run.path / "review_sample.csv", index=False)
        counts["segundos"] = round(time.monotonic() - t0, 1)
        manifest = run.write_manifest(
            [src.describe() for src in self.sources.values()],
            counts,
            extra={
                "area": {"name": self.cfg.area.name, "bbox": list(self.cfg.area.bbox)},
                "config": str(self.cfg.path.name),
            },
        )
        log.info(
            "A1: fim, %d casas, %d recortes, %d consultas feitas agora; manifesto %s",
            len(houses), len(chips), fetched, manifest,
        )  # fmt: skip
        return manifest

    # -- partes ---------------------------------------------------------------------------------
    def _first_checks(self) -> dict[tuple[str, str], str]:
        """Conferência inicial (conferencia_inicial.csv) por (chip_id, atributo), quando existe."""
        if not self.review_dir or not (self.review_dir / "conferencia_inicial.csv").is_file():
            return {}
        df = pd.read_csv(self.review_dir / "conferencia_inicial.csv", dtype=str)
        out = {}
        for r in df.itertuples(index=False):
            out[(r.chip_id, "pool")] = r.inicial_pool
            out[(r.chip_id, "solar")] = r.inicial_solar
        return out

    @staticmethod
    def _agreement(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Rótulo automático x conferência inicial, por atributo (só os conferidos)."""
        g = labels.dropna(subset=["first_check"])
        out = {}
        for attr in ATTRIBUTES:
            a = g[g["attribute"] == attr]
            lab = a["label"].map(lambda v: "vazio" if pd.isna(v) else str(int(v)))
            pair = lab + " x " + a["first_check"]
            out[attr] = {k: int(v) for k, v in pair.value_counts().sort_index().items()}
        return out

    def _review_sample(
        self, chips: pd.DataFrame, labels: pd.DataFrame, houses: pd.DataFrame
    ) -> pd.DataFrame:
        """Amostra estratificada de 200 recortes do teste; grava em processed/."""
        wide = labels.pivot(index="chip_id", columns="attribute", values="label").reset_index()
        meta = chips[["chip_id", "house_id", "parcel_id", "split", "path"]].merge(
            houses[["house_id", "address"]], on="house_id"
        )
        wide = wide.merge(meta, on="chip_id")
        test = wide[wide["split"] == "test"]
        rv = self.cfg.review
        sample = ReviewSampler(rv["quotas"], int(rv["seed"])).sample(test)
        reasons = labels.pivot(index="chip_id", columns="attribute", values="reason")
        sample = sample.merge(
            reasons.rename(columns={a: f"reason_{a}" for a in ATTRIBUTES}).reset_index(),
            on="chip_id",
        )
        cols = ["chip_id", "house_id", "parcel_id", "address", "split", "stratum", "pool",
                "solar", "reason_pool", "reason_solar", "path"]  # fmt: skip
        sample = sample[cols].rename(columns={"pool": "label_pool", "solar": "label_solar"})
        target = self.data_dir / SAMPLE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(target, index=False)
        return sample

    @staticmethod
    def _label_counts(
        chips: pd.DataFrame, labels: pd.DataFrame, houses: pd.DataFrame
    ) -> dict[str, Any]:
        """Contagens do manifesto: rótulo por atributo, motivo, fonte da evidência e split."""
        out: dict[str, Any] = {
            "casas": len(houses),
            "lotes_com_casa": houses["parcel_id"].nunique(),
        }
        out["casas_com_endereco_pct"] = round(100 * houses["address"].ne("").mean(), 2)
        out["casas_com_lote_pct"] = round(100 * houses["parcel_id"].notna().mean(), 2)
        rot, mot, fonte = {}, {}, {}
        for attr in ATTRIBUTES:
            g = labels[labels["attribute"] == attr]
            rot[attr] = {
                "1": int((g["label"] == 1).sum()),
                "0": int((g["label"] == 0).sum()),
                "vazio": int(g["label"].isna().sum()),
            }
            mot[attr] = {k: int(v) for k, v in g["reason"].value_counts().items()}
            fonte[attr] = {k: int(v) for k, v in g["label_source"].value_counts().items()}
        out["rotulos"] = rot
        out["vazios_por_motivo"] = mot
        out["fonte_da_evidencia"] = fonte
        split = {}
        for name, g in chips.groupby("split"):
            gl = labels[labels["chip_id"].isin(g["chip_id"])]
            split[name] = {
                "recortes": len(g),
                "celulas": g["cell_id"].nunique(),
                **{
                    f"{a}_1": int(((gl["attribute"] == a) & (gl["label"] == 1)).sum())
                    for a in ATTRIBUTES
                },
            }
        out["split"] = split
        by_cell = chips.groupby("cell_id")["split"].nunique()
        out["celulas_em_mais_de_um_split"] = int((by_cell > 1).sum())
        return out

    def _write_catalog(
        self,
        parcels: pd.DataFrame,
        addresses: pd.DataFrame,
        houses: pd.DataFrame,
        buildings: pd.DataFrame,
        bdf: pd.DataFrame,
        permits: pd.DataFrame,
        pools: pd.DataFrame,
        chips: pd.DataFrame,
        labels: pd.DataFrame,
        rasters: list[NaipRaster],
    ) -> dict[str, int]:
        s = self.sources

        def bronze(df: pd.DataFrame, src: A1Source) -> pd.DataFrame:
            out = df.copy()
            out["_run_id"] = src.key  # a consulta que trouxe o dado (a pasta em raw/)
            out["_query"] = json.dumps(src.query(), ensure_ascii=False)
            out["_source"] = src.name
            return out

        def props(src: A1Source) -> pd.DataFrame:
            feats = src.features()  # type: ignore[attr-defined]
            df = pd.DataFrame([f["properties"] for f in feats])
            df.columns = [c.lower() for c in df.columns]
            df["geometry_json"] = [json.dumps(f["geometry"]) for f in feats]
            if "objectid_1" in df.columns:
                df = df.rename(columns={"objectid": "objectid_tcad", "objectid_1": "objectid"})
            return df

        def wkt(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["geom_wkt"] = [g.wkt for g in out.pop("geometry")]
            return out

        naip = pd.DataFrame(
            [
                {"item_id": i["item_id"], "datetime": i["datetime"], "gsd": i["gsd"],
                 "epsg": i["epsg"], "bbox": json.dumps(i["bbox"]),
                 "path": str(Path(i["path"]).relative_to(self.data_dir))}
                for i in s["naip"].items()
            ]
        )  # fmt: skip
        braw = bdf.copy()
        braw["geometry"] = [bytes(g) for g in braw["geometry"]]
        braw["bbox"] = braw["bbox"].map(json.dumps)
        osm_raw = pd.DataFrame(
            [{"osm_id": f"{e['type']}/{e['id']}", "json": json.dumps(e)}
             for e in s["osm_pools"].elements()]
        ).drop_duplicates("osm_id")  # fmt: skip
        praw = pd.DataFrame(s["permits"].rows()).drop_duplicates("permit_number", keep="last")
        src_of = {
            "silver.parcel": "parcels", "silver.address": "addresses", "silver.house": "buildings",
            "silver.permit_event": "permits", "silver.osm_pool": "osm_pools",
            "silver.chip": "naip", "silver.label": "a1_labels",
        }  # fmt: skip
        silver = {
            "silver.parcel": wkt(parcels),
            "silver.address": addresses,
            "silver.house": wkt(houses),
            "silver.permit_event": permits,
            "silver.osm_pool": wkt(pools),
            "silver.chip": chips,
            "silver.label": labels,
        }
        for name, df in silver.items():
            df["_source"] = src_of[name]
        tables = {
            "bronze.naip_item": bronze(naip, s["naip"]),
            "bronze.parcel_raw": bronze(props(s["parcels"]), s["parcels"]),
            "bronze.address_raw": bronze(props(s["addresses"]), s["addresses"]),
            "bronze.building_raw": bronze(braw, s["buildings"]),
            "bronze.osm_pool_raw": bronze(osm_raw, s["osm_pools"]),
            "bronze.permit_raw": bronze(praw, s["permits"]),
            **silver,
        }
        _ = (buildings, rasters)
        return self.catalog.merge_all(tables, prune=set(silver))
