"""Montagem da A1: lote, casa e endereço; ofício da licença; recorte NAIP; NDVI; rótulos; split.

Toda a lógica é pura (DataFrame e numpy entram, DataFrame sai); só o `ChipMaker` lê e grava disco.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, mapping, shape
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)

ATTRIBUTES = ("pool", "solar")
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0

# motivos de rótulo vazio (o recorte não permite responder)
REASON_TREE = "arvore"
REASON_OUTSIDE = "fora_do_recorte"
REASON_NO_PHOTO = "sem_foto"


def area_m2(geom: BaseGeometry) -> float:
    """Área aproximada em m² de uma geometria em graus (escala local pela latitude)."""
    lat = geom.centroid.y
    return geom.area * M_PER_DEG_LAT * M_PER_DEG_LON_EQ * math.cos(math.radians(lat))


def chip_id_for(item_id: str, house_id: str, year: int) -> str:
    """Chave estável do recorte: hash(item NAIP, casa, ano)."""
    return hashlib.sha1(f"{item_id}|{house_id}|{year}".encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------
# Licenças
# ------------------------------------------------------------------------------------------
class PermitTradeClassifier:
    """Diz o ofício de uma licença (piscina, solar) por regras de palavra na descrição."""

    def __init__(self, trades: dict[str, dict[str, str]]) -> None:
        self.rules = {
            name: (re.compile(r["include"]), re.compile(r["exclude"]) if r.get("exclude") else None)
            for name, r in trades.items()
        }

    def classify(self, description: str | None) -> set[str]:
        """Ofícios encontrados na descrição (maiúsculas); vazio se nenhum."""
        text = (description or "").upper()
        found = set()
        for name, (inc, exc) in self.rules.items():
            if inc.search(text) and not (exc and exc.search(text)):
                found.add(name)
        return found

    def events(self, rows: list[dict[str, Any]], parcel_ids: set[str]) -> tuple[pd.DataFrame, dict]:
        """Licenças com ofício, uma por permit_number, ligadas a um lote que existe."""
        df = pd.DataFrame(rows)
        stats: dict[str, Any] = {"linhas": len(df)}
        if df.empty:
            return pd.DataFrame(columns=PERMIT_COLUMNS), stats | {"no_catalogo": 0}
        df = df.sort_values(["permit_number", "issue_date"]).drop_duplicates(
            "permit_number", keep="last"
        )
        stats["unicas"] = len(df)
        trades = df["description"].map(self.classify)
        df = df.assign(
            is_pool=trades.map(lambda t: "pool" in t),
            is_solar=trades.map(lambda t: "solar" in t),
            trade=trades.map(lambda t: ",".join(sorted(t))),
        )
        df = df[df["trade"] != ""]
        stats["com_oficio"] = len(df)
        df["parcel_id"] = df["tcad_id"].astype(str).str.strip().str.zfill(10)
        in_layer = df["parcel_id"].isin(parcel_ids)
        stats["sem_lote"] = int((~in_layer).sum())
        df = df[in_layer]
        stats["no_catalogo"] = len(df)
        stats["piscina"] = int(df["is_pool"].sum())
        stats["solar"] = int(df["is_solar"].sum())
        out = df.rename(columns={"original_address1": "address"})
        out["issue_date"] = out["issue_date"].str.slice(0, 10)
        return out[PERMIT_COLUMNS].reset_index(drop=True), stats


PERMIT_COLUMNS = [
    "permit_number", "parcel_id", "trade", "is_pool", "is_solar", "issue_date", "address",
    "description",
]  # fmt: skip


# ------------------------------------------------------------------------------------------
# Lote, endereço e casa
# ------------------------------------------------------------------------------------------
class HouseBuilder:
    """Monta lote, endereço e casa: a casa é o maior prédio do lote; lote sem endereço sai."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.min_area = float(cfg["min_area_m2"])
        self.max_area = float(cfg["max_area_m2"])
        self.max_addresses = int(cfg["max_addresses"])
        self.address_type = int(cfg["address_type"])

    def parcels(self, features: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict]:
        """Lotes com `parcel_id` (PID_10) único; sem PID sai; PID repetido vira a união."""
        recs = []
        for f in features:
            p = f.get("properties") or {}
            pid = (p.get("PID_10") or "").strip()
            if not pid or not f.get("geometry"):
                recs.append(None)
                continue
            recs.append(
                {
                    "parcel_id": pid,
                    "prop_id": p.get("PROP_ID"),
                    "situs": p.get("SITUS"),
                    "zoning": p.get("ZONING"),
                    "geometry": shapely.make_valid(shape(f["geometry"])),
                }
            )
        ok = [r for r in recs if r]
        stats = {"feicoes": len(features), "sem_pid": len(recs) - len(ok)}
        df = pd.DataFrame(ok)
        if df.empty:
            return pd.DataFrame(
                columns=["parcel_id", "prop_id", "situs", "zoning", "geometry"]
            ), stats
        dup = df["parcel_id"].duplicated(keep=False)
        stats["pid_repetido"] = int(df.loc[dup, "parcel_id"].nunique())
        if dup.any():
            merged = df[dup].groupby("parcel_id", as_index=False).agg(
                {"prop_id": "first", "situs": "first", "zoning": "first",
                 "geometry": lambda g: shapely.union_all(list(g))}
            )  # fmt: skip
            df = pd.concat([df[~dup], merged], ignore_index=True)
        df = df.sort_values("parcel_id").reset_index(drop=True)
        stats["lotes"] = len(df)
        return df, stats

    def addresses(
        self, features: list[dict[str, Any]], parcels: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Pontos de endereço (só ADDRESS_TYPE de endereço) com o lote que os contém."""
        recs = []
        for f in features:
            p = f.get("properties") or {}
            if p.get("ADDRESS_TYPE") != self.address_type or not f.get("geometry"):
                continue
            lon, lat = f["geometry"]["coordinates"][:2]
            recs.append(
                {
                    "address_id": int(p["PLACE_ID"]),
                    "address": (p.get("FULL_STREET_NAME") or "").strip(),
                    "lon": float(lon),
                    "lat": float(lat),
                }
            )
        df = pd.DataFrame(recs, columns=["address_id", "address", "lon", "lat"])
        stats = {"pontos": len(features), "tipo_endereco": len(df)}
        df = df[df["address"] != ""].drop_duplicates("address_id").reset_index(drop=True)
        pts = [Point(x, y) for x, y in zip(df["lon"], df["lat"], strict=True)]
        df["parcel_id"] = self._containing(pts, parcels)
        stats["sem_lote"] = int(df["parcel_id"].isna().sum())
        df = df.dropna(subset=["parcel_id"]).reset_index(drop=True)
        stats["com_lote"] = len(df)
        return df, stats

    @staticmethod
    def _containing(geoms: list[BaseGeometry], parcels: pd.DataFrame) -> list[str | None]:
        """O lote que contém cada geometria (ponto ou ponto interno); None se nenhum."""
        out: list[str | None] = [None] * len(geoms)
        if not geoms or parcels.empty:
            return out
        tree = shapely.STRtree(list(parcels["geometry"]))
        idx_in, idx_tree = tree.query(geoms, predicate="within")
        pids = parcels["parcel_id"].to_numpy()
        for i, t in zip(idx_in, idx_tree, strict=True):
            if out[i] is None:  # lotes sobrepostos: fica o primeiro (ordem do parcel_id)
                out[i] = pids[t]
        return out

    def buildings(self, df: pd.DataFrame, parcels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Prédios do Overture com área em m² e o lote que contém o seu ponto interno."""
        geoms = [shapely.from_wkb(bytes(g)) for g in df["geometry"]]
        out = pd.DataFrame({"building_id": df["id"].astype(str), "geometry": geoms})
        out["area_m2"] = [area_m2(g) for g in geoms]
        out["parcel_id"] = self._containing([g.representative_point() for g in geoms], parcels)
        stats = {"predios": len(out), "sem_lote": int(out["parcel_id"].isna().sum())}
        return out, stats

    def houses(
        self, parcels: pd.DataFrame, addresses: pd.DataFrame, buildings: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Uma casa por lote: o maior prédio, com o endereço mais perto dele."""
        lots: dict[str, Any] = {"lotes": len(parcels)}
        b = buildings.dropna(subset=["parcel_id"])
        b = b.sort_values(["parcel_id", "area_m2", "building_id"], ascending=[True, False, True])
        n_build = b.groupby("parcel_id").size()
        biggest = b.drop_duplicates("parcel_id", keep="first").set_index("parcel_id")
        addr_by_lot = {k: g for k, g in addresses.groupby("parcel_id")}
        rows, reasons = [], {"sem_predio": 0, "sem_endereco": 0, "multifamiliar": 0,
                             "area_fora_da_faixa": 0}  # fmt: skip
        for pid in parcels["parcel_id"]:
            if pid not in biggest.index:
                reasons["sem_predio"] += 1
                continue
            addrs = addr_by_lot.get(pid)
            if addrs is None or addrs.empty:
                reasons["sem_endereco"] += 1
                continue
            if len(addrs) > self.max_addresses:
                reasons["multifamiliar"] += 1
                continue
            bld = biggest.loc[pid]
            if not (self.min_area <= bld["area_m2"] <= self.max_area):
                reasons["area_fora_da_faixa"] += 1
                continue
            c = bld["geometry"].centroid
            d = (addrs["lon"] - c.x) ** 2 + (addrs["lat"] - c.y) ** 2
            best = addrs.assign(_d=d).sort_values(["_d", "address_id"]).iloc[0]
            rows.append(
                {
                    "house_id": bld["building_id"],
                    "parcel_id": pid,
                    "address_id": int(best["address_id"]),
                    "address": best["address"],
                    "lon": float(c.x),
                    "lat": float(c.y),
                    "area_m2": round(float(bld["area_m2"]), 1),
                    "n_buildings": int(n_build[pid]),
                    "geometry": bld["geometry"],
                }
            )
        df = pd.DataFrame(rows)
        lots |= reasons | {"casas": len(df)}
        return df, lots


# ------------------------------------------------------------------------------------------
# Piscinas do OSM
# ------------------------------------------------------------------------------------------
def osm_pool_geometries(elements: list[dict[str, Any]]) -> pd.DataFrame:
    """Polígono de cada piscina do OSM (way fechado ou relation com 'outer')."""
    recs = []
    for e in elements:
        rings: list[list[tuple[float, float]]] = []
        if e["type"] == "way" and e.get("geometry"):
            rings = [[(p["lon"], p["lat"]) for p in e["geometry"]]]
        elif e["type"] == "relation":
            for m in e.get("members", []):
                if m.get("role") == "outer" and m.get("geometry"):
                    rings.append([(p["lon"], p["lat"]) for p in m["geometry"]])
        polys = [shapely.Polygon(r) for r in rings if len(r) >= 4]
        if not polys:
            continue
        geom = shapely.make_valid(shapely.union_all(polys))
        recs.append({"osm_id": f"{e['type']}/{e['id']}", "geometry": geom})
    return pd.DataFrame(recs, columns=["osm_id", "geometry"])


# ------------------------------------------------------------------------------------------
# Recorte e NDVI
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NaipRaster:
    """Um GeoTIFF NAIP local (janela da quadrícula sobre a área)."""

    item_id: str
    year: int
    path: Path


class NdviReasoner:
    """Fração de vegetação (NDVI) no telhado e no quintal, e o motivo de rótulo vazio."""

    def __init__(self, veg_threshold: float, roof_tree_frac: float, yard_tree_frac: float) -> None:
        self.veg_threshold = veg_threshold
        self.roof_tree_frac = roof_tree_frac
        self.yard_tree_frac = yard_tree_frac

    @staticmethod
    def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """NDVI = (NIR - R) / (NIR + R), 0 onde a soma é 0."""
        r, n = red.astype(np.float32), nir.astype(np.float32)
        s = n + r
        return np.where(s > 0, (n - r) / np.where(s > 0, s, 1), 0.0)

    def veg_fraction(self, red: np.ndarray, nir: np.ndarray, mask: np.ndarray) -> float | None:
        """Fração dos pixels da máscara com NDVI acima do limiar; None se a máscara é vazia."""
        if mask.sum() == 0:
            return None
        v = self.ndvi(red, nir)[mask.astype(bool)]
        return float((v > self.veg_threshold).mean())

    def reason(
        self, attribute: str, roof_frac: float | None, yard_frac: float | None
    ) -> str | None:
        """Motivo que impede responder pela foto, ou None."""
        if attribute == "solar" and roof_frac is not None and roof_frac > self.roof_tree_frac:
            return REASON_TREE
        if attribute == "pool" and yard_frac is not None and yard_frac > self.yard_tree_frac:
            return REASON_TREE
        return None


class ChipMaker:
    """Corta 96 × 96 px do NAIP centrado no contorno; grava PNG RGB e o NIR (PNG cinza)."""

    def __init__(
        self,
        rasters: list[NaipRaster],
        out_dir: Path,
        reasoner: NdviReasoner,
        size_px: int = 96,
        max_nodata_frac: float = 0.01,
    ) -> None:
        self.rasters = rasters
        self.out_dir = Path(out_dir)
        self.rgb_dir = self.out_dir / "chips"
        self.nir_dir = self.out_dir / "chips_nir"
        self.reasoner = reasoner
        self.size = size_px
        self.max_nodata = max_nodata_frac

    def _window(self, src: Any, x: float, y: float) -> Any:
        """Janela size×size centrada em (x, y) no CRS do raster, ou None se sai do raster."""
        from rasterio.windows import Window

        row, col = src.index(x, y)
        half = self.size // 2
        r0, c0 = row - half, col - half
        if r0 < 0 or c0 < 0 or r0 + self.size > src.height or c0 + self.size > src.width:
            return None
        return Window(c0, r0, self.size, self.size)

    def make(
        self, houses: pd.DataFrame, parcels: pd.DataFrame, pools: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """Um recorte por casa (o primeiro raster que cobre a janela com dados)."""
        import rasterio
        from PIL import Image
        from rasterio.features import rasterize
        from rasterio.warp import transform, transform_geom

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.nir_dir.mkdir(parents=True, exist_ok=True)
        stats = {"casas": len(houses), "criados": 0, "reaproveitados": 0, "sem_foto": 0}
        parcel_geom = dict(zip(parcels["parcel_id"], parcels["geometry"], strict=True))
        pools_by_lot = {
            k: list(g["geometry"])
            for k, g in pools.dropna(subset=["parcel_id"]).groupby("parcel_id")
        }
        srcs = [(r, rasterio.open(r.path)) for r in self.rasters]
        rows = []
        try:
            crs = srcs[0][1].crs
            xs, ys = transform("EPSG:4326", crs, list(houses["lon"]), list(houses["lat"]))
            for (_, h), x, y in zip(houses.iterrows(), xs, ys, strict=True):
                found = self._cut(srcs, h["house_id"], x, y)
                if found is None:
                    stats["sem_foto"] += 1
                    continue
                raster, src, win, rgb, nir, reused = found
                stats["reaproveitados" if reused else "criados"] += 1
                tf = src.window_transform(win)
                to_px = lambda g: transform_geom("EPSG:4326", crs, mapping(g))  # noqa: E731
                shp = (self.size, self.size)
                roof = rasterize([to_px(h["geometry"])], out_shape=shp, transform=tf)
                lot = rasterize([to_px(parcel_geom[h["parcel_id"]])], out_shape=shp, transform=tf)
                yard = (lot == 1) & (roof == 0)
                red = rgb[..., 0]
                roof_frac = self.reasoner.veg_fraction(red, nir, roof)
                yard_frac = self.reasoner.veg_fraction(red, nir, yard)
                left, top = tf.c, tf.f
                right, bottom = left + tf.a * self.size, top + tf.e * self.size
                pools_in = 0
                for pg in pools_by_lot.get(h["parcel_id"], []):
                    p = pg.representative_point()
                    (px,), (py,) = transform("EPSG:4326", crs, [p.x], [p.y])
                    pools_in += int(left <= px <= right and bottom <= py <= top)
                rows.append(
                    {
                        "chip_id": chip_id_for(raster.item_id, h["house_id"], raster.year),
                        "house_id": h["house_id"],
                        "parcel_id": h["parcel_id"],
                        "item_id": raster.item_id,
                        "image_year": raster.year,
                        "x_center": float(x),
                        "y_center": float(y),
                        "epsg": int(crs.to_epsg()),
                        "roof_tree_frac": roof_frac,
                        "yard_tree_frac": yard_frac,
                        "osm_pools_in_chip": pools_in,
                    }
                )
        finally:
            for _, s in srcs:
                s.close()
        df = pd.DataFrame(rows)
        if not df.empty:
            df["path"] = [f"processed/chips/{c}.png" for c in df["chip_id"]]
            df["nir_path"] = [f"processed/chips_nir/{c}.png" for c in df["chip_id"]]
        log.info(
            "recortes: %d criados, %d reaproveitados, %d sem foto",
            stats["criados"], stats["reaproveitados"], stats["sem_foto"],
        )  # fmt: skip
        _ = Image  # o import fica aqui: PIL só é preciso quando há recorte
        return df, stats

    def _cut(self, srcs: list[tuple[NaipRaster, Any]], house_id: str, x: float, y: float) -> Any:
        """Lê (ou reaproveita do disco) o recorte; tenta os rasters na ordem de mais margem."""
        from PIL import Image

        def margin(pair: tuple[NaipRaster, Any]) -> float:
            b = pair[1].bounds
            return -min(x - b.left, b.right - x, y - b.bottom, b.top - y)

        for raster, src in sorted(srcs, key=margin):
            cid = chip_id_for(raster.item_id, house_id, raster.year)
            rgb_path, nir_path = self.rgb_dir / f"{cid}.png", self.nir_dir / f"{cid}.png"
            win = self._window(src, x, y)
            if win is None:
                continue
            if rgb_path.is_file() and nir_path.is_file():
                rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
                nir = np.asarray(Image.open(nir_path).convert("L"))
                return raster, src, win, rgb, nir, True
            data = src.read(window=win)  # (4, H, W): R, G, B, NIR
            nodata = float((data == 0).all(axis=0).mean())
            if nodata > self.max_nodata:
                continue
            rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
            nir = data[3].astype(np.uint8)
            Image.fromarray(rgb, "RGB").save(rgb_path)
            Image.fromarray(nir, "L").save(nir_path)
            return raster, src, win, rgb, nir, False
        return None


# ------------------------------------------------------------------------------------------
# Rótulos e split
# ------------------------------------------------------------------------------------------
class LabelBuilder:
    """Rótulo por recorte × atributo: 1 com evidência (OSM ou licença), 0 sem, vazio com motivo."""

    def __init__(self, reasoner: NdviReasoner) -> None:
        self.reasoner = reasoner

    def build(
        self, chips: pd.DataFrame, permits: pd.DataFrame, pools: pd.DataFrame
    ) -> pd.DataFrame:
        """Uma linha por (chip_id, attribute)."""
        pool_permits = _ids_by_lot(permits[permits["is_pool"]], "permit_number")
        solar_permits = _ids_by_lot(permits[permits["is_solar"]], "permit_number")
        osm = _ids_by_lot(pools.dropna(subset=["parcel_id"]), "osm_id")
        rows = []
        for c in chips.itertuples(index=False):
            for attr in ATTRIBUTES:
                osm_ids = osm.get(c.parcel_id, []) if attr == "pool" else []
                permit_ids = (pool_permits if attr == "pool" else solar_permits).get(
                    c.parcel_id, []
                )
                sources = [s for s, ids in (("osm", osm_ids), ("permit", permit_ids)) if ids]
                has = bool(sources)
                reason = self.reasoner.reason(attr, c.roof_tree_frac, c.yard_tree_frac)
                if (
                    reason is None
                    and attr == "pool"
                    and osm_ids
                    and not permit_ids
                    and c.osm_pools_in_chip == 0
                ):
                    reason = REASON_OUTSIDE  # a piscina do OSM está no lote, mas fora da foto
                rows.append(
                    {
                        "chip_id": c.chip_id,
                        "attribute": attr,
                        "label": None if reason else int(has),
                        "has_evidence": has,
                        "label_source": "+".join(sources) if sources else "nenhuma",
                        "evidence_id": (osm_ids + permit_ids)[0] if has else None,
                        "reason": reason,
                        "first_check": None,
                        "lara_check": None,
                    }
                )
        df = pd.DataFrame(rows)
        df["label"] = df["label"].astype("Int64")
        for col in ("evidence_id", "reason", "first_check", "lara_check"):
            df[col] = df[col].astype("string")  # texto mesmo quando tudo é vazio
        return df


def _ids_by_lot(df: pd.DataFrame, col: str) -> dict[str, list[str]]:
    return {k: sorted(g[col].astype(str)) for k, g in df.groupby("parcel_id")}


class AreaSplitter:
    """Split por células de 1 km: células inteiras vão para treino, validação ou teste."""

    def __init__(self, cell_m: float, fractions: dict[str, float], seed: int) -> None:
        self.cell_m = cell_m
        self.fractions = fractions
        self.seed = seed

    def cell_of(self, x: float, y: float) -> str:
        """Id da célula de um ponto em metros (CRS projetado)."""
        return f"E{int(x // self.cell_m)}_N{int(y // self.cell_m)}"

    def assign(self, cells: Iterable[str]) -> dict[str, str]:
        """Sorteia as células (seed fixa) e enche treino, validação e teste pela contagem."""
        counts = pd.Series(list(cells)).value_counts()
        order = sorted(counts.index)
        random.Random(self.seed).shuffle(order)
        total = counts.sum()
        names = list(self.fractions)
        limits = np.cumsum([self.fractions[n] for n in names]) * total
        out, acc, k = {}, 0, 0
        for cell in order:
            while k < len(names) - 1 and acc >= limits[k]:
                k += 1
            out[cell] = names[k]
            acc += counts[cell]
        return out


class ReviewSampler:
    """Amostra estratificada do teste para a conferência a olho (cotas por estrato)."""

    def __init__(self, quotas: dict[str, int], seed: int) -> None:
        self.quotas = quotas
        self.seed = seed

    @staticmethod
    def stratum(pool: Any, solar: Any) -> str:
        """Estrato de um recorte pelos rótulos."""
        pool = None if pd.isna(pool) else int(pool)
        solar = None if pd.isna(solar) else int(solar)
        if pool == 1:
            return "pool_pos"
        if solar == 1:
            return "solar_pos"
        if pool is None or solar is None:
            return "empty"
        return "neg"

    def sample(self, wide: pd.DataFrame) -> pd.DataFrame:
        """`wide` tem chip_id, pool, solar (só teste). Devolve as linhas sorteadas com estrato."""
        w = wide.copy()
        w["stratum"] = [self.stratum(p, s) for p, s in zip(w["pool"], w["solar"], strict=True)]
        parts = []
        for name, n in self.quotas.items():
            g = w[w["stratum"] == name].sort_values("chip_id")
            parts.append(g.sample(n=min(n, len(g)), random_state=self.seed))
        out = pd.concat(parts)
        short = sum(self.quotas.values()) - len(out)
        if short > 0:  # um estrato pequeno: completa com negativos
            rest = w[~w["chip_id"].isin(out["chip_id"])].sort_values("chip_id")
            out = pd.concat([out, rest.sample(n=min(short, len(rest)), random_state=self.seed)])
        return out.sort_values(["stratum", "chip_id"]).reset_index(drop=True)
