"""Um pedaço sintético de Austin para os testes da A1: lotes, endereços, prédios, piscinas,
licenças e duas quadrículas NAIP (GeoTIFF) pequenas. Sem rede."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box, mapping

LON0, LAT0 = -97.79, 30.31
M_LAT = 110_574.0
M_LON = 111_320.0 * math.cos(math.radians(LAT0))
CRS = "EPSG:26914"


def ll(dx_m: float, dy_m: float) -> tuple[float, float]:
    """Ponto a dx, dy metros do centro, em lon/lat."""
    return LON0 + dx_m / M_LON, LAT0 + dy_m / M_LAT


def square(cx: float, cy: float, half_w: float, half_h: float | None = None) -> shapely.Polygon:
    """Retângulo em lon/lat centrado a (cx, cy) metros, com meias larguras em metros."""
    half_h = half_w if half_h is None else half_h
    x0, y0 = ll(cx - half_w, cy - half_h)
    x1, y1 = ll(cx + half_w, cy + half_h)
    return box(x0, y0, x1, y1)


# Lotes: (pid, centro x, centro y, meia largura, meia altura)
LOTS = {
    "0000000001": (0, 0, 20, 20),  # casa + garagem, endereço, hidrante, piscina OSM, POOL DECK
    "0000000002": (50, 0, 20, 20),  # árvore sobre o telhado, licença SOLAR
    "0000000003": (0, 50, 20, 20),  # sem endereço
    "0000000004": (50, 50, 20, 20),  # sem prédio
    "0000000005": (125, 0, 20, 20),  # perto da borda leste da quadrícula A (usa a B)
    "0000000006": (0, -90, 15, 60),  # piscina OSM longe da casa (fora do recorte)
    "0000000007": (900, 900, 20, 20),  # fora das duas quadrículas: sem foto
}
HOUSES = {  # pid -> (centro x, centro y, meia largura) do prédio principal
    "0000000001": (0, 0, 6),
    "0000000002": (50, 0, 6),
    "0000000003": (0, 50, 6),
    "0000000005": (125, 0, 6),
    "0000000006": (0, -45, 6),
    "0000000007": (900, 900, 6),
}


def parcel_features() -> list[dict[str, Any]]:
    """Feições GeoJSON da camada de lotes."""
    feats = []
    for i, (pid, (cx, cy, hw, hh)) in enumerate(LOTS.items(), start=1):
        feats.append(
            {
                "type": "Feature",
                "geometry": mapping(square(cx, cy, hw, hh)),
                "properties": {"OBJECTID_1": i, "PID_10": pid, "PROP_ID": 100 + i,
                               "SITUS": str(i), "ZONING": "SF-3", "LAND_VALUE": None},
            }
        )  # fmt: skip
    return feats


def address_features() -> list[dict[str, Any]]:
    """Pontos de endereço: um por lote com casa (menos o 3) e um hidrante no lote 1."""
    pts = [
        (1, 101, "0000000001", 0, -15, 1, "101 TESTE LN"),
        (2, 102, "0000000001", 10, 15, 4, ""),  # hidrante: ADDRESS_TYPE 4
        (3, 103, "0000000002", 50, -15, 1, "103 TESTE LN"),
        (4, 104, "0000000005", 125, -15, 1, "104 TESTE LN"),
        (5, 105, "0000000006", 0, -35, 1, "105 TESTE LN"),
        (6, 106, "0000000007", 900, 885, 1, "106 TESTE LN"),
        (7, 107, "0000000004", 50, 40, 1, "107 TESTE LN"),
    ]
    feats = []
    for oid, place, _pid, x, y, typ, full in pts:
        lon, lat = ll(x, y)
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"OBJECTID": oid, "PLACE_ID": place, "ADDRESS": place,
                               "ADDRESS_FRACTION": None, "PREFIX_DIRECTION": None,
                               "FULL_STREET_NAME": full, "ADDRESS_TYPE": typ},
            }
        )  # fmt: skip
    return feats


def buildings_frame() -> pd.DataFrame:
    """Prédios no formato do parquet do Overture (geometria WKB, bbox em struct)."""
    rows = []
    for pid, (cx, cy, hw) in HOUSES.items():
        rows.append((f"casa-{pid}", square(cx, cy, hw)))
    rows.append(("garagem-0000000001", square(12, 12, 2.5)))  # menor que a casa do lote 1
    out = []
    for bid, g in rows:
        x0, y0, x1, y1 = g.bounds
        out.append(
            {
                "id": bid,
                "geometry": shapely.to_wkb(g),
                "bbox": {"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1},
                "height": None, "num_floors": None, "subtype": None, "class": None,
                "src_dataset": "teste",
            }
        )  # fmt: skip
    return pd.DataFrame(out)


def osm_elements() -> list[dict[str, Any]]:
    """Duas piscinas do OSM: uma perto da casa 1, outra longe da casa 6 (fora do recorte)."""

    def way(i: int, poly: shapely.Polygon) -> dict[str, Any]:
        coords = list(poly.exterior.coords)
        return {"type": "way", "id": i, "tags": {"leisure": "swimming_pool"},
                "geometry": [{"lon": x, "lat": y} for x, y in coords]}  # fmt: skip

    return [way(11, square(-12, -10, 3)), way(12, square(0, -140, 3))]


def permit_rows() -> list[dict[str, Any]]:
    """Licenças: POOL DECK no lote 1, SOLAR no 2, demolição de piscina, e uma sem lote."""
    base = {"permittype": "BP", "latitude": str(LAT0), "longitude": str(LON0)}
    return [
        base | {"permit_number": "2019-1 BP", "tcad_id": "0000000001",
                "issue_date": "2019-05-01T00:00:00.000", "description": "NEW POOL DECK",
                "original_address1": "101 TESTE LN"},
        base | {"permit_number": "2020-2 EP", "tcad_id": "0000000002",
                "issue_date": "2020-05-01T00:00:00.000",
                "description": "Install rooftop solar PV system", "original_address1": "103"},
        base | {"permit_number": "2020-3 BP", "tcad_id": "0000000005",
                "issue_date": "2020-06-01T00:00:00.000", "description": "DEMO EXISTING POOL",
                "original_address1": "104"},
        base | {"permit_number": "2021-4 BP", "tcad_id": "9999999999",
                "issue_date": "2021-01-01T00:00:00.000", "description": "POOL",
                "original_address1": "x"},
    ]  # fmt: skip


def write_rasters(folder: Path) -> list[Path]:
    """Duas quadrículas 4 bandas a 0,6 m (A e B, B deslocada para leste); árvore no lote 2."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform

    folder.mkdir(parents=True, exist_ok=True)
    (x0,), (y0,) = transform("EPSG:4326", CRS, [LON0], [LAT0])
    paths = []
    for name, east_m in (("item_a", 0.0), ("item_b", 120.0)):
        size = 320  # 192 m
        left, top = x0 - 96 + east_m, y0 + 96
        tf = from_origin(left, top, 0.6, 0.6)
        data = np.zeros((4, size, size), dtype=np.uint8)
        data[0:3] = 120
        data[3] = 60  # NIR baixo: telhado e chão (NDVI negativo)
        # árvore sobre a casa do lote 2 (x de 40 a 60 m, y de -10 a 10 m): NDVI alto
        c0, c1 = int((x0 + 40 - left) / 0.6), int((x0 + 60 - left) / 0.6)
        r0, r1 = int((top - (y0 + 10)) / 0.6), int((top - (y0 - 10)) / 0.6)
        if 0 <= c0 < size:
            data[0, r0:r1, max(c0, 0) : min(c1, size)] = 30
            data[3, r0:r1, max(c0, 0) : min(c1, size)] = 200
        path = folder / f"{name}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", width=size, height=size, count=4, dtype="uint8",
            crs=CRS, transform=tf,
        ) as dst:  # fmt: skip
            dst.write(data)
        paths.append(path)
    return paths


def stac_doc(paths: list[Path]) -> dict[str, Any]:
    """Resposta STAC com um item por GeoTIFF sintético."""
    feats = []
    for p in paths:
        feats.append(
            {
                "type": "Feature",
                "id": p.stem,
                "bbox": [LON0 - 0.01, LAT0 - 0.01, LON0 + 0.01, LAT0 + 0.01],
                "properties": {"datetime": "2022-06-08T16:00:00Z", "gsd": 0.6,
                               "proj:epsg": 26914},
                "assets": {"image": {"href": str(p)}},
            }
        )  # fmt: skip
    return {"type": "FeatureCollection", "features": feats}


class FakeResp:
    """Resposta HTTP mínima (status, texto e JSON)."""

    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self.content = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.text = self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.content)


class WorldSession:
    """Sessão falsa que responde ArcGIS, Overpass, Socrata e STAC com o mundo sintético.

    `fail_first` faz as primeiras chamadas de cada URL devolverem erro (504 e erro em JSON)."""

    def __init__(self, stac: dict[str, Any] | None = None, fail_first: int = 0) -> None:
        self.stac = stac or {"type": "FeatureCollection", "features": []}
        self.fail_first = fail_first
        self.calls: list[tuple[str, str]] = []
        self._seen: dict[str, int] = {}

    def request(self, method: str, url: str, **kw: Any) -> FakeResp:
        self.calls.append((method, url))
        n = self._seen.get(url, 0)
        self._seen[url] = n + 1
        if n < self.fail_first:
            return (
                FakeResp(504, b"<html>gateway timeout</html>")
                if n % 2 == 0
                else (FakeResp(200, {"error": {"code": 500, "message": "limite"}}))
            )
        params = kw.get("params") or {}
        if "parcel" in url or "address" in url:
            feats = parcel_features() if "parcel" in url else address_features()
            off, size = int(params["resultOffset"]), int(params["resultRecordCount"])
            return FakeResp(200, {"type": "FeatureCollection", "features": feats[off : off + size]})
        if "interpreter" in url:
            return FakeResp(200, {"version": 0.6, "elements": osm_elements()})
        if "resource" in url:
            off, size = int(params["$offset"]), int(params["$limit"])
            return FakeResp(200, permit_rows()[off : off + size])
        if "stac" in url:
            return FakeResp(200, self.stac)
        raise AssertionError(f"URL inesperada: {url}")


def world_sources(data_dir: Path, session: WorldSession, raster_dir: Path) -> dict[str, Any]:
    """As 6 fontes da A1 com o config real, respondidas pelo mundo sintético (sem rede).

    NAIP copia os GeoTIFF sintéticos em vez de ler o COG; Overture grava o parquet local."""
    import shutil

    import duckdb

    from visaocomp.a1.sources import (
        NaipSource,
        OvertureBuildingsSource,
        load_a1_config,
        make_sources,
    )

    class LocalNaip(NaipSource):
        def _read_window(self, item: dict[str, Any], target: Path) -> None:
            shutil.copy(item["assets"]["image"]["href"], target)

    class LocalOverture(OvertureBuildingsSource):
        def fetch(self, tmp: Path) -> int:
            df = buildings_frame()
            flat = df.drop(columns=["bbox"]).assign(
                xmin=[b["xmin"] for b in df["bbox"]], xmax=[b["xmax"] for b in df["bbox"]],
                ymin=[b["ymin"] for b in df["bbox"]], ymax=[b["ymax"] for b in df["bbox"]],
            )  # fmt: skip
            con = duckdb.connect()
            con.register("flat", flat)
            con.execute(
                "COPY (SELECT id, geometry, struct_pack(xmin := xmin, xmax := xmax, ymin := ymin, "
                "ymax := ymax) AS bbox, height, num_floors, subtype, class, src_dataset FROM flat) "
                f"TO '{tmp / self.FILE}' (FORMAT parquet)"
            )
            con.close()
            return len(df)

    cfg = load_a1_config()
    session.stac = stac_doc(write_rasters(raster_dir))
    srcs = make_sources(cfg, data_dir, session)  # type: ignore[arg-type]
    for s in srcs.values():
        s.client.sleep = lambda _s: None
    n, b = srcs["naip"], srcs["buildings"]
    srcs["naip"] = LocalNaip(data_dir, n.client, n.stac_url, n.collection, n.year, n.bbox)
    srcs["buildings"] = LocalOverture(data_dir, b.client, b.path, b.release, b.region, b.bbox)
    return srcs
