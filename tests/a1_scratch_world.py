"""Um catálogo minúsculo da A1 para os testes do ViT do zero: recortes, rótulos, casas e piscinas.

Piscina = um quadrado azul a leste da casa; solar = um retângulo escuro no telhado. Sem rede.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shapely
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape

EPSG = 26914
X0, Y0 = 620_000.0, 3_355_000.0
PIXEL_M = 0.6


def _wgs(poly: shapely.Polygon) -> str:
    return shape(transform_geom(f"EPSG:{EPSG}", "EPSG:4326", mapping(poly))).wkt


def build_world(root: Path, n: int = 30, size: int = 96) -> Path:
    """Grava chips PNG em root/processed/chips e o catalog.duckdb; devolve o caminho do catálogo."""
    from PIL import Image

    rng = np.random.default_rng(0)
    chips_dir = root / "processed" / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    chips, houses, labels, pools = [], [], [], []
    splits = ["train"] * (n * 6 // 10) + ["val"] * (n * 2 // 10)
    splits += ["test"] * (n - len(splits))
    for i in range(n):
        cid, hid = f"chip{i:03d}", f"house{i:03d}"
        x, y = X0 + 200.0 * i, Y0
        pool, solar = i % 2, int(i % 3 == 0)
        img = rng.integers(90, 140, (size, size, 3), dtype=np.uint8)
        img[40:56, 40:56] = (150, 120, 100)  # telhado
        if solar:
            img[44:52, 42:54] = (20, 20, 40)
        if pool:
            img[44:52, 70:78] = (40, 120, 230)
            pools.append(
                {"osm_id": f"way/{i}", "geom_wkt": _wgs(box(x + 13, y - 2.4, x + 17.8, y + 2.4))}
            )
        Image.fromarray(img).save(chips_dir / f"{cid}.png")
        chips.append({"chip_id": cid, "house_id": hid, "path": f"processed/chips/{cid}.png",
                      "split": splits[i], "x_center": x, "y_center": y, "epsg": EPSG})  # fmt: skip
        houses.append({"house_id": hid, "geom_wkt": _wgs(box(x - 4.8, y - 4.8, x + 4.8, y + 4.8))})
        empty_pool = i == 4  # um rótulo vazio, com motivo
        check = splits[i] == "test"
        solar_check = "não dá" if i == n - 1 else "sim" if solar else "não"
        labels += [
            {"chip_id": cid, "attribute": "pool", "label": None if empty_pool else pool,
             "reason": "arvore" if empty_pool else None,
             "first_check": ("sim" if pool else "não") if check else None},
            {"chip_id": cid, "attribute": "solar", "label": solar, "reason": None,
             "first_check": solar_check if check else None},
        ]  # fmt: skip
    cat = root / "catalog.duckdb"
    con = duckdb.connect(str(cat))
    con.execute("CREATE SCHEMA silver")
    for name, rows in (("chip", chips), ("house", houses), ("label", labels), ("osm_pool", pools)):
        df = pd.DataFrame(rows)
        if name == "label":
            df["label"] = df["label"].astype("Int64")
        con.register("df", df)
        con.execute(f"CREATE TABLE silver.{name} AS SELECT * FROM df")
        con.unregister("df")
    con.close()
    return cat
