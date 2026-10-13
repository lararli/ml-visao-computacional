"""Recortes da A1 lidos do catálogo (só leitura), contrato dos dados e máscaras de região."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
REQUIRED = ("chip_id", "path", "split", "x_center", "y_center", "epsg", "house_wkt")


class DataContractError(ValueError):
    """Os dados não batem com o que o treino espera."""


class ChipCatalogReader:
    """Lê recorte, split, rótulos e conferência do catalog.duckdb pela chave chip_id."""

    def __init__(self, catalog_path: Path, attributes: tuple[str, ...]) -> None:
        self.catalog_path = Path(catalog_path)
        self.attributes = attributes

    def _sql(self) -> str:
        cols = []
        for a in self.attributes:
            for col, alias in (
                ("label", a),
                ("reason", f"{a}_reason"),
                ("first_check", f"{a}_check"),
            ):
                cols.append(f"max(CASE WHEN l.attribute = '{a}' THEN l.{col} END) AS {alias}")
        return f"""
            SELECT c.chip_id, c.house_id, c.path, c.split, c.x_center, c.y_center, c.epsg,
                   h.geom_wkt AS house_wkt, {", ".join(cols)}
            FROM silver.chip c
            JOIN silver.house h USING (house_id)
            LEFT JOIN silver.label l USING (chip_id)
            GROUP BY ALL
            ORDER BY c.chip_id
        """  # noqa: S608 (atributos vêm do config, não do usuário)

    def read(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Devolve (recortes com rótulos, piscinas do OSM em WKT lon/lat)."""
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"catálogo não existe: {self.catalog_path}. Rode `make data`.")
        con = duckdb.connect(str(self.catalog_path), read_only=True)
        try:
            chips = con.execute(self._sql()).df()
            pools = con.execute("SELECT osm_id, geom_wkt FROM silver.osm_pool").df()
        finally:
            con.close()
        self.validate(chips)
        log.info(
            "catálogo: %d recortes (%s)",
            len(chips),
            ", ".join(f"{s} {int((chips['split'] == s).sum())}" for s in SPLITS),
        )
        return chips, pools

    def validate(self, chips: pd.DataFrame) -> None:
        """Contrato: colunas, splits, rótulos em {0, 1, vazio} e vazio sempre com motivo."""
        missing = [c for c in (*REQUIRED, *self.attributes) if c not in chips.columns]
        if missing:
            raise DataContractError(f"faltam colunas {missing}; vieram {list(chips.columns)}")
        bad = set(chips["split"].dropna()) - set(SPLITS)
        if bad or chips["split"].isna().any():
            raise DataContractError(f"split esperado em {SPLITS}; veio {bad or 'vazio'}")
        for s in SPLITS:
            if not (chips["split"] == s).any():
                raise DataContractError(f"o split {s} está vazio")
        for a in self.attributes:
            vals = set(chips[a].dropna().astype(int))
            if not vals <= {0, 1}:
                raise DataContractError(f"rótulo de {a} esperado em {{0, 1}}; veio {vals}")
            if not (chips.loc[chips["split"] == "train", a] == 1).any():
                raise DataContractError(f"{a}: nenhum positivo no treino")
            no_reason = chips[a].isna() & chips[f"{a}_reason"].isna()
            if no_reason.any():
                raise DataContractError(f"{a}: {int(no_reason.sum())} rótulos vazios sem motivo")


def label_matrix(chips: pd.DataFrame, attributes: tuple[str, ...]) -> np.ndarray:
    """Rótulos (N, A) em float: 1, 0, ou NaN quando vazio (não entra na perda nem na métrica)."""
    return chips[list(attributes)].astype("float64").to_numpy(dtype=np.float32)


def check_matrix(chips: pd.DataFrame, attributes: tuple[str, ...]) -> np.ndarray:
    """Conferência a olho (N, A): 'sim' = 1, 'não' = 0; 'não dá' ou sem conferência = NaN."""
    out = np.full((len(chips), len(attributes)), np.nan, dtype=np.float32)
    for j, a in enumerate(attributes):
        col = chips[f"{a}_check"].fillna("")
        out[col.eq("sim").to_numpy(), j] = 1.0
        out[col.eq("não").to_numpy(), j] = 0.0
    return out


def load_images(paths: list[str], root: Path, size: int) -> torch.Tensor:
    """Lê os PNG pelo caminho gravado no catálogo; (N, 3, size, size) uint8. Para cedo se falhar."""
    from PIL import Image

    out = torch.empty((len(paths), 3, size, size), dtype=torch.uint8)
    for i, rel in enumerate(paths):
        path = Path(root) / rel
        try:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise DataContractError(f"recorte ilegível: {path} ({exc})") from exc
        if arr.shape != (size, size, 3):
            raise DataContractError(f"recorte {path}: esperado {size}x{size}x3, veio {arr.shape}")
        out[i] = torch.from_numpy(arr.copy()).permute(2, 0, 1)
    return out


def augment(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Giro de 0/90/180/270 graus, espelho e brilho leve, sorteados por imagem (no dispositivo)."""
    b = x.size(0)
    dev = x.device
    k = torch.randint(0, 4, (b,), generator=gen).to(dev)
    flip = (torch.rand(b, generator=gen) < 0.5).to(dev)
    out = x
    for r in range(1, 4):
        sel = (k == r).view(b, 1, 1, 1)
        out = torch.where(sel, torch.rot90(x, r, dims=(2, 3)), out)
    out = torch.where(flip.view(b, 1, 1, 1), out.flip(3), out)
    scale = (1.0 + 0.2 * (torch.rand(b, 1, 1, 1, generator=gen) - 0.5)).to(dev)
    return out * scale


@dataclass
class RegionMasks:
    """Máscaras (N, grid, grid) com a fração de cada patch coberta pelo telhado e pela piscina."""

    roof: np.ndarray
    pool: np.ndarray


class RegionMasker:
    """Põe o contorno da casa e as piscinas do OSM na grade de patches de cada recorte."""

    def __init__(self, size: int, patch: int, pixel_m: float) -> None:
        self.size = size
        self.patch = patch
        self.pixel_m = pixel_m

    def _grid(self, mask: np.ndarray) -> np.ndarray:
        g = self.size // self.patch
        return mask.reshape(g, self.patch, g, self.patch).mean(axis=(1, 3))

    def masks(self, chips: pd.DataFrame, pools: pd.DataFrame) -> RegionMasks:
        """Uma máscara por recorte; o recorte é centrado na casa (x_center, y_center)."""
        import shapely
        from affine import Affine
        from rasterio.features import rasterize
        from rasterio.warp import transform_geom
        from shapely.geometry import box, mapping, shape

        g = self.size // self.patch
        roof = np.zeros((len(chips), g, g), dtype=np.float32)
        pool = np.zeros_like(roof)
        half = (self.size / 2 + 0.5) * self.pixel_m  # o centro cai no pixel size/2
        by_epsg: dict[int, tuple] = {}
        for i, r in enumerate(chips.itertuples(index=False)):
            crs = f"EPSG:{int(r.epsg)}"
            if int(r.epsg) not in by_epsg:
                geoms = [shapely.from_wkt(w) for w in pools["geom_wkt"]]
                projected = [shape(transform_geom("EPSG:4326", crs, mapping(p))) for p in geoms]
                by_epsg[int(r.epsg)] = (projected, shapely.STRtree(projected))
            x0, y0 = r.x_center - half, r.y_center + half
            tf = Affine(self.pixel_m, 0, x0, 0, -self.pixel_m, y0)
            frame = box(x0, y0 - self.size * self.pixel_m, x0 + self.size * self.pixel_m, y0)
            if isinstance(r.house_wkt, str):
                house = transform_geom("EPSG:4326", crs, mapping(shapely.from_wkt(r.house_wkt)))
                m = rasterize([house], (self.size, self.size), transform=tf, all_touched=False)
                roof[i] = self._grid(m.astype(np.float32))
            projected, tree = by_epsg[int(r.epsg)]
            inside = [mapping(projected[j]) for j in tree.query(frame, predicate="intersects")]
            if inside:
                m = rasterize(inside, (self.size, self.size), transform=tf, all_touched=False)
                pool[i] = self._grid(m.astype(np.float32))
        return RegionMasks(roof=roof, pool=pool)
