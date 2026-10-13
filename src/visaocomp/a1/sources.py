"""Fontes públicas da A1: NAIP, lotes e endereços (ArcGIS), prédios (Overture), piscinas (OSM)
e licenças (Socrata). Cada consulta grava a resposta crua e um _done.json; rodar de novo não
consulta a rede."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from visaocomp.config import SourceConfig
from visaocomp.data.sources import DataSource, DownloadFailedError, VerifyResult, sha256_file

log = logging.getLogger(__name__)

DONE = "_done.json"
SKIP_MSG = "consulta já gravada, não consulta de novo"


@dataclass(frozen=True)
class Area:
    """Caixa da área de estudo em WGS84 (lon_min, lat_min, lon_max, lat_max)."""

    name: str
    bbox: tuple[float, float, float, float]
    margin_deg: float = 0.0

    @property
    def padded(self) -> tuple[float, float, float, float]:
        """A caixa com a folga, para pegar o que está na borda."""
        m = self.margin_deg
        x0, y0, x1, y1 = self.bbox
        return (round(x0 - m, 6), round(y0 - m, 6), round(x1 + m, 6), round(y1 + m, 6))


def query_key(query: dict[str, Any]) -> str:
    """Chave estável de uma consulta: sha1 do JSON ordenado (12 caracteres)."""
    blob = json.dumps(query, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


class RetryingClient:
    """Faz a chamada, valida a resposta e repete com espera; nunca devolve resposta de erro."""

    def __init__(
        self,
        session: requests.Session | None = None,
        user_agent: str = "visaocomp",
        retries: int = 5,
        backoff_s: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.session = session or requests.Session()
        self.user_agent = user_agent
        self.retries = retries
        self.backoff_s = backoff_s
        self.sleep = sleep

    def call(
        self,
        method: str,
        url: str,
        validate: Callable[[requests.Response], None],
        **kwargs: Any,
    ) -> requests.Response:
        """Chama `method url`; `validate` levanta ValueError se a resposta é erro ou limite."""
        headers = {"User-Agent": self.user_agent, **kwargs.pop("headers", {})}
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.request(method, url, headers=headers, timeout=300, **kwargs)
                if resp.status_code != 200:
                    raise ValueError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                validate(resp)
                return resp
            except (requests.RequestException, ValueError) as exc:
                last = exc
                log.warning("%s: tentativa %d/%d falhou (%s)", url, attempt, self.retries, exc)
                if attempt < self.retries:
                    self.sleep(self.backoff_s * attempt)
        raise DownloadFailedError(f"{url}: falhou {self.retries} vezes: {last}")


def _json_ok(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError as exc:
        raise ValueError(f"resposta não é JSON: {resp.text[:200]}") from exc


def validate_arcgis(resp: requests.Response) -> None:
    """ArcGIS devolve 200 com {'error': ...} quando falha: isso não é dado."""
    doc = _json_ok(resp)
    if not isinstance(doc, dict) or "error" in doc or "features" not in doc:
        raise ValueError(f"ArcGIS respondeu erro: {str(doc)[:200]}")


def validate_overpass(resp: requests.Response) -> None:
    """Overpass devolve 200 com 'remark' de erro ou timeout: isso não é dado."""
    doc = _json_ok(resp)
    remark = str(doc.get("remark", "")) if isinstance(doc, dict) else ""
    if not isinstance(doc, dict) or "elements" not in doc:
        raise ValueError(f"Overpass sem 'elements': {str(doc)[:200]}")
    if "error" in remark.lower() or "timed out" in remark.lower():
        raise ValueError(f"Overpass com erro: {remark[:200]}")


def validate_socrata(resp: requests.Response) -> None:
    """Socrata devolve uma lista; um dict é mensagem de erro."""
    doc = _json_ok(resp)
    if not isinstance(doc, list):
        raise ValueError(f"Socrata respondeu erro: {str(doc)[:200]}")


def validate_stac(resp: requests.Response) -> None:
    """A busca STAC devolve uma FeatureCollection."""
    doc = _json_ok(resp)
    if not isinstance(doc, dict) or "features" not in doc:
        raise ValueError(f"STAC respondeu erro: {str(doc)[:200]}")


class A1Source(DataSource):
    """Uma consulta pública da A1: grava as páginas cruas em raw/a1/<fonte>/<chave>/."""

    name: str = "a1"
    kind: str = "a1"

    def __init__(self, data_dir: Path, client: RetryingClient) -> None:
        cfg = SourceConfig(
            name=self.name, activity="A1", kind=self.kind, owner="", dataset="",
            version=None, sha256=None, dest=f"raw/a1/{self.name}",
        )  # fmt: skip
        super().__init__(cfg, data_dir)
        self.client = client

    # -- o que cada fonte define ------------------------------------------------------------
    def query(self) -> dict[str, Any]:
        """A consulta, como dado: URL e parâmetros. Define a chave da pasta."""
        raise NotImplementedError

    def fetch(self, tmp: Path) -> int:
        """Baixa as páginas para `tmp` e devolve o número de registros."""
        raise NotImplementedError

    # -- comum ------------------------------------------------------------------------------
    @property
    def key(self) -> str:
        """Chave da consulta (sha1 curto)."""
        return query_key(self.query())

    @property
    def folder(self) -> Path:
        """Pasta da consulta em raw/."""
        return self.dest_dir / self.key

    @property
    def marker_path(self) -> Path:
        return self.folder / DONE

    def marker(self) -> dict[str, Any]:
        """O _done.json da consulta (consulta, data, contagem, arquivos e sha256)."""
        return json.loads(self.marker_path.read_text(encoding="utf-8"))

    def verify(self) -> VerifyResult:
        """Confere o _done.json e o sha256 de cada arquivo gravado."""
        if not self.marker_path.is_file():
            return VerifyResult(False, "consulta ainda não gravada")
        doc = self.marker()
        n_bytes = 0
        for name, sha in doc["files"].items():
            path = self.folder / name
            if not path.is_file() or sha256_file(path) != sha:
                return VerifyResult(False, f"arquivo {name} ausente ou alterado")
            n_bytes += path.stat().st_size
        digest = hashlib.sha256(json.dumps(doc["files"], sort_keys=True).encode()).hexdigest()
        return VerifyResult(True, "ok", sha256=digest, n_files=len(doc["files"]), n_bytes=n_bytes)

    def download(self, force: bool = False) -> bool:
        """Faz a consulta e grava; pula se o _done.json existe e os arquivos conferem."""
        if not force and self.verify().ok:
            log.info("%s: %s (%s)", self.name, SKIP_MSG, self.key)
            return False
        tmp = self.folder.with_name(self.key + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        started = time.monotonic()
        n_records = self.fetch(tmp)
        files = {p.name: sha256_file(p) for p in sorted(tmp.iterdir()) if p.is_file()}
        doc = {
            "source": self.name,
            "query": self.query(),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_records": n_records,
            "seconds": round(time.monotonic() - started, 1),
            "files": files,
        }
        (tmp / DONE).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.folder.exists():
            shutil.rmtree(self.folder)
        tmp.rename(self.folder)
        log.info("%s: %d registros gravados em %s", self.name, n_records, self.folder)
        return True

    def load(self) -> Path:
        """Pasta com as respostas cruas, depois de verificar."""
        result = self.verify()
        if not result.ok:
            raise FileNotFoundError(f"{self.name}: {result.reason}. Rode `make data`.")
        return self.folder

    def describe(self) -> dict[str, Any]:
        """Linha do manifesto: consulta, data, contagem e checksum dos arquivos."""
        result = self.verify()
        doc = self.marker() if result.ok else {}
        return {
            "name": self.name,
            "activity": "A1",
            "kind": self.kind,
            "query": self.query(),
            "key": self.key,
            "fetched_at": doc.get("fetched_at"),
            "n_records": doc.get("n_records"),
            "sha256": result.sha256,
            "n_files": result.n_files,
            "n_bytes": result.n_bytes,
            "verified": result.ok,
        }

    def pages(self) -> list[Any]:
        """Lê as páginas JSON gravadas, em ordem."""
        folder = self.load()
        return [
            json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("page_*.json"))
        ]


class ArcgisLayerSource(A1Source):
    """Camada do ArcGIS Online da prefeitura (lotes ou endereços), paginada por resultOffset."""

    kind = "arcgis"

    def __init__(
        self,
        name: str,
        data_dir: Path,
        client: RetryingClient,
        url: str,
        fields: list[str],
        order_by: str,
        bbox: tuple[float, float, float, float],
        page_size: int = 2000,
    ) -> None:
        self.name = name
        self.url, self.fields, self.order_by = url, list(fields), order_by
        self.bbox, self.page_size = tuple(bbox), page_size
        super().__init__(data_dir, client)

    def params(self, offset: int) -> dict[str, Any]:
        """Parâmetros da página que começa em `offset`."""
        return {
            "where": "1=1",
            "geometry": ",".join(str(v) for v in self.bbox),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(self.fields),
            "outSR": 4326,
            "orderByFields": self.order_by,
            "resultOffset": offset,
            "resultRecordCount": self.page_size,
            "f": "geojson",
        }

    def query(self) -> dict[str, Any]:
        return {"url": self.url, "params": self.params(0) | {"resultOffset": "<paginado>"}}

    def fetch(self, tmp: Path) -> int:
        total, offset, page = 0, 0, 1
        while True:
            resp = self.client.call("GET", self.url, validate_arcgis, params=self.params(offset))
            doc = resp.json()
            feats = doc["features"]
            (tmp / f"page_{page:04d}.json").write_bytes(resp.content)
            total += len(feats)
            more = doc.get("exceededTransferLimit") or doc.get("properties", {}).get(
                "exceededTransferLimit"
            )
            if not feats or (len(feats) < self.page_size and not more):
                return total
            offset += len(feats)
            page += 1

    def features(self) -> list[dict[str, Any]]:
        """Todas as feições GeoJSON das páginas."""
        return [f for doc in self.pages() for f in doc["features"]]


class OverpassSource(A1Source):
    """Piscinas do OpenStreetMap (leisure=swimming_pool) com data fixa da consulta."""

    name = "osm_pools"
    kind = "overpass"

    def __init__(
        self,
        data_dir: Path,
        client: RetryingClient,
        urls: list[str],
        bbox: tuple[float, float, float, float],
        date: str,
        timeout_s: int = 180,
    ) -> None:
        self.urls, self.bbox, self.date, self.timeout_s = list(urls), tuple(bbox), date, timeout_s
        self.answered_by: str | None = None
        super().__init__(data_dir, client)

    def ql(self) -> str:
        """A consulta Overpass QL (sul, oeste, norte, leste)."""
        x0, y0, x1, y1 = self.bbox
        return (
            f'[out:json][timeout:{self.timeout_s}][date:"{self.date}"];'
            f'(way["leisure"="swimming_pool"]({y0},{x0},{y1},{x1});'
            f'relation["leisure"="swimming_pool"]({y0},{x0},{y1},{x1}););'
            "out geom;"
        )

    def query(self) -> dict[str, Any]:
        # o espelho que respondeu não muda a consulta: a chave depende só da QL e da data
        return {"service": "overpass", "data": self.ql()}

    def fetch(self, tmp: Path) -> int:
        last: Exception | None = None
        for url in self.urls:  # servidor principal primeiro; espelho quando ele está sem carga
            try:
                resp = self.client.call("POST", url, validate_overpass, data={"data": self.ql()})
            except DownloadFailedError as exc:
                last = exc
                continue
            self.answered_by = url
            (tmp / "page_0001.json").write_bytes(resp.content)
            (tmp / "answered_by.txt").write_text(url + "\n", encoding="utf-8")
            return len(resp.json()["elements"])
        raise DownloadFailedError(f"overpass: nenhum servidor respondeu: {last}")

    def elements(self) -> list[dict[str, Any]]:
        """Os elementos (ways e relations) da resposta."""
        return [e for doc in self.pages() for e in doc["elements"]]


class SocrataPermitsSource(A1Source):
    """Licenças de obra de Austin (Socrata 3syk-w9eu): POOL ou SOLAR, residenciais, até 2021."""

    name = "permits"
    kind = "socrata"
    WORDS = ("POOL", "SOLAR", "PHOTOVOLTAIC", " PV ")

    def __init__(
        self,
        data_dir: Path,
        client: RetryingClient,
        url: str,
        bbox: tuple[float, float, float, float],
        fields: list[str],
        issued_before: str,
        permit_class: str = "Residential",
        page_size: int = 5000,
    ) -> None:
        self.url, self.bbox, self.fields = url, tuple(bbox), list(fields)
        self.issued_before, self.permit_class, self.page_size = (
            issued_before, permit_class, page_size,
        )  # fmt: skip
        super().__init__(data_dir, client)

    def where(self) -> str:
        """O filtro $where (SoQL)."""
        x0, y0, x1, y1 = self.bbox
        words = " OR ".join(f"upper(description) like '%{w}%'" for w in self.WORDS)
        return (
            f"({words}) AND permit_class_mapped='{self.permit_class}' "
            f"AND issue_date < '{self.issued_before}' AND tcad_id IS NOT NULL "
            f"AND latitude between {y0} and {y1} AND longitude between {x0} and {x1}"
        )

    def params(self, offset: int) -> dict[str, Any]:
        """Parâmetros de uma página."""
        return {
            "$select": ",".join(self.fields),
            "$where": self.where(),
            "$order": "permit_number",
            "$limit": self.page_size,
            "$offset": offset,
        }

    def query(self) -> dict[str, Any]:
        return {"url": self.url, "params": self.params(0) | {"$offset": "<paginado>"}}

    def fetch(self, tmp: Path) -> int:
        total, offset, page = 0, 0, 1
        while True:
            resp = self.client.call("GET", self.url, validate_socrata, params=self.params(offset))
            rows = resp.json()
            (tmp / f"page_{page:04d}.json").write_bytes(resp.content)
            total += len(rows)
            if len(rows) < self.page_size:
                return total
            offset += len(rows)
            page += 1

    def rows(self) -> list[dict[str, Any]]:
        """Todas as licenças das páginas."""
        return [r for doc in self.pages() for r in doc]


class OvertureBuildingsSource(A1Source):
    """Contornos de prédios do Overture Maps (S3 anônimo), baixados uma vez por caixa."""

    name = "buildings"
    kind = "overture"
    FILE = "buildings.parquet"

    def __init__(
        self,
        data_dir: Path,
        client: RetryingClient,
        path: str,
        release: str,
        region: str,
        bbox: tuple[float, float, float, float],
    ) -> None:
        self.path, self.release, self.region, self.bbox = path, release, region, tuple(bbox)
        super().__init__(data_dir, client)

    def sql(self, target: Path) -> str:
        """A consulta DuckDB: prédios cuja caixa cruza a área."""
        x0, y0, x1, y1 = self.bbox
        return (
            "COPY (SELECT id, geometry, bbox, height, num_floors, subtype, class, "
            "sources[1].dataset AS src_dataset "
            f"FROM read_parquet('{self.path}', hive_partitioning=1) "
            f"WHERE bbox.xmax >= {x0} AND bbox.xmin <= {x1} "
            f"AND bbox.ymax >= {y0} AND bbox.ymin <= {y1}) "
            f"TO '{target}' (FORMAT parquet)"
        )

    def query(self) -> dict[str, Any]:
        return {"release": self.release, "path": self.path, "bbox": list(self.bbox)}

    def fetch(self, tmp: Path) -> int:
        import duckdb

        target = tmp / self.FILE
        con = duckdb.connect()
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            # acesso anônimo: o bucket do Overture é público
            con.execute(
                f"CREATE SECRET overture (TYPE s3, PROVIDER config, REGION '{self.region}')"
            )
            con.execute(self.sql(target))
            return con.execute(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0]
        finally:
            con.close()

    def parquet(self) -> Path:
        """O parquet local com os prédios da caixa."""
        return self.load() / self.FILE


class NaipSource(A1Source):
    """NAIP do Planetary Computer: busca STAC e leitura por janela do COG assinado (uma vez)."""

    name = "naip"
    kind = "stac"
    ITEMS = "page_0001.json"

    def __init__(
        self,
        data_dir: Path,
        client: RetryingClient,
        stac_url: str,
        collection: str,
        year: int,
        bbox: tuple[float, float, float, float],
    ) -> None:
        self.stac_url, self.collection, self.year, self.bbox = (
            stac_url, collection, year, tuple(bbox),
        )  # fmt: skip
        super().__init__(data_dir, client)

    def search_body(self) -> dict[str, Any]:
        """Corpo da busca STAC."""
        return {
            "collections": [self.collection],
            "bbox": list(self.bbox),
            "datetime": f"{self.year}-01-01/{self.year}-12-31",
            "limit": 100,
        }

    def query(self) -> dict[str, Any]:
        return {"url": self.stac_url, "body": self.search_body()}

    def fetch(self, tmp: Path) -> int:
        resp = self.client.call("POST", self.stac_url, validate_stac, json=self.search_body())
        (tmp / self.ITEMS).write_bytes(resp.content)
        items = sorted(resp.json()["features"], key=lambda f: f["id"])
        for item in items:
            self._read_window(item, tmp / f"{item['id']}.tif")
        return len(items)

    def _read_window(self, item: dict[str, Any], target: Path) -> None:
        """Lê do COG só a janela que cobre a área (as 4 bandas) e grava um GeoTIFF local."""
        import planetary_computer
        import rasterio
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds

        href = planetary_computer.sign(item["assets"]["image"]["href"])
        env = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "GDAL_HTTP_MULTIRANGE": "YES",
            "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
            "GDAL_HTTP_MAX_RETRY": "5",
            "GDAL_HTTP_RETRY_DELAY": "3",
        }
        with rasterio.Env(**env), rasterio.open(href) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *self.bbox)
            win = from_bounds(*bounds, transform=src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            win = win.round_offsets().round_lengths()
            data = src.read(window=win)
            profile = src.profile | {
                "driver": "GTiff",
                "width": int(win.width),
                "height": int(win.height),
                "transform": src.window_transform(win),
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(data)
        log.info("naip: %s janela %dx%d gravada", item["id"], int(win.width), int(win.height))

    def items(self) -> list[dict[str, Any]]:
        """Itens NAIP gravados: id, data, CRS e caminho do GeoTIFF local."""
        folder = self.load()
        doc = json.loads((folder / self.ITEMS).read_text(encoding="utf-8"))
        out = []
        for f in sorted(doc["features"], key=lambda f: f["id"]):
            out.append(
                {
                    "item_id": f["id"],
                    "datetime": f["properties"].get("datetime"),
                    "gsd": f["properties"].get("gsd"),
                    "epsg": f["properties"].get("proj:epsg"),
                    "bbox": f["bbox"],
                    "path": folder / f"{f['id']}.tif",
                }
            )
        return out


@dataclass(frozen=True)
class A1Config:
    """O configs/a1.yaml inteiro (área, fontes e regras)."""

    area: Area
    user_agent: str
    retries: int
    backoff_s: float
    sources: dict[str, dict[str, Any]]
    trades: dict[str, dict[str, str]]
    house: dict[str, Any]
    chip: dict[str, Any]
    ndvi: dict[str, float]
    split: dict[str, Any]
    review: dict[str, Any]
    path: Path


def load_a1_config(path: Path | None = None) -> A1Config:
    """Lê o configs/a1.yaml."""
    import yaml

    from visaocomp.paths import repo_root

    path = Path(path) if path else repo_root() / "configs" / "a1.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    a = doc["area"]
    return A1Config(
        area=Area(a["name"], tuple(a["bbox"]), float(a.get("margin_deg", 0.0))),
        user_agent=doc["user_agent"],
        retries=int(doc.get("retries", 5)),
        backoff_s=float(doc.get("backoff_s", 5)),
        sources=doc["sources"],
        trades=doc["trades"],
        house=doc["house"],
        chip=doc["chip"],
        ndvi=doc["ndvi"],
        split=doc["split"],
        review=doc["review"],
        path=path,
    )


def make_sources(
    cfg: A1Config, data_dir: Path, session: requests.Session | None = None
) -> dict[str, A1Source]:
    """Cria as 6 consultas da A1 a partir do config."""
    client = RetryingClient(session, cfg.user_agent, cfg.retries, cfg.backoff_s)
    s, box, pad = cfg.sources, cfg.area.bbox, cfg.area.padded
    return {
        "naip": NaipSource(
            data_dir, client, s["naip"]["stac_url"], s["naip"]["collection"],
            int(s["naip"]["year"]), pad,
        ),
        "parcels": ArcgisLayerSource(
            "parcels", data_dir, client, s["parcels"]["url"], s["parcels"]["fields"],
            s["parcels"]["order_by"], box, int(s["parcels"]["page_size"]),
        ),
        "addresses": ArcgisLayerSource(
            "addresses", data_dir, client, s["addresses"]["url"], s["addresses"]["fields"],
            s["addresses"]["order_by"], pad, int(s["addresses"]["page_size"]),
        ),
        "buildings": OvertureBuildingsSource(
            data_dir, client, s["buildings"]["path"], str(s["buildings"]["release"]),
            s["buildings"]["region"], pad,
        ),
        "osm_pools": OverpassSource(
            data_dir, client, s["osm_pools"]["urls"], pad, str(s["osm_pools"]["date"]),
            int(s["osm_pools"]["timeout_s"]),
        ),
        "permits": SocrataPermitsSource(
            data_dir, client, s["permits"]["url"], pad, s["permits"]["fields"],
            str(s["permits"]["issued_before"]), s["permits"]["permit_class"],
            int(s["permits"]["page_size"]),
        ),
    }  # fmt: skip
