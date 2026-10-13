"""Fontes de dados: download idempotente com checksum, extração em raw/ e verificação."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import requests

from visaocomp.config import SourceConfig

log = logging.getLogger(__name__)

CHUNK = 1024 * 1024
MARKER = ".extracted.json"
READY_MSG = "já existe e passa no checksum"


class SourceChangedError(RuntimeError):
    """O checksum baixado é diferente do gravado: a fonte mudou."""


class DownloadFailedError(RuntimeError):
    """O download falhou depois de todas as tentativas."""


@dataclass(frozen=True)
class VerifyResult:
    """Resultado de `verify()`: ok, o motivo quando não, e o que foi medido."""

    ok: bool
    reason: str
    sha256: str | None = None
    n_files: int = 0
    n_bytes: int = 0


def sha256_file(path: Path) -> str:
    """Calcula o sha256 de um arquivo, lendo em blocos."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


class DataSource(ABC):
    """Uma fonte de dados: baixa, verifica e entrega o caminho do dado bruto (raw/)."""

    def __init__(self, cfg: SourceConfig, data_dir: Path) -> None:
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.dest_dir = self.data_dir / cfg.dest
        self.archive_path = self.data_dir / "raw" / "_archives" / f"{cfg.name}.zip"

    @property
    def marker_path(self) -> Path:
        """Arquivo que marca a extração concluída (checksum e contagens)."""
        return self.dest_dir / MARKER

    @abstractmethod
    def download(self, force: bool = False) -> bool:
        """Baixa e extrai a fonte. Devolve True se baixou, False se já estava pronta."""

    @abstractmethod
    def verify(self) -> VerifyResult:
        """Confere o arquivo baixado e a pasta extraída contra o checksum do config."""

    @abstractmethod
    def load(self) -> Path:
        """Devolve a pasta do dado bruto, depois de verificar."""

    def describe(self) -> dict[str, Any]:
        """Dados da fonte para o manifesto: fonte, versão, sha256 e contagens."""
        result = self.verify()
        return {
            "name": self.cfg.name,
            "activity": self.cfg.activity,
            "kind": self.cfg.kind,
            "source": f"{self.cfg.owner}/{self.cfg.dataset}",
            "version": self.cfg.version,
            "url": self.cfg.url,
            "dest": self.cfg.dest,
            "sha256": result.sha256,
            "n_files": result.n_files,
            "n_bytes": result.n_bytes,
            "verified": result.ok,
        }


class KaggleSource(DataSource):
    """Dataset do Kaggle pelo endpoint público, sem credencial."""

    def __init__(
        self,
        cfg: SourceConfig,
        data_dir: Path,
        session: requests.Session | None = None,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> None:
        super().__init__(cfg, data_dir)
        self.session = session or requests.Session()
        self.retries = retries
        self.backoff = backoff
        self.new_sha256: str | None = None

    # -- verify ---------------------------------------------------------------------------
    def verify(self) -> VerifyResult:
        """Confere sha256 do .zip, o marcador de extração e a contagem de arquivos."""
        if not self.archive_path.is_file():
            return VerifyResult(False, "arquivo .zip não existe")
        if not self.marker_path.is_file():
            return VerifyResult(False, "pasta extraída não existe ou está incompleta")
        sha = sha256_file(self.archive_path)
        if self.cfg.sha256 is None:
            return VerifyResult(False, "config sem sha256 gravado", sha256=sha)
        if sha != self.cfg.sha256:
            return VerifyResult(False, f"sha256 do .zip difere do config ({sha})", sha256=sha)
        marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        files = [p for p in self.dest_dir.rglob("*") if p.is_file() and p.name != MARKER]
        if marker.get("sha256") != sha or len(files) != marker.get("n_files"):
            return VerifyResult(False, "pasta extraída não bate com o marcador", sha256=sha)
        n_bytes = sum(p.stat().st_size for p in files)
        return VerifyResult(True, "ok", sha256=sha, n_files=len(files), n_bytes=n_bytes)

    # -- download -------------------------------------------------------------------------
    def download(self, force: bool = False) -> bool:
        """Baixa (com checksum) e extrai. Pula se já existe e passa no checksum."""
        name = self.cfg.name
        if not force:
            result = self.verify()
            if result.ok:
                log.info("%s: %s (%d arquivos)", name, READY_MSG, result.n_files)
                return False
            log.info("%s: precisa baixar (%s)", name, result.reason)
        part = self.archive_path.with_suffix(".zip.part")
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        sha = self._fetch_with_retries(part)
        if self.cfg.sha256 is not None and sha != self.cfg.sha256:
            part.unlink(missing_ok=True)
            raise SourceChangedError(
                f"{name}: a fonte mudou. Esperado sha256 {self.cfg.sha256}, veio {sha}. "
                "Nada foi sobrescrito. Se a mudança é esperada, apague o sha256 do config."
            )
        if self.cfg.sha256 is None:
            self.new_sha256 = sha
            self.cfg = replace(self.cfg, sha256=sha)
            log.info("%s: primeiro download, sha256 %s", name, sha)
        part.replace(self.archive_path)
        n_files = self._extract(sha)
        log.info("%s: baixado e extraído, %d arquivos", name, n_files)
        return True

    def _fetch_with_retries(self, part: Path) -> str:
        """Tenta o download `retries` vezes; descarta o arquivo parcial a cada falha."""
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            part.unlink(missing_ok=True)
            try:
                return self._fetch(part)
            except (requests.RequestException, OSError, DownloadFailedError) as exc:
                part.unlink(missing_ok=True)
                last = exc
                log.warning(
                    "%s: tentativa %d/%d falhou (%s); parcial descartado",
                    self.cfg.name, attempt, self.retries, exc,
                )  # fmt: skip
                if attempt < self.retries:
                    time.sleep(self.backoff * attempt)
        raise DownloadFailedError(f"{self.cfg.name}: download falhou {self.retries} vezes: {last}")

    def _fetch(self, part: Path) -> str:
        """Baixa em streaming para `part` e devolve o sha256. Confere o tamanho anunciado."""
        digest = hashlib.sha256()
        written = 0
        next_log = 256 * CHUNK
        with self.session.get(self.cfg.url, stream=True, timeout=(30, 120)) as resp:
            resp.raise_for_status()
            expected = resp.headers.get("Content-Length")
            encoded = resp.headers.get("Content-Encoding")
            log.info("%s: baixando %s bytes", self.cfg.name, expected or "?")
            with open(part, "wb") as fh:
                for chunk in resp.iter_content(CHUNK):
                    fh.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if written >= next_log:
                        log.info("%s: %d MB", self.cfg.name, written // CHUNK)
                        next_log += 256 * CHUNK
        if expected and not encoded and written != int(expected):
            raise DownloadFailedError(f"truncado: {written} de {expected} bytes")
        if not zipfile.is_zipfile(part):
            raise DownloadFailedError("o arquivo baixado não é um .zip")
        return digest.hexdigest()

    def _extract(self, sha: str) -> int:
        """Extrai o .zip em dest_dir (via pasta temporária) e grava o marcador."""
        tmp = self.dest_dir.with_name(self.dest_dir.name + ".extracting")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        root = tmp.resolve()
        n_files = 0
        with zipfile.ZipFile(self.archive_path) as zf:
            for info in zf.infolist():
                target = (tmp / info.filename).resolve()
                if root != target and root not in target.parents:
                    raise DownloadFailedError(f"caminho suspeito no .zip: {info.filename}")
                if not info.is_dir():
                    n_files += 1
            zf.extractall(tmp)
        if self.dest_dir.exists():
            shutil.rmtree(self.dest_dir)
        tmp.rename(self.dest_dir)
        marker = {"sha256": sha, "n_files": n_files, "version": self.cfg.version}
        self.marker_path.write_text(json.dumps(marker), encoding="utf-8")
        return n_files

    # -- load -----------------------------------------------------------------------------
    def load(self) -> Path:
        """Verifica e devolve a pasta com os arquivos brutos."""
        result = self.verify()
        if not result.ok:
            raise FileNotFoundError(f"{self.cfg.name}: {result.reason}. Rode `make data`.")
        return self.dest_dir

    def files(self) -> list[Path]:
        """Lista os arquivos extraídos (sem o marcador), em ordem."""
        return sorted(p for p in self.load().rglob("*") if p.is_file() and p.name != MARKER)


def build_source(
    cfg: SourceConfig, data_dir: Path, session: requests.Session | None = None
) -> DataSource:
    """Cria a fonte certa para o `kind` do config."""
    if cfg.kind == "kaggle":
        return KaggleSource(cfg, data_dir, session=session)
    raise ValueError(f"kind desconhecido: {cfg.kind}")
