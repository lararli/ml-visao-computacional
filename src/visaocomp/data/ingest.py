"""DataIngestor: baixa e verifica todas as fontes do config e grava o manifesto."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from visaocomp.catalog import Catalog
from visaocomp.config import DataConfig, record_sha256
from visaocomp.data.sources import DataSource, build_source
from visaocomp.run import RunDir

log = logging.getLogger(__name__)


class DataIngestor:
    """Roda as fontes do config em ordem, cria o catálogo e grava outputs/<run_id>/manifest.json."""

    def __init__(
        self,
        cfg: DataConfig,
        data_dir: Path,
        outputs_root: Path,
        session: requests.Session | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.outputs_root = Path(outputs_root)
        self.sources: list[DataSource] = [
            build_source(s, self.data_dir, session) for s in cfg.sources
        ]

    def run(self, force: bool = False) -> Path:
        """Baixa o que falta, verifica tudo, cria o catálogo e devolve o caminho do manifesto."""
        log.info("ingestão: início, %d fontes, dados em %s", len(self.sources), self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        for source in self.sources:
            if source.download(force=force):
                downloaded += 1
            new_sha = getattr(source, "new_sha256", None)
            if new_sha:
                record_sha256(self.cfg.path, source.cfg.name, new_sha)
                log.info("%s: sha256 gravado em %s", source.cfg.name, self.cfg.path.name)
                source.new_sha256 = None  # type: ignore[attr-defined]
        described = [s.describe() for s in self.sources]
        bad = [d["name"] for d in described if not d["verified"]]
        if bad:
            raise RuntimeError(f"fontes que não passaram na verificação: {', '.join(bad)}")
        with Catalog(self.data_dir / self.cfg.catalog, self.cfg.schemas) as catalog:
            schemas = catalog.ensure_schemas()
        run = RunDir(self.outputs_root, "data")
        counts = {
            "sources": len(described),
            "downloaded": downloaded,
            "skipped": len(described) - downloaded,
            "files": sum(d["n_files"] for d in described),
            "bytes": sum(d["n_bytes"] for d in described),
            "schemas": schemas,
        }
        manifest = run.write_manifest(described, counts)
        log.info(
            "ingestão: fim, %d baixadas, %d puladas, %d arquivos; manifesto %s",
            downloaded, counts["skipped"], counts["files"], manifest,
        )  # fmt: skip
        return manifest
