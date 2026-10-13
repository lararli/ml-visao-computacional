"""Catálogo DuckDB: abre o arquivo e cria os schemas bronze, silver e gold."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import TracebackType

import duckdb

log = logging.getLogger(__name__)

DEFAULT_SCHEMAS = ("bronze", "silver", "gold")
_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class Catalog:
    """O `catalog.duckdb` do projeto: camadas bronze, silver e gold."""

    def __init__(self, path: Path, schemas: tuple[str, ...] = DEFAULT_SCHEMAS) -> None:
        for name in schemas:
            if not _IDENT.match(name):
                raise ValueError(f"nome de schema inválido: {name}")
        self.path = Path(path)
        self.schemas = tuple(schemas)
        self._con: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Abre (e cria, se preciso) o arquivo do catálogo."""
        if self._con is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(self.path))
        return self._con

    def ensure_schemas(self) -> list[str]:
        """Cria os schemas que faltam (idempotente) e devolve os que existem."""
        con = self.connect()
        for name in self.schemas:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {name}")
        existing = self.list_schemas()
        log.info("catálogo %s: schemas %s", self.path.name, ", ".join(existing))
        return existing

    def list_schemas(self) -> list[str]:
        """Lista os schemas de camada que existem (bronze, silver, gold)."""
        rows = (
            self.connect().execute("SELECT schema_name FROM information_schema.schemata").fetchall()
        )
        present = {r[0] for r in rows}
        return [s for s in self.schemas if s in present]

    def close(self) -> None:
        """Fecha a conexão."""
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> Catalog:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
