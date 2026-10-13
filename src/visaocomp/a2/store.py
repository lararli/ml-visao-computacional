"""EmbeddingStore: a tabela silver.ads16_embedding no catalog.duckdb."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_TABLE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")


class EmbeddingStore:
    """Guarda o vetor de cada imagem (chave: `image_path` + `model`) e busca por cosseno.

    O `catalog.duckdb` é compartilhado com outras etapas: cada operação abre a conexão,
    faz o trabalho e fecha. Se outro processo segura o arquivo, espera e tenta de novo.
    """

    def __init__(
        self,
        db_path: Path,
        table: str = "silver.ads16_embedding",
        dim: int = 512,
        lock_attempts: int = 120,
        lock_wait: float = 2.0,
    ) -> None:
        if not _TABLE.match(table):
            raise ValueError(f"nome de tabela inválido: {table}")
        self.db_path = Path(db_path)
        self.table = table
        self.dim = dim
        self.lock_attempts = lock_attempts
        self.lock_wait = lock_wait
        self._ready = False

    @contextmanager
    def _session(self, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = None
        for attempt in range(1, self.lock_attempts + 1):
            try:
                con = duckdb.connect(str(self.db_path), read_only=read_only)
                break
            except duckdb.IOException as exc:
                if "lock" not in str(exc).lower() or attempt == self.lock_attempts:
                    raise
                log.info(
                    "catalog.duckdb ocupado; tentativa %d, espero %.0fs", attempt, self.lock_wait
                )
                time.sleep(self.lock_wait)
        assert con is not None
        try:
            yield con
        finally:
            con.close()

    def ensure_table(self) -> None:
        """Cria o schema e a tabela se faltarem (idempotente; uma vez por objeto)."""
        if self._ready:
            return
        schema = self.table.split(".")[0]
        with self._session() as con:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            con.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.table} (
                    image_path VARCHAR NOT NULL,
                    model VARCHAR NOT NULL,
                    source VARCHAR,
                    folder VARCHAR,
                    category VARCHAR,
                    vector FLOAT[{self.dim}] NOT NULL,
                    embedded_at TIMESTAMP DEFAULT current_timestamp,
                    PRIMARY KEY (image_path, model)
                )"""
            )
        self._ready = True

    def existing(self, paths: Sequence[str], model: str) -> set[str]:
        """Quais das `paths` já têm vetor guardado para este modelo."""
        self.ensure_table()
        frame = pd.DataFrame({"image_path": list(paths)})
        with self._session() as con:
            con.register("wanted", frame)
            rows = con.execute(
                f"SELECT t.image_path FROM {self.table} t JOIN wanted w USING (image_path) "
                "WHERE t.model = ?",
                [model],
            ).fetchall()
        return {r[0] for r in rows}

    def add(self, meta: pd.DataFrame, vectors: np.ndarray, model: str) -> int:
        """Grava os vetores novos; o que já existe é ignorado. Devolve quantos entraram."""
        if len(meta) != len(vectors):
            raise ValueError(f"{len(meta)} linhas de metadados para {len(vectors)} vetores")
        if len(meta) and vectors.shape[1] != self.dim:
            raise ValueError(f"esperava vetores de {self.dim} dimensões; vieram {vectors.shape[1]}")
        if not len(meta):
            return 0
        self.ensure_table()
        frame = meta[["image_path", "source", "folder", "category"]].copy()
        frame["model"] = model
        frame["vector"] = [v.astype(np.float32).tolist() for v in vectors]
        with self._session() as con:
            before = con.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
            con.register("incoming", frame)
            con.execute(
                f"INSERT INTO {self.table} (image_path, model, source, folder, category, vector) "
                f"SELECT image_path, model, source, folder, category, vector::FLOAT[{self.dim}] "
                "FROM incoming ON CONFLICT DO NOTHING"
            )
            after = con.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
        return int(after - before)

    def load(self, paths: Sequence[str], model: str) -> tuple[pd.DataFrame, np.ndarray]:
        """Metadados e matriz [n, dim] dos `paths` pedidos, na ordem de `paths`."""
        self.ensure_table()
        frame = pd.DataFrame({"image_path": list(paths), "pos": range(len(paths))})
        with self._session(read_only=True) as con:
            con.register("wanted", frame)
            df = con.execute(
                f"SELECT t.image_path, t.source, t.folder, t.category, t.vector, w.pos "
                f"FROM {self.table} t JOIN wanted w USING (image_path) WHERE t.model = ? "
                "ORDER BY w.pos",
                [model],
            ).df()
        matrix = (
            np.stack([np.asarray(v, dtype=np.float32) for v in df["vector"]])
            if len(df)
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        return df.drop(columns=["vector", "pos"]).reset_index(drop=True), matrix

    def search(
        self, query: np.ndarray, k: int, model: str, paths: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """As `k` imagens mais parecidas com o vetor (cosseno feito pelo DuckDB)."""
        self.ensure_table()
        q = np.asarray(query, dtype=np.float32).reshape(-1).tolist()
        sql = (
            f"SELECT t.image_path, t.source, t.folder, t.category, "
            f"array_cosine_similarity(t.vector, ?::FLOAT[{self.dim}]) AS score "
            f"FROM {self.table} t "
        )
        args: list = [q]
        with self._session(read_only=True) as con:
            if paths is not None:
                con.register("scope", pd.DataFrame({"image_path": list(paths)}))
                sql += "JOIN scope s USING (image_path) "
            sql += "WHERE t.model = ? ORDER BY score DESC, t.image_path LIMIT ?"
            args += [model, int(k)]
            return con.execute(sql, args).df()

    def count(self, model: str | None = None) -> int:
        """Quantos vetores a tabela tem (de um modelo, ou todos)."""
        self.ensure_table()
        with self._session(read_only=True) as con:
            if model is None:
                return int(con.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0])
            return int(
                con.execute(
                    f"SELECT count(*) FROM {self.table} WHERE model = ?", [model]
                ).fetchone()[0]
            )
