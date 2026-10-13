"""TextSearch: busca de imagens por frase, com o cosseno feito dentro do DuckDB."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from visaocomp.a2.config import Query
from visaocomp.a2.embedder import Embedder
from visaocomp.a2.store import EmbeddingStore

log = logging.getLogger(__name__)


class TextSearch:
    """Transforma a consulta em vetor e pede ao DuckDB as `k` imagens mais parecidas."""

    def __init__(
        self,
        embedder: Embedder,
        store: EmbeddingStore,
        scope: Sequence[str] | None = None,
        k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.scope = list(scope) if scope is not None else None
        self.k = k

    def query(self, text: str) -> list[dict[str, Any]]:
        """Top-k de uma consulta: posição, caminho, score, origem e categoria."""
        vec = self.embedder.embed_texts([text])[0]
        found = self.store.search(vec, self.k, self.embedder.model_name, self.scope)
        return [
            {
                "rank": i + 1,
                "image_path": r.image_path,
                "score": float(r.score),
                "source": r.source,
                "category": r.category or "",
            }
            for i, r in enumerate(found.itertuples())
        ]

    def run_all(self, queries: Sequence[Query]) -> list[dict[str, Any]]:
        """Roda todas as consultas e devolve o conteúdo do queries.json."""
        out = [
            {"id": q.id, "text": q.text, "nivel": q.nivel, "results": self.query(q.text)}
            for q in queries
        ]
        log.info("busca: %d consultas, top-%d cada", len(out), self.k)
        return out
