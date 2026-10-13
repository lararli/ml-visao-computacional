"""Leitura do configs/a2.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from visaocomp.paths import repo_root


@dataclass(frozen=True)
class Query:
    """Uma consulta textual e o nível de especificidade que ela testa."""

    id: str
    text: str
    nivel: str


@dataclass(frozen=True)
class A2Config:
    """Parâmetros da atividade 2 (CLIP no ADS-16)."""

    model: str
    batch_size: int
    seed: int
    sample_size: int
    embedding_table: str
    top_k: int
    examples_per_object: int
    n_threshold_check: int
    threshold_check_file: str
    templates: tuple[str, ...]
    descriptions: tuple[str, ...]
    queries: tuple[Query, ...]
    tokenizer_demo: dict[str, Any]
    path: Path

    def prompts(self, description: str) -> list[str]:
        """Aplica cada template à descrição (prompt ensembling)."""
        return [t.format(description) for t in self.templates]


def default_a2_path() -> Path:
    """Caminho padrão do configs/a2.yaml."""
    return repo_root() / "configs" / "a2.yaml"


def load_a2_config(path: Path | None = None) -> A2Config:
    """Lê o YAML e valida o que o PD exige: 20+ descrições e 8+ consultas."""
    path = Path(path) if path else default_a2_path()
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    descriptions = tuple(doc["descriptions"])
    queries = tuple(Query(q["id"], q["text"], q.get("nivel", "")) for q in doc["queries"])
    if len(descriptions) < 20 or len(set(descriptions)) != len(descriptions):
        raise ValueError("o PD pede 20+ descrições distintas")
    if len(queries) < 8:
        raise ValueError("o PD pede 8+ consultas")
    return A2Config(
        model=doc["model"],
        batch_size=int(doc["batch_size"]),
        seed=int(doc["seed"]),
        sample_size=int(doc["sample_size"]),
        embedding_table=doc["embedding_table"],
        top_k=int(doc["top_k"]),
        examples_per_object=int(doc["examples_per_object"]),
        n_threshold_check=int(doc["n_threshold_check"]),
        threshold_check_file=doc["threshold_check_file"],
        templates=tuple(doc["templates"]),
        descriptions=descriptions,
        queries=queries,
        tokenizer_demo=dict(doc["tokenizer_demo"]),
        path=path,
    )
