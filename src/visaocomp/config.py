"""Leitura do configs/data.yaml e gravação do sha256 da primeira execução."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from visaocomp.paths import repo_root


@dataclass(frozen=True)
class SourceConfig:
    """Uma fonte de dados do config."""

    name: str
    activity: str
    kind: str
    owner: str
    dataset: str
    version: int | None
    sha256: str | None
    dest: str

    @property
    def url(self) -> str:
        """URL do endpoint público de download, com a versão fixada quando houver."""
        base = f"https://www.kaggle.com/api/v1/datasets/download/{self.owner}/{self.dataset}"
        return f"{base}?datasetVersionNumber={self.version}" if self.version else base


@dataclass(frozen=True)
class DataConfig:
    """O config de dados inteiro."""

    catalog: str
    schemas: tuple[str, ...]
    sources: tuple[SourceConfig, ...]
    path: Path

    def source(self, name: str) -> SourceConfig:
        """Devolve a fonte pelo nome."""
        for src in self.sources:
            if src.name == name:
                return src
        raise KeyError(f"fonte desconhecida: {name}")


def default_config_path() -> Path:
    """Caminho padrão do configs/data.yaml."""
    return repo_root() / "configs" / "data.yaml"


def load_data_config(path: Path | None = None) -> DataConfig:
    """Lê o YAML e valida os campos obrigatórios."""
    path = Path(path) if path else default_config_path()
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources = []
    for item in doc["sources"]:
        sha = item.get("sha256")
        sources.append(
            SourceConfig(
                name=item["name"],
                activity=item.get("activity", ""),
                kind=item["kind"],
                owner=item["owner"],
                dataset=item["dataset"],
                version=item.get("version"),
                sha256=str(sha).lower() if sha else None,
                dest=item["dest"],
            )
        )
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"nomes de fonte repetidos em {path}")
    return DataConfig(
        catalog=doc.get("catalog", "catalog.duckdb"),
        schemas=tuple(doc.get("schemas", ("bronze", "silver", "gold"))),
        sources=tuple(sources),
        path=path,
    )


def record_sha256(path: Path, name: str, sha256: str) -> None:
    """Grava o sha256 na linha `sha256:` da fonte `name`, sem mexer nos comentários."""
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    in_block = False
    for i, line in enumerate(lines):
        if re.match(r"\s*-\s*name:\s*", line):
            in_block = re.match(rf"\s*-\s*name:\s*{re.escape(name)}\s*$", line) is not None
        elif in_block and re.match(r"\s+sha256:", line):
            indent = re.match(r"\s+", line).group(0)  # type: ignore[union-attr]
            lines[i] = f"{indent}sha256: {sha256}\n"
            Path(path).write_text("".join(lines), encoding="utf-8")
            return
    raise KeyError(f"não achei a linha sha256 da fonte {name} em {path}")
