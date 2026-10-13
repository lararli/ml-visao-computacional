"""Ingestão de dados: fontes (`DataSource`) e o orquestrador (`DataIngestor`)."""

from visaocomp.data.ingest import DataIngestor
from visaocomp.data.sources import (
    DataSource,
    DownloadFailedError,
    KaggleSource,
    SourceChangedError,
    VerifyResult,
    build_source,
)

__all__ = [
    "DataIngestor",
    "DataSource",
    "DownloadFailedError",
    "KaggleSource",
    "SourceChangedError",
    "VerifyResult",
    "build_source",
]
