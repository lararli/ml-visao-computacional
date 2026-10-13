"""RunDir: a pasta outputs/<run_id>/ de cada execução, com manifest.json."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visaocomp.paths import repo_root


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    """Commit atual e se há alterações não commitadas. Sem git ou sem commit: 'sem-commit'."""
    cwd = cwd or repo_root()

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (subprocess.SubprocessError, OSError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit or "sem-commit", "dirty": bool(status)}


class RunDir:
    """Cria outputs/<run_id>/ (nunca reaproveita uma pasta) e grava o manifesto."""

    def __init__(self, outputs_root: Path, stage: str, run_id: str | None = None) -> None:
        self.stage = stage
        self.outputs_root = Path(outputs_root)
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        base = run_id or f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{stage}"
        self.run_id = base
        self.path = self.outputs_root / base
        n = 1
        while True:
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                n += 1
                self.run_id = f"{base}-{n}"
                self.path = self.outputs_root / self.run_id
        self.created_at = datetime.now(UTC).isoformat(timespec="seconds")

    def write_json(self, name: str, payload: Any) -> Path:
        """Grava um JSON na pasta da execução (falha se o arquivo já existe)."""
        target = self.path / name
        if target.exists():
            raise FileExistsError(f"{target} já existe; uma execução nunca sobrescreve")
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def write_manifest(
        self,
        sources: list[dict[str, Any]],
        counts: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Grava manifest.json: fonte, versão, sha256, contagens e commit do git."""
        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "stage": self.stage,
            "created_at": self.created_at,
            "git": git_info(),
            "sources": sources,
            "counts": counts or {},
        }
        manifest.update(extra or {})
        return self.write_json("manifest.json", manifest)


def latest_manifest(outputs_root: Path, stage: str) -> dict[str, Any] | None:
    """Lê o manifest.json mais recente de uma etapa, ou None se não houver."""
    candidates = sorted(Path(outputs_root).glob("*/manifest.json"))
    for path in reversed(candidates):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("stage") == stage:
            return doc
    return None
