"""Dados da A4: leitura das imagens, remoção de cópias por hash perceptual e montagem do caso."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from visaocomp.a4.config import CaseConfig

log = logging.getLogger(__name__)

IMAGE_EXT = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageRecord:
    """Uma imagem e o rótulo (índice da classe)."""

    path: Path
    label: int


def list_records(root: Path, classes: tuple[str, ...]) -> list[ImageRecord]:
    """Lista as imagens de root/<CLASSE>/, em ordem; para cedo se uma classe faltar."""
    records: list[ImageRecord] = []
    for label, name in enumerate(classes):
        folder = Path(root) / name
        if not folder.is_dir():
            raise FileNotFoundError(f"esperava a pasta da classe {name} em {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise ValueError(f"a classe {name} está vazia em {folder}")
        records.extend(ImageRecord(p, label) for p in files)
    return records


def load_gray(path: Path, size: int) -> np.ndarray:
    """Abre a imagem em tons de cinza, redimensiona para size x size e devolve uint8 (H, W)."""
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((size, size), Image.Resampling.BILINEAR)
            return np.asarray(img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"imagem ilegível: {path} ({exc})") from exc


class XrayDataset(Dataset):
    """Raio-X em tons de cinza: devolve (tensor [1, H, W] em [0, 1], rótulo)."""

    def __init__(self, records: list[ImageRecord], size: int) -> None:
        self.records = list(records)
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        arr = load_gray(rec.path, self.size)
        return torch.from_numpy(arr).unsqueeze(0).float() / 255.0, rec.label

    def stack(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Carrega tudo em memória: (uint8 [N, 1, H, W], rótulos long [N])."""
        arrs = [load_gray(r.path, self.size) for r in self.records]
        images = torch.from_numpy(np.stack(arrs)).unsqueeze(1)
        labels = torch.tensor([r.label for r in self.records], dtype=torch.long)
        return images, labels


def hamming_matrix_rows(rows: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    """Distância de Hamming entre cada hash de `rows` e todos os `hashes` (uint64)."""
    return np.bitwise_count(rows[:, None] ^ hashes[None, :]).astype(np.int16)


class _UnionFind:
    """União de conjuntos para agrupar cópias."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class DedupResult:
    """Resultado da remoção de cópias."""

    kept: list[ImageRecord]
    hashes: dict[str, int]
    report: dict[str, Any]


class Deduplicator:
    """Remove cópias e quase-cópias por hash perceptual (pHash), antes de qualquer split."""

    def __init__(
        self, hash_size: int = 8, max_distance: int = 4, cache_path: Path | None = None
    ) -> None:
        if hash_size != 8:
            raise ValueError("só hash_size=8 (64 bits) é suportado")
        self.hash_size = hash_size
        self.max_distance = max_distance
        self.cache_path = Path(cache_path) if cache_path else None

    def hash_one(self, path: Path) -> int:
        """pHash de 64 bits da imagem, como inteiro."""
        try:
            with Image.open(path) as img:
                return int(str(imagehash.phash(img.convert("L"), hash_size=self.hash_size)), 16)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"imagem ilegível: {path} ({exc})") from exc

    def hash_all(self, paths: list[Path]) -> np.ndarray:
        """pHash de todas as imagens, com cache em CSV (chave: caminho e tamanho)."""
        cache: dict[tuple[str, int], int] = {}
        if self.cache_path and self.cache_path.exists():
            df = pd.read_csv(self.cache_path, dtype={"path": str, "size": int, "phash": str})
            cache = {(r.path, r.size): int(r.phash, 16) for r in df.itertuples()}
        out, rows, misses = [], [], 0
        for p in paths:
            key = (str(p), Path(p).stat().st_size)
            if key not in cache:
                cache[key] = self.hash_one(p)
                misses += 1
            out.append(cache[key])
            rows.append({"path": key[0], "size": key[1], "phash": f"{cache[key]:016x}"})
        if self.cache_path and misses:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(self.cache_path, index=False)
        log.info(
            "pHash: %d imagens, %d calculadas, %d do cache", len(paths), misses, len(paths) - misses
        )
        return np.array(out, dtype=np.uint64)

    def near_pairs(self, hashes: np.ndarray, chunk: int = 512) -> list[tuple[int, int]]:
        """Pares (i, j), i < j, com distância de Hamming <= max_distance."""
        pairs: list[tuple[int, int]] = []
        for start in range(0, len(hashes), chunk):
            dist = hamming_matrix_rows(hashes[start : start + chunk], hashes)
            ii, jj = np.nonzero(dist <= self.max_distance)
            for i, j in zip(ii + start, jj, strict=True):
                if i < j:
                    pairs.append((int(i), int(j)))
        return pairs

    @staticmethod
    def groups(n: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
        """Agrupa os índices ligados por pares (cada grupo em ordem)."""
        uf = _UnionFind(n)
        for a, b in pairs:
            uf.union(a, b)
        found: dict[int, list[int]] = {}
        for i in range(n):
            found.setdefault(uf.find(i), []).append(i)
        return list(found.values())

    def run(self, records: list[ImageRecord]) -> DedupResult:
        """Mantém uma imagem por grupo de cópias; descarta grupos com rótulos diferentes."""
        hashes = self.hash_all([r.path for r in records])
        groups = self.groups(len(records), self.near_pairs(hashes))
        kept: list[ImageRecord] = []
        dup_removed, conflict_removed, conflict_groups = 0, 0, 0
        removed_by_class: dict[int, int] = {}
        for g in groups:
            labels = {records[i].label for i in g}
            if len(labels) > 1:
                conflict_groups += 1
                conflict_removed += len(g)
                for i in g:
                    removed_by_class[records[i].label] = (
                        removed_by_class.get(records[i].label, 0) + 1
                    )
                continue
            kept.append(records[g[0]])
            dup_removed += len(g) - 1
            if len(g) > 1:
                lab = records[g[0]].label
                removed_by_class[lab] = removed_by_class.get(lab, 0) + len(g) - 1
        kept.sort(key=lambda r: (r.label, str(r.path)))
        hash_map = {str(r.path): int(h) for r, h in zip(records, hashes, strict=True)}
        report = {
            "n_in": len(records),
            "n_out": len(kept),
            "copias_removidas": dup_removed,
            "grupos_com_copias": sum(1 for g in groups if len(g) > 1),
            "grupos_com_rotulos_diferentes": conflict_groups,
            "removidas_por_conflito_de_rotulo": conflict_removed,
            "removidas_por_classe": {int(k): int(v) for k, v in sorted(removed_by_class.items())},
            "hash_size": self.hash_size,
            "max_distance": self.max_distance,
        }
        log.info(
            "dedup: %d -> %d imagens (%d cópias, %d em grupos com rótulos diferentes)",
            len(records),
            len(kept),
            dup_removed,
            conflict_removed,
        )
        return DedupResult(kept=kept, hashes=hash_map, report=report)


@dataclass
class CaseSplit:
    """O caso remontado: uma linha por imagem, com a parte (test, train, val)."""

    frame: pd.DataFrame
    classes: tuple[str, ...]

    def records(self, split: str) -> list[ImageRecord]:
        """Imagens de uma parte, em ordem."""
        sub = self.frame[self.frame["split"] == split]
        return [
            ImageRecord(Path(p), int(lab)) for p, lab in zip(sub["path"], sub["label"], strict=True)
        ]

    def counts(self) -> dict[str, dict[str, int]]:
        """Contagem por parte e classe."""
        out: dict[str, dict[str, int]] = {}
        for split in ("train", "val", "test"):
            sub = self.frame[self.frame["split"] == split]
            out[split] = {c: int((sub["label"] == i).sum()) for i, c in enumerate(self.classes)}
        return out

    def save(self, path: Path) -> Path:
        """Grava o split em CSV (caminho, rótulo, classe, parte, pHash)."""
        self.frame.to_csv(path, index=False)
        return path

    @classmethod
    def load(cls, path: Path, classes: tuple[str, ...]) -> CaseSplit:
        """Lê um split gravado."""
        return cls(pd.read_csv(path, dtype={"phash": str}), classes)


class CaseBuilder:
    """Separa o teste real primeiro e depois sorteia o caso do PD (840/240/120) do resto."""

    def __init__(self, cfg: CaseConfig, classes: tuple[str, ...]) -> None:
        self.cfg = cfg
        self.classes = classes

    def build(self, records: list[ImageRecord], hashes: dict[str, int]) -> CaseSplit:
        """Devolve o split com teste, treino e validação, sem imagem repetida entre as partes."""
        rng = np.random.default_rng(self.cfg.seed)
        rows = []
        for label, name in enumerate(self.classes):
            pool = sorted((r for r in records if r.label == label), key=lambda r: str(r.path))
            need = self.cfg.test_per_class + self.cfg.train_counts[name]
            if len(pool) < need:
                raise ValueError(
                    f"classe {name}: esperava ao menos {need} imagens após dedup, veio {len(pool)}"
                )
            order = rng.permutation(len(pool))
            test_idx = order[: self.cfg.test_per_class]
            case_idx = order[self.cfg.test_per_class : need]
            n_val = int(round(len(case_idx) * self.cfg.val_fraction))
            parts = [("test", test_idx), ("val", case_idx[:n_val]), ("train", case_idx[n_val:])]
            for split, idx in parts:
                for i in idx:
                    rec = pool[int(i)]
                    rows.append(
                        {
                            "path": str(rec.path),
                            "label": label,
                            "classe": name,
                            "split": split,
                            "phash": f"{hashes[str(rec.path)]:016x}",
                        }
                    )
        frame = pd.DataFrame(rows)
        split = CaseSplit(frame, self.classes)
        log.info("caso: %s", split.counts())
        return split
