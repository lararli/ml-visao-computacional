"""Leitura das imagens: varredura com contrato de dados, split estratificado e Dataset."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRecord:
    """Uma imagem aceita: caminho relativo à raiz, classe (índice) e sha256 do arquivo."""

    path: str
    label: int
    sha256: str


@dataclass
class ScanReport:
    """O que a varredura achou: classes, imagens aceitas e o que foi pulado, com a razão."""

    root: str
    classes: list[str]
    records: list[ImageRecord]
    found: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    duplicates: list[dict[str, str]] = field(default_factory=list)
    ignored_dirs: list[str] = field(default_factory=list)

    @property
    def kept_per_class(self) -> dict[str, int]:
        """Imagens aceitas por classe."""
        counts = dict.fromkeys(self.classes, 0)
        for rec in self.records:
            counts[self.classes[rec.label]] += 1
        return counts

    def counts(self) -> dict[str, int]:
        """Contagens para o manifesto: lidas, aceitas, puladas, duplicadas."""
        return {
            "files_found": self.found,
            "files_kept": len(self.records),
            "files_skipped_unreadable": len(self.skipped),
            "files_removed_duplicate": len(self.duplicates),
            "dirs_ignored": len(self.ignored_dirs),
        }


class ImageScanner:
    """Varre root/<classe>/*: confere que cada arquivo abre, tira idênticos e lista as classes."""

    def __init__(
        self,
        root: Path,
        extensions: tuple[str, ...] = (".bmp", ".jpg", ".jpeg", ".png"),
        dedupe: bool = True,
    ) -> None:
        self.root = Path(root)
        self.extensions = tuple(e.lower() for e in extensions)
        self.dedupe = dedupe

    def _class_files(self, folder: Path) -> list[Path]:
        return sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in self.extensions
        )

    def scan(self) -> ScanReport:
        """Varre as classes. Falha cedo se a raiz não tem nenhuma subpasta com imagens."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"pasta de imagens não existe: {self.root}")
        class_dirs: list[tuple[str, list[Path]]] = []
        ignored: list[str] = []
        for sub in sorted(p for p in self.root.iterdir() if p.is_dir()):
            files = self._class_files(sub)
            if files:
                class_dirs.append((sub.name, files))
            else:
                ignored.append(sub.name)  # ex.: a cópia aninhada data/data do dataset
        if not class_dirs:
            raise ValueError(
                f"nenhuma classe em {self.root}: esperava subpastas com imagens "
                f"({', '.join(self.extensions)}); achei {[p.name for p in self.root.iterdir()]}"
            )
        classes = [name for name, _ in class_dirs]
        report = ScanReport(str(self.root), classes, [], ignored_dirs=ignored)
        seen: dict[str, str] = {}
        for label, (_name, files) in enumerate(class_dirs):
            for path in files:
                report.found += 1
                rel = path.relative_to(self.root).as_posix()
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with Image.open(path) as img:
                        img.load()  # decodifica tudo: pega arquivo truncado
                except Exception as exc:  # noqa: BLE001 - qualquer falha de leitura vira registro
                    report.skipped.append({"path": rel, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                if self.dedupe and digest in seen:
                    report.duplicates.append({"path": rel, "same_as": seen[digest]})
                    continue
                seen[digest] = rel
                report.records.append(ImageRecord(rel, label, digest))
        log.info(
            "varredura de %s: %d classes, %d arquivos, %d aceitos, %d ilegíveis, %d duplicados",
            self.root, len(classes), report.found, len(report.records),
            len(report.skipped), len(report.duplicates),
        )  # fmt: skip
        return report


@dataclass(frozen=True)
class Split:
    """Índices (nas `records`) de treino, validação e teste."""

    train: list[int]
    val: list[int]
    test: list[int]


def stratified_split(labels: list[int], train_frac: float, val_frac: float, seed: int) -> Split:
    """Divide por classe com seed fixa; toda classe deixa pelo menos 1 imagem no treino."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(labels)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in sorted(set(labels)):
        idx = np.flatnonzero(arr == label)
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_test = min(round((1.0 - train_frac - val_frac) * n), n - 1)
        n_val = min(round(val_frac * n), n - 1 - n_test)
        test.extend(idx[:n_test].tolist())
        val.extend(idx[n_test : n_test + n_val].tolist())
        train.extend(idx[n_test + n_val :].tolist())
    return Split(sorted(train), sorted(val), sorted(test))


def build_transform(resize: int, crop: int) -> T.Compose:
    """Resize, recorte central e tensor em [0, 1]. A normalização mora dentro do modelo."""
    return T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
        ]
    )


class ImageFolderDataset(Dataset):
    """Entrega (tensor RGB, classe) para as imagens aceitas pela varredura."""

    def __init__(self, root: Path, records: list[ImageRecord], transform: T.Compose) -> None:
        self.root = Path(root)
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        """Número de imagens."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Abre a imagem, converte para RGB (PNG com alfa vira RGB) e aplica a transformação."""
        rec = self.records[index]
        with Image.open(self.root / rec.path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, rec.label

    def subset(self, indices: list[int]) -> ImageFolderDataset:
        """Novo Dataset só com os índices dados."""
        return ImageFolderDataset(self.root, [self.records[i] for i in indices], self.transform)
