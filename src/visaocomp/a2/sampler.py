"""Ads16Sampler: confere a estrutura do ADS-16 e sorteia a amostra estratificada."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
EXPECTED_AD_FOLDERS = tuple(str(n) for n in range(1, 21))


class ContractError(ValueError):
    """A estrutura do ADS-16 não é a esperada (diz o que esperava e o que veio)."""


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Divide `n` entre os estratos em proporção ao tamanho (maior resto), sem passar do estrato."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(r) for k, r in raw.items()}
    leftover = n - sum(alloc.values())
    order = sorted(sizes, key=lambda k: (-(raw[k] - alloc[k]), k))
    for k in order[:leftover]:
        alloc[k] += 1
    return {k: min(alloc[k], sizes[k]) for k in sizes}


def stratified_sample(df: pd.DataFrame, column: str, n: int, seed: int) -> pd.DataFrame:
    """Amostra `n` linhas com alocação proporcional por `column`; mesma seed, mesma amostra."""
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    rng = np.random.default_rng(seed)
    groups = {k: g.sort_values("image_path") for k, g in df.groupby(column, sort=True)}
    alloc = allocate({k: len(g) for k, g in groups.items()}, n)
    parts = []
    for key in sorted(groups):
        g = groups[key]
        take = alloc[key]
        idx = rng.choice(len(g), size=take, replace=False) if take < len(g) else np.arange(len(g))
        parts.append(g.iloc[np.sort(idx)])
    return pd.concat(parts).sort_values("image_path").reset_index(drop=True)


class Ads16Sampler:
    """Lê a pasta do ADS-16, valida a estrutura e escolhe a amostra.

    O ADS-16 tem `Ads/<n>/` (n = 1..20, 15 anúncios cada) e `Corpus/U<id>/` (fotos pessoais).
    A chave estável de cada imagem é o caminho lógico (`Ads/3/7.png`), sem a pasta da parte.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._scan: pd.DataFrame | None = None

    def _parts(self) -> list[Path]:
        parts = sorted(self.root.glob("ADS16_Benchmark_part*/ADS16_Benchmark_part*"))
        if not parts:
            raise ContractError(f"esperava ADS16_Benchmark_part*/ em {self.root}; não achei")
        return parts

    def category_map(self) -> dict[str, str]:
        """Mapa pasta -> categoria: cabeçalho de `U0001-RT.csv` (Cat0..Cat19), na ordem."""
        for part in self._parts():
            for rt in sorted((part / "Corpus" / "Corpus").glob("U*/U*-RT.csv")):
                with rt.open(encoding="utf-8", newline="") as fh:
                    rows = list(csv.reader(fh, delimiter=";"))
                if len(rows) >= 2 and len(rows[1]) == 20:
                    ids = [c.removeprefix("Cat") for c in rows[0]]
                    if ids != [str(i) for i in range(20)]:
                        raise ContractError(f"esperava Cat0..Cat19; veio {rows[0]}")
                    return {str(i + 1): name for i, name in enumerate(rows[1])}
        raise ContractError("esperava um U*-RT.csv com 20 categorias no Corpus; não achei")

    def scan(self) -> pd.DataFrame:
        """Lista todas as imagens: anúncios (`ad`) e fotos pessoais do corpus (`pessoal`)."""
        if self._scan is not None:
            return self._scan
        cat = self.category_map()
        rows: list[dict[str, Any]] = []
        for part in self._parts():
            for folder_dir in sorted((part / "Ads" / "Ads").iterdir()):
                if not folder_dir.is_dir():
                    continue
                for img in sorted(folder_dir.iterdir()):
                    if img.suffix.lower() in IMAGE_SUFFIXES:
                        rows.append(
                            {
                                "image_path": f"Ads/{folder_dir.name}/{img.name}",
                                "abs_path": str(img),
                                "source": "ad",
                                "folder": folder_dir.name,
                                "category": cat.get(folder_dir.name, ""),
                                "stratum": f"ad/{folder_dir.name}",
                            }
                        )
            for user_dir in sorted((part / "Corpus" / "Corpus").glob("U*")):
                for sub in sorted(user_dir.glob("U*-IM-*")):
                    if not sub.is_dir():
                        continue
                    pol = sub.name.rsplit("-", 1)[-1]
                    for img in sorted(sub.iterdir()):
                        if img.suffix.lower() in IMAGE_SUFFIXES:
                            rows.append(
                                {
                                    "image_path": f"Corpus/{user_dir.name}/{sub.name}/{img.name}",
                                    "abs_path": str(img),
                                    "source": "pessoal",
                                    "folder": "",
                                    "category": "",
                                    "stratum": f"pessoal/{pol}",
                                }
                            )
        df = pd.DataFrame(rows)
        dup = df[df.duplicated("image_path", keep=False)]
        if not dup.empty:
            raise ContractError(
                f"caminhos lógicos repetidos entre as partes: {list(dup.image_path)[:3]}"
            )
        self._scan = df
        return df

    def validate(self) -> dict[str, Any]:
        """Contrato de entrada: pastas 1..20 com anúncios. Devolve contagens para o manifesto."""
        df = self.scan()
        ads = df[df.source == "ad"]
        per_folder = ads.groupby("folder").size().to_dict()
        missing = [f for f in EXPECTED_AD_FOLDERS if per_folder.get(f, 0) == 0]
        extra = sorted(set(per_folder) - set(EXPECTED_AD_FOLDERS))
        if missing or extra:
            raise ContractError(
                f"esperava as pastas de anúncios 1..20; faltam {missing}, a mais {extra}"
            )
        odd = {f: n for f, n in per_folder.items() if n != 15}
        report = {
            "n_ads": int(len(ads)),
            "n_pessoal": int((df.source == "pessoal").sum()),
            "ads_per_folder": {f: int(per_folder[f]) for f in EXPECTED_AD_FOLDERS},
            "folders_with_other_than_15": {f: int(n) for f, n in odd.items()},
        }
        log.info(
            "ADS-16: %d anúncios em 20 pastas, %d fotos pessoais; pastas fora de 15: %s",
            report["n_ads"],
            report["n_pessoal"],
            odd or "nenhuma",
        )
        return report

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Amostra de `n` imagens: todos os anúncios (por pasta) primeiro, o resto das fotos.

        O ADS-16 só tem ~300 anúncios e o PD pede 500+; o que falta vem das fotos pessoais,
        estratificadas por polaridade (POS/NEG).
        """
        df = self.scan()
        ads = df[df.source == "ad"]
        n_ads = min(n, len(ads))
        parts = [stratified_sample(ads, "stratum", n_ads, seed)]
        rest = n - n_ads
        if rest > 0:
            parts.append(stratified_sample(df[df.source == "pessoal"], "stratum", rest, seed))
        out = pd.concat(parts).sort_values("image_path").reset_index(drop=True)
        log.info(
            "amostra: %d imagens (%d anúncios, %d fotos pessoais), seed %d",
            len(out),
            int((out.source == "ad").sum()),
            int((out.source == "pessoal").sum()),
            seed,
        )
        return out
