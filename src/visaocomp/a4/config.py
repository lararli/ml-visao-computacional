"""Leitura do configs/a4.yaml em dataclasses tipadas."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from visaocomp.paths import repo_root


@dataclass(frozen=True)
class DedupConfig:
    """Parâmetros do hash perceptual."""

    hash_size: int = 8
    max_distance: int = 4


@dataclass(frozen=True)
class CaseConfig:
    """Como o caso do PD é remontado a partir do dataset."""

    seed: int
    test_per_class: int
    train_counts: dict[str, int]
    val_fraction: float


@dataclass(frozen=True)
class GanRunConfig:
    """Uma variante de treino da GAN (sem ou com mitigação)."""

    name: str
    label_smoothing: float = 1.0
    spectral_norm: bool = False
    replay_size: int = 0
    replay_prob: float = 0.0

    @property
    def strategies(self) -> list[str]:
        """Lista legível das mitigações ligadas."""
        out = []
        if self.label_smoothing < 1.0:
            out.append(f"label smoothing ({self.label_smoothing})")
        if self.spectral_norm:
            out.append("spectral norm no discriminador")
        if self.replay_size > 0 and self.replay_prob > 0:
            out.append(f"replay buffer ({self.replay_size}, p={self.replay_prob})")
        return out


@dataclass(frozen=True)
class GanConfig:
    """Hiperparâmetros comuns da GAN condicional."""

    seed: int
    image_size: int
    nz: int
    ngf: int
    ndf: int
    epochs: int
    batch_size: int
    lr: float
    betas: tuple[float, float]
    sample_every: int
    fid_every: int
    fid_n: int
    runs: tuple[GanRunConfig, ...]


@dataclass(frozen=True)
class SyntheticConfig:
    """Quantas COVID sintéticas entram no treino."""

    n_covid: int
    seed: int


@dataclass(frozen=True)
class ClassifierConfig:
    """Classificador usado nas duas condições (sem e com sintético)."""

    seeds: tuple[int, ...]
    image_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    pretrained: bool
    augment: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class A4Config:
    """O config inteiro da A4."""

    source: str
    classes: tuple[str, ...]
    positive_class: str
    dedup: DedupConfig
    case: CaseConfig
    gan: GanConfig
    synthetic: SyntheticConfig
    classifier: ClassifierConfig
    raw: dict[str, Any]

    @property
    def positive_index(self) -> int:
        """Índice da classe positiva (COVID)."""
        return self.classes.index(self.positive_class)

    @property
    def sha256(self) -> str:
        """Checksum do config: duas execuções com o mesmo valor usaram o mesmo config."""
        text = json.dumps(self.raw, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Config como dicionário, para o manifesto."""
        out = asdict(self)
        out.pop("raw")
        return out

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> A4Config:
        """Monta o config a partir do YAML lido."""
        g = doc["gan"]
        runs = tuple(GanRunConfig(name=k, **v) for k, v in g["runs"].items())
        gan = GanConfig(
            **{k: v for k, v in g.items() if k not in ("runs", "betas")},
            betas=(float(g["betas"][0]), float(g["betas"][1])),
            runs=runs,
        )
        c = doc["classifier"]
        classifier = ClassifierConfig(
            **{k: v for k, v in c.items() if k != "seeds"}, seeds=tuple(c["seeds"])
        )
        classes = tuple(doc["classes"])
        case = CaseConfig(**doc["case"])
        if set(case.train_counts) != set(classes):
            raise ValueError(
                f"train_counts {sorted(case.train_counts)} difere das classes {classes}"
            )
        return cls(
            source=doc["source"],
            classes=classes,
            positive_class=doc["positive_class"],
            dedup=DedupConfig(**doc.get("dedup", {})),
            case=case,
            gan=gan,
            synthetic=SyntheticConfig(**doc["synthetic"]),
            classifier=classifier,
            raw=copy.deepcopy(doc),
        )


def default_a4_config_path() -> Path:
    """Caminho padrão do configs/a4.yaml."""
    return repo_root() / "configs" / "a4.yaml"


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Aplica `patch` sobre `base`, recursivo nos dicionários (devolve uma cópia)."""
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_a4_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> A4Config:
    """Lê o YAML da A4 e aplica sobrescritas (usadas pelo smoke)."""
    path = Path(path) if path else default_a4_config_path()
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if overrides:
        doc = deep_update(doc, overrides)
    return A4Config.from_dict(doc)
