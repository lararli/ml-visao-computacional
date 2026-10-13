"""ClipEmbedder: imagens e textos viram vetores L2 no mesmo espaço do CLIP."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)


def pick_device() -> torch.device:
    """Dispositivo automático: cuda, depois mps, depois cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normaliza cada linha para norma 1 (produto escalar = cosseno). Linha zero fica zero."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


class Embedder(Protocol):
    """O que o resto do pacote precisa de um embedder (o CLIP real ou um falso nos testes)."""

    model_name: str

    def embed_images(self, paths: Sequence[Path]) -> tuple[np.ndarray, list[int], list[str]]:
        """Devolve (vetores das legíveis, posições legíveis, erros das ilegíveis)."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Devolve um vetor L2 por texto."""
        ...


def _features(out: Any) -> torch.Tensor:
    """`get_*_features` devolve um tensor (versões antigas) ou um objeto com `pooler_output`."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


class ClipEmbedder:
    """Carrega o CLIP uma vez e gera embeddings normalizados, em lotes e sem gradiente."""

    def __init__(
        self,
        model_name: str,
        device: torch.device | None = None,
        batch_size: int = 32,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or pick_device()
        self.batch_size = batch_size
        self._model = model
        self._processor = processor

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            from transformers import CLIPModel, CLIPProcessor

            log.info("carregando %s em %s", self.model_name, self.device)
            self._model = CLIPModel.from_pretrained(self.model_name).eval().to(self.device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._model, self._processor

    @property
    def processor(self) -> Any:
        """O `CLIPProcessor` (tokenizador e pré-processamento de imagem)."""
        return self._load()[1]

    @property
    def model(self) -> Any:
        """O `CLIPModel`."""
        return self._load()[0]

    @staticmethod
    def open_image(path: Path) -> Image.Image:
        """Abre e converte para RGB; levanta se o arquivo estiver corrompido."""
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")

    def embed_images(self, paths: Sequence[Path]) -> tuple[np.ndarray, list[int], list[str]]:
        """Vetores das imagens legíveis; as ilegíveis são puladas e contadas, sem parar."""
        model, processor = self._load()
        vectors: list[np.ndarray] = []
        ok: list[int] = []
        errors: list[str] = []
        for start in range(0, len(paths), self.batch_size):
            images, idx = [], []
            for i in range(start, min(start + self.batch_size, len(paths))):
                try:
                    images.append(self.open_image(Path(paths[i])))
                    idx.append(i)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    errors.append(f"{paths[i]}: {exc}")
            if not images:
                continue
            pixel = processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            with torch.no_grad():
                feats = _features(model.get_image_features(pixel_values=pixel))
            vectors.append(l2_normalize(feats.float().cpu().numpy()))
            ok.extend(idx)
        out = np.concatenate(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
        return out, ok, errors

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Vetores dos textos, com padding até o maior do lote e attention mask."""
        model, processor = self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            enc = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                feats = _features(model.get_text_features(**enc))
            out.append(l2_normalize(feats.float().cpu().numpy()))
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)

    def embed_prompt_groups(self, groups: Sequence[Sequence[str]]) -> np.ndarray:
        """Um vetor por grupo de frases: média dos vetores e nova normalização (ensembling)."""
        flat = [p for g in groups for p in g]
        vecs = self.embed_texts(flat)
        out, pos = [], 0
        for g in groups:
            out.append(vecs[pos : pos + len(g)].mean(axis=0))
            pos += len(g)
        return l2_normalize(np.stack(out))
