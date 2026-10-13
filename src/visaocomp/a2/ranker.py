"""ObjectRanker e ThresholdCalibrator: frequência semântica de objetos e o limiar de presença."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


class ObjectRanker:
    """Conta em quantas imagens cada descrição aparece (cosseno >= limiar) e ordena."""

    def __init__(self, descriptions: Sequence[str], threshold: float) -> None:
        self.descriptions = list(descriptions)
        self.threshold = float(threshold)

    @staticmethod
    def similarities(images: np.ndarray, texts: np.ndarray) -> np.ndarray:
        """Cosseno de cada imagem com cada descrição; os vetores já vêm normalizados (L2)."""
        return np.asarray(images, dtype=np.float32) @ np.asarray(texts, dtype=np.float32).T

    def rank(self, sims: np.ndarray, source: Sequence[str]) -> pd.DataFrame:
        """Tabela ordenada por frequência: contagem, score médio e a frequência por origem."""
        if sims.shape[1] != len(self.descriptions):
            raise ValueError(f"{sims.shape[1]} colunas para {len(self.descriptions)} descrições")
        source_arr = np.asarray(list(source))
        present = sims >= self.threshold
        top1 = sims.argmax(axis=1)
        rows = []
        for j, desc in enumerate(self.descriptions):
            mask = present[:, j]
            row: dict[str, Any] = {
                "description": desc,
                "n_present": int(mask.sum()),
                "frequency": float(mask.mean()),
                "mean_score": float(sims[:, j].mean()),
                "mean_score_present": float(sims[mask, j].mean()) if mask.any() else float("nan"),
                "n_top1": int((top1 == j).sum()),
            }
            for origin in ("ad", "pessoal"):
                sel = source_arr == origin
                row[f"frequency_{origin}"] = float(mask[sel].mean()) if sel.any() else float("nan")
            rows.append(row)
        df = pd.DataFrame(rows).sort_values(
            ["frequency", "mean_score"], ascending=[False, False], kind="stable"
        )
        df.insert(0, "rank", range(1, len(df) + 1))
        return df.reset_index(drop=True)

    def examples(self, sims: np.ndarray, description: str, k: int) -> list[tuple[int, float]]:
        """Os `k` índices de imagem com maior score na descrição (acima do limiar, se houver)."""
        j = self.descriptions.index(description)
        order = np.argsort(-sims[:, j], kind="stable")
        above = [int(i) for i in order if sims[i, j] >= self.threshold][:k]
        pick = above if len(above) == k else [int(i) for i in order[:k]]
        return [(i, float(sims[i, j])) for i in pick]


class ThresholdCalibrator:
    """Escolhe o limiar a partir da distribuição das similaridades e de pares conferidos a olho."""

    PERCENTILE_EDGES = (75.0, 90.0, 95.0, 98.0, 99.5, 100.0)

    def __init__(self, seed: int, n_pairs: int) -> None:
        self.seed = seed
        self.n_pairs = n_pairs

    @staticmethod
    def distribution(sims: np.ndarray) -> dict[str, float]:
        """Média, desvio e percentis de todos os pares (imagem, descrição)."""
        flat = sims.reshape(-1)
        pct = {f"p{p}": float(np.percentile(flat, p)) for p in (1, 10, 25, 50, 75, 90, 95, 99)}
        return {
            "n_pairs": int(flat.size),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            **pct,
        }

    def candidate_pairs(
        self, sims: np.ndarray, image_paths: Sequence[str], descriptions: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Sorteia pares espalhados entre os percentis 75 e 100, no máximo uma vez por imagem."""
        rng = np.random.default_rng(self.seed)
        edges = np.percentile(sims, self.PERCENTILE_EDGES)
        n_bins = len(edges) - 1
        per_bin = [
            self.n_pairs // n_bins + (1 if b < self.n_pairs % n_bins else 0) for b in range(n_bins)
        ]
        used: set[int] = set()
        out: list[dict[str, Any]] = []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            inside = (sims >= lo) & (sims <= hi if b == n_bins - 1 else sims < hi)
            cells = np.argwhere(inside)
            cells = cells[rng.permutation(len(cells))]
            taken = 0
            for i, j in cells if per_bin[b] else []:
                if int(i) in used:
                    continue
                used.add(int(i))
                out.append(
                    {
                        "image_path": image_paths[int(i)],
                        "description": descriptions[int(j)],
                        "score": float(sims[i, j]),
                    }
                )
                taken += 1
                if taken == per_bin[b]:
                    break
        if len(out) < self.n_pairs:  # faixa com poucas imagens distintas: completa da faixa alta
            cells = np.argwhere(sims >= edges[0])
            for i, j in cells[rng.permutation(len(cells))]:
                if len(out) == self.n_pairs:
                    break
                if int(i) not in used:
                    used.add(int(i))
                    out.append(
                        {
                            "image_path": image_paths[int(i)],
                            "description": descriptions[int(j)],
                            "score": float(sims[i, j]),
                        }
                    )
        return sorted(out, key=lambda p: p["score"])

    @staticmethod
    def choose(scores: Sequence[float], present: Sequence[bool]) -> dict[str, Any]:
        """Limiar que mais concorda com os rótulos (empate: o mais alto). Devolve a curva toda."""
        s = np.asarray(scores, dtype=float)
        y = np.asarray(present, dtype=bool)
        if len(s) == 0 or len(s) != len(y):
            raise ValueError("preciso de pares rotulados: mesmo número de scores e de rótulos")
        curve = []
        for thr in sorted(set(s.tolist())):
            pred = s >= thr
            tp = int((pred & y).sum())
            curve.append(
                {
                    "threshold": float(thr),
                    "accuracy": float((pred == y).mean()),
                    "precision": tp / int(pred.sum()) if pred.any() else float("nan"),
                    "recall": tp / int(y.sum()) if y.any() else float("nan"),
                }
            )
        best = max(c["accuracy"] for c in curve)
        chosen = [c for c in curve if c["accuracy"] == best][-1]
        # o limiar fica no meio entre o score escolhido e o vizinho abaixo (não cola no dado)
        below = [c["threshold"] for c in curve if c["threshold"] < chosen["threshold"]]
        thr = (chosen["threshold"] + below[-1]) / 2 if below else chosen["threshold"]
        pred = s >= thr
        tp = int((pred & y).sum())
        return {
            "threshold": float(thr),
            "accuracy": float((pred == y).mean()),
            "precision": tp / int(pred.sum()) if pred.any() else float("nan"),
            "recall": tp / int(y.sum()) if y.any() else float("nan"),
            "n_labeled": int(len(s)),
            "n_present": int(y.sum()),
            "curve": curve,
        }
