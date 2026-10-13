"""AlignmentProbe: mede, sem treinar nada, se imagem e texto do CLIP moram no mesmo espaço."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from visaocomp.a2.embedder import Embedder, l2_normalize


class AlignmentProbe:
    """Classifica os anúncios pela pasta só com o nome da categoria e mede o espaço."""

    PROMPT = "an advertisement for {}"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    @staticmethod
    def topk_accuracy(sims: np.ndarray, truth: np.ndarray, k: int) -> float:
        """Fração de linhas em que a classe verdadeira está entre as `k` de maior similaridade."""
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        return float((order == truth[:, None]).any(axis=1).mean())

    def run(
        self,
        image_vecs: np.ndarray,
        meta: pd.DataFrame,
        category_map: Mapping[str, str],
        text_vecs: np.ndarray,
        seed: int,
    ) -> dict[str, Any]:
        """Zero-shot por categoria (só anúncios) e a geometria do espaço (cosseno por par)."""
        folders = sorted(category_map, key=int)
        names = [category_map[f] for f in folders]
        cat_vecs = self.embedder.embed_texts([self.PROMPT.format(n) for n in names])
        is_ad = (meta["source"] == "ad").to_numpy()
        ad_vecs = image_vecs[is_ad]
        truth = np.array([folders.index(f) for f in meta.loc[is_ad, "folder"]])
        sims = ad_vecs @ cat_vecs.T
        top1 = self.topk_accuracy(sims, truth, 1)
        top5 = self.topk_accuracy(sims, truth, 5)
        pred = sims.argmax(axis=1)
        per_cat = {
            names[c]: float((pred[truth == c] == c).mean())
            for c in range(len(names))
            if (truth == c).any()
        }
        matched = float(sims[np.arange(len(truth)), truth].mean())
        mismatched = float(
            (sims.sum() - sims[np.arange(len(truth)), truth].sum()) / (sims.size - len(truth))
        )

        rng = np.random.default_rng(seed)
        n = len(image_vecs)
        a, b = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
        keep = a != b
        img_img = float((image_vecs[a[keep]] * image_vecs[b[keep]]).sum(1).mean())
        txt_txt = float(
            ((text_vecs @ text_vecs.T).sum() - len(text_vecs))
            / (len(text_vecs) ** 2 - len(text_vecs))
        )
        img_txt = float((image_vecs @ text_vecs.T).mean())
        gap = float(
            np.linalg.norm(l2_normalize(image_vecs.mean(0)) - l2_normalize(text_vecs.mean(0)))
        )
        return {
            "n_ads": int(len(truth)),
            "n_categories": len(names),
            "zero_shot_top1": top1,
            "zero_shot_top5": top5,
            "chance_top1": 1 / len(names),
            "chance_top5": 5 / len(names),
            "cos_image_vs_true_category_text": matched,
            "cos_image_vs_other_category_text": mismatched,
            "cos_image_image": img_img,
            "cos_text_text": txt_txt,
            "cos_image_text": img_txt,
            "modality_gap": gap,
            "per_category_top1": per_cat,
        }
