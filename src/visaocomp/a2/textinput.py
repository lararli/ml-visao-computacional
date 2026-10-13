"""TextInputComparison: as mesmas frases nos tokenizadores do CLIP e do BERT, com a máscara."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from visaocomp.a2.embedder import ClipEmbedder, l2_normalize


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((l2_normalize(a) * l2_normalize(b)).sum())


class TextInputComparison:
    """Mostra a saída real dos dois tokenizadores e mede o efeito de tirar a attention mask."""

    CLIP_LEN = 77

    def __init__(self, embedder: ClipEmbedder, bert_name: str) -> None:
        self.embedder = embedder
        self.bert_name = bert_name

    def _clip_tokens(self, sentences: Sequence[str]) -> list[dict[str, Any]]:
        tok = self.embedder.processor.tokenizer
        enc = tok(list(sentences), padding="max_length", max_length=self.CLIP_LEN, truncation=True)
        out = []
        for text, ids, mask in zip(sentences, enc["input_ids"], enc["attention_mask"], strict=True):
            n_real = int(sum(mask))
            out.append(
                {
                    "text": text,
                    "tokens": tok.convert_ids_to_tokens(ids[:n_real]),
                    "input_ids_real": ids[:n_real],
                    "input_ids_padding": ids[n_real : n_real + 3],
                    "length": len(ids),
                    "n_real": n_real,
                    "n_padding": len(ids) - n_real,
                    "attention_mask_head": mask[: n_real + 3],
                    "attention_mask_sum": n_real,
                }
            )
        return out

    def _bert_tokens(self, tok: Any, sentences: Sequence[str]) -> list[dict[str, Any]]:
        enc = tok(list(sentences), padding=True)
        out = []
        for text, ids, mask, seg in zip(
            sentences, enc["input_ids"], enc["attention_mask"], enc["token_type_ids"], strict=True
        ):
            n_real = int(sum(mask))
            out.append(
                {
                    "text": text,
                    "tokens": tok.convert_ids_to_tokens(ids),
                    "input_ids": ids,
                    "length": len(ids),
                    "n_real": n_real,
                    "n_padding": len(ids) - n_real,
                    "attention_mask": mask,
                    "token_type_ids": seg,
                    "attention_mask_sum": n_real,
                }
            )
        return out

    def _clip_embed(self, sentence: str, padded: bool, use_mask: bool) -> np.ndarray:
        """Vetor de texto do CLIP; `padded` completa até 77 posições; `use_mask` liga a máscara."""
        model, proc = self.embedder.model, self.embedder.processor
        kw: dict[str, Any] = (
            {"padding": "max_length", "max_length": self.CLIP_LEN} if padded else {}
        )
        enc = proc.tokenizer([sentence], return_tensors="pt", **kw)
        ids = enc["input_ids"].to(self.embedder.device)
        mask = enc["attention_mask"].to(self.embedder.device)
        if not use_mask:
            mask = torch.ones_like(mask)
        with torch.no_grad():
            out = model.get_text_features(input_ids=ids, attention_mask=mask)
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        return feats.float().cpu().numpy()[0]

    def _clip_mask_experiment(self, sentence: str) -> dict[str, Any]:
        """A frase sozinha, com padding e máscara, e com padding sem máscara: o vetor muda?"""
        alone = self._clip_embed(sentence, padded=False, use_mask=True)
        return {
            "sentence": sentence,
            "cos_alone_vs_padded_with_mask": _cos(
                alone, self._clip_embed(sentence, padded=True, use_mask=True)
            ),
            "cos_alone_vs_padded_without_mask": _cos(
                alone, self._clip_embed(sentence, padded=True, use_mask=False)
            ),
        }

    @staticmethod
    def _bert_cls(bert: Any, enc: Any, use_mask: bool) -> np.ndarray:
        ids, seg = enc["input_ids"], enc["token_type_ids"]
        mask = enc["attention_mask"] if use_mask else torch.ones_like(ids)
        with torch.no_grad():
            hid = bert(input_ids=ids, attention_mask=mask, token_type_ids=seg).last_hidden_state
        return hid[0, 0].numpy()

    def _bert_mask_experiment(
        self, tok: Any, bert: Any, sentence: str, pad_to: int
    ) -> dict[str, Any]:
        """O vetor [CLS] da frase sozinha, com padding e máscara, e com padding sem máscara."""
        alone = self._bert_cls(bert, tok([sentence], return_tensors="pt"), True)
        padded = tok([sentence], return_tensors="pt", padding="max_length", max_length=pad_to)
        return {
            "sentence": sentence,
            "padded_to": pad_to,
            "cos_alone_vs_padded_with_mask": _cos(alone, self._bert_cls(bert, padded, True)),
            "cos_alone_vs_padded_without_mask": _cos(alone, self._bert_cls(bert, padded, False)),
        }

    def run(self, sentences: Sequence[str]) -> dict[str, Any]:
        """Saída dos dois tokenizadores para as mesmas frases e os experimentos de máscara."""
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.bert_name)
        bert = AutoModel.from_pretrained(self.bert_name).eval()
        bert_tokens = self._bert_tokens(tok, sentences)
        pad_to = bert_tokens[0]["length"]
        shortest = min(sentences, key=len)
        clip_tok = self.embedder.processor.tokenizer
        return {
            "sentences": list(sentences),
            "clip": {
                "model": self.embedder.model_name,
                "max_length": self.CLIP_LEN,
                "pad_token": clip_tok.pad_token,
                "bos_token": clip_tok.bos_token,
                "eos_token": clip_tok.eos_token,
                "pad_token_id": clip_tok.pad_token_id,
                "eos_token_id": clip_tok.eos_token_id,
                "tokens": self._clip_tokens(sentences),
                "mask_experiment": self._clip_mask_experiment(shortest),
            },
            "bert": {
                "model": self.bert_name,
                "padding": "longest (o maior do lote)",
                "cls_token": tok.cls_token,
                "sep_token": tok.sep_token,
                "pad_token": tok.pad_token,
                "tokens": bert_tokens,
                "mask_experiment": self._bert_mask_experiment(tok, bert, shortest, pad_to),
            },
        }
