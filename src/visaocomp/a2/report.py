"""A2Report: gera reports/sections/a2.md a partir de outputs/<run_id>/ e do texto de análise.

Todo número do relatório vem dos arquivos de resultado. O texto de análise (escrito depois de olhar
as figuras) fica em `reports/a2_narrativa.yaml` e cita os números por `{{marcador}}`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from visaocomp.a2.config import A2Config

PENDING = "(texto de análise pendente)"
_MARK = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def fill(text: str, values: dict[str, str]) -> str:
    """Troca `{{nome}}` pelo valor; marcador desconhecido é erro (evita número digitado errado)."""

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"marcador desconhecido no texto de análise: {{{{{key}}}}}")
        return values[key]

    return _MARK.sub(sub, text)


def _pct(x: float, digits: int = 0) -> str:
    return f"{100 * x:.{digits}f}%"


class A2Report:
    """Monta o texto das seções R17 a R20 com tabelas e números lidos dos resultados."""

    def __init__(self, run_path: Path, cfg: A2Config, narrative_path: Path | None = None) -> None:
        self.run_path = Path(run_path)
        self.out = self.run_path / "a2"
        self.cfg = cfg
        self.narrative: dict[str, Any] = {}
        if narrative_path and Path(narrative_path).exists():
            self.narrative = yaml.safe_load(Path(narrative_path).read_text(encoding="utf-8")) or {}

    # ---- leitura ---------------------------------------------------------------------------

    def _json(self, name: str, folder: Path | None = None) -> Any:
        return json.loads(((folder or self.out) / name).read_text(encoding="utf-8"))

    def values(self) -> dict[str, str]:
        """Todos os marcadores que o texto de análise pode usar."""
        metrics = self._json("metrics.json", self.run_path)
        manifest = self._json("manifest.json", self.run_path)
        thr = self._json("threshold.json")
        ali = self._json("alignment.json")
        ranking = pd.read_csv(self.out / "ranking.csv")
        queries = self._json("queries.json")
        v: dict[str, str] = {
            "run_id": self.run_path.name,
            "n_sample": str(metrics["n_sample"]),
            "n_ads": str(manifest["counts"]["ads"]),
            "n_personal": str(manifest["counts"]["personal_photos"]),
            "n_descriptions": str(metrics["n_descriptions"]),
            "n_queries": str(metrics["n_queries"]),
            "threshold": f"{thr['threshold']:.3f}",
            "thr_accuracy": _pct(thr["accuracy"]),
            "thr_precision": _pct(thr["precision"]),
            "thr_recall": _pct(thr["recall"]),
            "thr_n": str(thr["n_labeled"]),
            "thr_n_present": str(thr["n_present"]),
            "dist_mean": f"{thr['distribution']['mean']:.3f}",
            "dist_p50": f"{thr['distribution']['p50']:.3f}",
            "dist_p90": f"{thr['distribution']['p90']:.3f}",
            "dist_p95": f"{thr['distribution']['p95']:.3f}",
            "dist_p99": f"{thr['distribution']['p99']:.3f}",
            "dist_max": f"{thr['distribution']['max']:.3f}",
            "zs_top1": _pct(ali["zero_shot_top1"], 1),
            "zs_top5": _pct(ali["zero_shot_top5"], 1),
            "zs_chance1": _pct(ali["chance_top1"], 0),
            "zs_chance5": _pct(ali["chance_top5"], 0),
            "zs_n": str(ali["n_ads"]),
            "cos_true": f"{ali['cos_image_vs_true_category_text']:.3f}",
            "cos_other": f"{ali['cos_image_vs_other_category_text']:.3f}",
            "cos_ii": f"{ali['cos_image_image']:.3f}",
            "cos_tt": f"{ali['cos_text_text']:.3f}",
            "cos_it": f"{ali['cos_image_text']:.3f}",
            "gap": f"{ali['modality_gap']:.3f}",
            "embeddings_new": str(metrics["embeddings_new"]),
            "embeddings_reused": str(metrics["embeddings_reused"]),
            "unreadable": str(metrics["unreadable"]),
        }
        per = ali["per_category_top1"]
        best = sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))
        v["zs_best"] = ", ".join(f"{k} ({_pct(x)})" for k, x in best[:3])
        v["zs_worst"] = ", ".join(f"{k} ({_pct(x)})" for k, x in best[-3:][::-1])
        for i, row in ranking.head(5).iterrows():
            n = i + 1
            v[f"top{n}_name"] = str(row.description)
            v[f"top{n}_freq"] = _pct(row.frequency, 1)
            v[f"top{n}_score"] = f"{row.mean_score_present:.3f}"
            v[f"top{n}_mean_all"] = f"{row.mean_score:.3f}"
            v[f"top{n}_freq_ad"] = _pct(row.frequency_ad, 1)
            v[f"top{n}_freq_foto"] = _pct(row.frequency_pessoal, 1)
        for i, row in ranking.iterrows():
            v[f"rank{i + 1}_name"] = str(row.description)
            v[f"rank{i + 1}_freq"] = _pct(row.frequency, 1)
        last = ranking.iloc[-1]
        v["last_name"], v["last_freq"] = str(last.description), _pct(last.frequency, 1)
        v["last_mean_all"] = f"{last.mean_score:.3f}"
        v["n_zero"] = str(int((ranking.n_present == 0).sum()))
        for q in queries:
            res = q["results"]
            v[f"{q['id']}_score1"] = f"{res[0]['score']:.3f}"
            v[f"{q['id']}_score5"] = f"{res[-1]['score']:.3f}"
            v[f"{q['id']}_n_ads"] = str(sum(r["source"] == "ad" for r in res))
            v[f"{q['id']}_cats"] = ", ".join(r["category"] or "foto pessoal" for r in res)
        tok = self._tokens_values()
        v.update(tok)
        return v

    def _tokens_values(self) -> dict[str, str]:
        path = self.out / "tokenizers.json"
        if not path.exists():
            return {}
        t = json.loads(path.read_text(encoding="utf-8"))
        v: dict[str, str] = {}
        for i, (c, b) in enumerate(zip(t["clip"]["tokens"], t["bert"]["tokens"], strict=True), 1):
            v[f"s{i}_clip_real"] = str(c["n_real"])
            v[f"s{i}_clip_pad"] = str(c["n_padding"])
            v[f"s{i}_bert_real"] = str(b["n_real"])
            v[f"s{i}_bert_pad"] = str(b["n_padding"])
        v["clip_len"] = str(t["clip"]["max_length"])
        v["bert_len"] = str(t["bert"]["tokens"][0]["length"])
        ce, be = t["clip"]["mask_experiment"], t["bert"]["mask_experiment"]
        v["clip_cos_mask"] = f"{ce['cos_alone_vs_padded_with_mask']:.4f}"
        v["clip_cos_nomask"] = f"{ce['cos_alone_vs_padded_without_mask']:.4f}"
        v["bert_cos_mask"] = f"{be['cos_alone_vs_padded_with_mask']:.4f}"
        v["bert_cos_nomask"] = f"{be['cos_alone_vs_padded_without_mask']:.4f}"
        v["exp_sentence"] = ce["sentence"]
        return v

    # ---- texto ------------------------------------------------------------------------------

    def _text(self, key: str, values: dict[str, str], sub: str | None = None) -> str:
        node = self.narrative.get(key)
        if sub is not None:
            node = (node or {}).get(sub) if isinstance(node, dict) else None
        return fill(str(node).strip(), values) if node else PENDING

    def _rel(self, name: str) -> str:
        return f"../../outputs/{self.run_path.name}/a2/{name}"

    def _ranking_table(self) -> str:
        df = pd.read_csv(self.out / "ranking.csv")
        lines = [
            "| # | Descrição | Imagens | Frequência | Score médio (todas) "
            "| Score médio (presentes) | Top-1 | Freq. anúncios | Freq. fotos |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in df.itertuples():
            present = "-" if pd.isna(r.mean_score_present) else f"{r.mean_score_present:.3f}"
            lines.append(
                f"| {r.rank} | {r.description} | {r.n_present} | {_pct(r.frequency, 1)} | "
                f"{r.mean_score:.3f} | {present} | {r.n_top1} | {_pct(r.frequency_ad, 1)} | "
                f"{_pct(r.frequency_pessoal, 1)} |"
            )
        return "\n".join(lines)

    def _query_blocks(self, values: dict[str, str]) -> str:
        blocks = []
        for q in self._json("queries.json"):
            rows = [
                "| # | Imagem | Origem / categoria | Score |",
                "|---|---|---|---|",
            ]
            for r in q["results"]:
                origin = r["category"] if r["source"] == "ad" else "foto pessoal do corpus"
                rows.append(f"| {r['rank']} | `{r['image_path']}` | {origin} | {r['score']:.3f} |")
            blocks.append(
                f'### {q["id"]}: "{q["text"]}" ({q["nivel"]})\n\n'
                f"![{q['id']}]({self._rel('queries/' + q['id'] + '.png')})\n\n"
                + "\n".join(rows)
                + "\n\n**Análise.** "
                + self._text("queries", values, q["id"])
            )
        return "\n\n".join(blocks)

    def _token_block(self) -> str:
        path = self.out / "tokenizers.json"
        if not path.exists():
            return "_(a comparação dos tokenizadores não foi gerada nesta execução)_"
        t = json.loads(path.read_text(encoding="utf-8"))
        lines = [
            "**CLIP** (`" + t["clip"]["model"] + "`): padding até "
            f"{t['clip']['max_length']} posições, tokens `{t['clip']['bos_token']}` e "
            f"`{t['clip']['eos_token']}`, padding também com `{t['clip']['pad_token']}`.\n"
        ]
        for c in t["clip"]["tokens"]:
            lines.append(
                f"- `{c['text']}`: tokens {c['tokens']}; ids reais {c['input_ids_real']}; "
                f"{c['n_real']} reais + {c['n_padding']} de padding "
                f"(ids do padding: {c['input_ids_padding']}...); "
                f"attention_mask = {c['attention_mask_head']}... (soma {c['attention_mask_sum']})"
            )
        lines.append(
            f"\n**BERT** (`{t['bert']['model']}`): padding até o maior do lote, "
            f"tokens `{t['bert']['cls_token']}` no começo e `{t['bert']['sep_token']}` no fim, "
            f"padding com `{t['bert']['pad_token']}`.\n"
        )
        for b in t["bert"]["tokens"]:
            lines.append(
                f"- `{b['text']}`: tokens {b['tokens']}; ids {b['input_ids']}; "
                f"attention_mask {b['attention_mask']}; token_type_ids {b['token_type_ids']}"
            )
        ce, be = t["clip"]["mask_experiment"], t["bert"]["mask_experiment"]
        lines.append(
            "\n**Experimento: o que a máscara muda?** Cosseno entre o vetor da frase "
            f"`{ce['sentence']}` sozinha (sem padding) e o vetor da mesma frase com padding.\n\n"
            "| Modelo | Padding com máscara | Padding sem máscara |\n|---|---|---|\n"
            f"| CLIP (vetor de texto, {t['clip']['max_length']} posições) | "
            f"{ce['cos_alone_vs_padded_with_mask']:.4f} | "
            f"{ce['cos_alone_vs_padded_without_mask']:.4f} |\n"
            f"| BERT (vetor `[CLS]`, {be['padded_to']} posições) | "
            f"{be['cos_alone_vs_padded_with_mask']:.4f} | "
            f"{be['cos_alone_vs_padded_without_mask']:.4f} |"
        )
        return "\n".join(lines)

    def build(self) -> str:
        """O texto inteiro da seção A2 do relatório."""
        v = self.values()
        t = lambda key: self._text(key, v)  # noqa: E731
        cat = self._json("category_map.json")
        cat_rows = "\n".join(
            f"| {f} | {name} | {cat['ads_per_folder'][f]} |"
            for f, name in cat["folder_to_category"].items()
        )
        thr = self._json("threshold.json")
        return f"""# Atividade 2: reconhecimento semântico em publicidade com CLIP

Execução `{v["run_id"]}`. Todos os números abaixo foram lidos de `outputs/{v["run_id"]}/a2/`
(`make run-a2`; `python scripts/run_a2.py --report-only` regenera este arquivo sem recalcular).
Nenhum modelo foi treinado: só embeddings do CLIP pré-treinado e consultas em linguagem natural.

## Dados e amostra

{t("dados")}

Mapa pasta para categoria do ADS-16 (fonte: `{cat["source"]}`):

| Pasta `Ads/<n>/` | Categoria | Anúncios |
|---|---|---|
{cat_rows}

Amostra: {v["n_sample"]} imagens, seed {self.cfg.seed}, lista em `a2/sample.csv` e no
`manifest.json`: {v["n_ads"]} anúncios (todos, estratificados pela pasta) e {v["n_personal"]} fotos
pessoais do corpus (estratificadas por polaridade positiva/negativa). Embeddings no DuckDB
(`{self.cfg.embedding_table}`): {v["embeddings_new"]} novos nesta execução,
{v["embeddings_reused"]} reaproveitados, {v["unreadable"]} ilegíveis.

## R17. Alinhamento entre imagem e texto no CLIP

{t("r17_texto")}

Evidência medida (`a2/alignment.json`):

| Medida | Valor |
|---|---|
| Pasta certa pelo nome (top-1, {v["zs_n"]} anúncios) | {v["zs_top1"]} (acaso {v["zs_chance1"]}) |
| Idem, categoria certa entre as 5 primeiras | {v["zs_top5"]} (acaso: {v["zs_chance5"]}) |
| Cosseno imagem e texto da categoria certa | {v["cos_true"]} |
| Cosseno imagem e texto de outra categoria | {v["cos_other"]} |
| Cosseno médio imagem e imagem | {v["cos_ii"]} |
| Cosseno médio texto e texto | {v["cos_tt"]} |
| Cosseno médio imagem e texto | {v["cos_it"]} |
| Distância entre o centro das imagens e o centro dos textos (modality gap) | {v["gap"]} |

Categorias que mais acertam: {v["zs_best"]}. Que menos acertam: {v["zs_worst"]}.

{t("r17_leitura")}

## R18. Ranking de objetos por frequência semântica

{v["n_descriptions"]} descrições, cada uma com dois templates (`{"`, `".join(self.cfg.templates)}`)
e a média dos vetores (prompt ensembling).

**Limiar de presença: cosseno >= {v["threshold"]}.** {t("threshold_texto")}

Pares conferidos a olho, com o score de cada um, em `a2/threshold.json`
(critério: {thr["criterion"]}; avaliador: {thr["rater"]}).

{self._ranking_table()}

![Os 5 objetos mais frequentes, com 4 exemplos cada]({self._rel("top5.png")})

{t("ranking_leitura")}

## R19. Busca semântica por texto

{v["n_queries"]} consultas, top-{self.cfg.top_k} cada, cosseno calculado dentro do DuckDB
(`array_cosine_similarity`). Resultados completos em `a2/queries.json`.

{self._query_blocks(v)}

### Leitura do conjunto

{t("queries_resumo")}

## R20. CLIP e BERT: como o texto vira entrada

{self._token_block()}

{t("r20_texto")}

## Limites

{t("limites")}
"""

    def write(self, path: Path) -> Path:
        """Grava o relatório (esta seção é regenerada, não editada à mão)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.build(), encoding="utf-8")
        return path
