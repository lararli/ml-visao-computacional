"""Um teste por item da rubrica da A2 (CLIP): prova a última execução do `make run-a2`.

Precisa de `make run-a2` já executado (marcador `data`, fora do `make test`).
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from visaocomp.a2.config import load_a2_config
from visaocomp.a2.store import EmbeddingStore
from visaocomp.paths import data_dir, repo_root
from visaocomp.run import latest_manifest

pytestmark = pytest.mark.data

PENDING = "texto de análise pendente"


@pytest.fixture(scope="module")
def run_dir():
    manifest = latest_manifest(repo_root() / "outputs", "a2")
    assert manifest is not None, "nenhuma execução da A2 em outputs/: rode `make run-a2`"
    return repo_root() / "outputs" / manifest["run_id"]


@pytest.fixture(scope="module")
def a2(run_dir):
    return run_dir / "a2"


@pytest.fixture(scope="module")
def report() -> str:
    path = repo_root() / "reports" / "sections" / "a2.md"
    assert path.exists(), "falta reports/sections/a2.md: rode `make run-a2`"
    return path.read_text(encoding="utf-8")


def section(report: str, tag: str) -> str:
    """O texto de uma seção `## R17...` até a próxima `## `."""
    m = re.search(rf"^## {tag}\b.*?(?=^## |\Z)", report, flags=re.S | re.M)
    assert m, f"a2.md não tem a seção {tag}"
    return m.group(0)


def words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def test_R17_alinhamento_clip(a2, report):
    ali = json.loads((a2 / "alignment.json").read_text(encoding="utf-8"))
    # evidência medida: sem treino, a categoria certa vence o acaso e o texto certo fica mais perto
    assert ali["zero_shot_top1"] > 3 * ali["chance_top1"]
    assert ali["zero_shot_top5"] > 2 * ali["chance_top5"]
    assert ali["cos_image_vs_true_category_text"] > ali["cos_image_vs_other_category_text"]
    assert ali["n_categories"] == 20 and ali["n_ads"] >= 300
    # a explicação por escrito
    text = section(report, "R17")
    low = text.lower()
    assert PENDING not in low and words(text) >= 250
    for term in ("contrastiv", "diagonal", "vocabulário aberto", "sem rótulo", "modality gap"):
        assert term in low, f"a seção R17 devia explicar: {term}"


def test_R18_ranking_de_objetos_ads16(a2, run_dir):
    ranking = pd.read_csv(a2 / "ranking.csv")
    assert len(ranking) >= 20 and ranking.description.is_unique
    assert {"frequency", "mean_score", "mean_score_present", "n_present"} <= set(ranking.columns)
    assert ranking.frequency.is_monotonic_decreasing and ranking.frequency.max() > 0
    pd_five = {"a car", "a person", "food product", "outdoor scenery", "text and logo"}
    assert pd_five <= set(ranking.description), "as 5 descrições de exemplo do PD estão no ranking"
    # o limiar tem justificativa: conferência a olho de 50 pares
    thr = json.loads((a2 / "threshold.json").read_text(encoding="utf-8"))
    assert thr["n_labeled"] >= 50 and 0.5 < thr["accuracy"] <= 1.0
    # os 5 mais frequentes, com 4 exemplos cada, figura gerada e imagens que existem
    top5 = list(ranking.description[:5])
    examples = json.loads((a2 / "top5_examples.json").read_text(encoding="utf-8"))
    assert list(examples) == top5 and all(len(v) == 4 for v in examples.values())
    assert (a2 / "top5.png").stat().st_size > 10_000
    sample = set(pd.read_csv(a2 / "sample.csv").image_path)
    assert all(e["image_path"] in sample for v in examples.values() for e in v)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["ads"] + manifest["counts"]["personal_photos"] >= 500


def test_R19_busca_semantica_8_consultas(a2, report):
    queries = json.loads((a2 / "queries.json").read_text(encoding="utf-8"))
    assert len(queries) >= 8
    assert len({q["nivel"] for q in queries}) >= 5, "as consultas variam em especificidade"
    text = section(report, "R19")
    for q in queries:
        assert len(q["results"]) == 5, q["id"]
        scores = [r["score"] for r in q["results"]]
        assert scores == sorted(scores, reverse=True)
        assert (a2 / "queries" / f"{q['id']}.png").stat().st_size > 5_000
        m = re.search(rf"### {q['id']}:.*?(?=^### |\Z)", text, flags=re.S | re.M)
        assert m, f"a2.md não tem a consulta {q['id']}"
        analysis = m.group(0).split("**Análise.**")[-1]
        assert PENDING not in analysis and words(analysis) >= 30, q["id"]
        assert q["text"] in m.group(0)
    assert any(q["text"] in ("happiness", "luxury") for q in queries), "há consulta abstrata"
    # o ranking vem do DuckDB: a mesma consulta, pedida de novo ao banco, dá o mesmo primeiro lugar
    cfg = load_a2_config()
    from visaocomp.a2.embedder import ClipEmbedder

    emb = ClipEmbedder(cfg.model, batch_size=cfg.batch_size)
    store = EmbeddingStore(data_dir() / "catalog.duckdb", cfg.embedding_table)
    first = queries[0]
    top = store.search(emb.embed_texts([first["text"]])[0], 1, cfg.model)
    assert top.image_path[0] in {r["image_path"] for r in first["results"]}


def test_R20_clip_vs_bert_padding_e_mask(a2, report):
    tok = json.loads((a2 / "tokenizers.json").read_text(encoding="utf-8"))
    clip, bert = tok["clip"], tok["bert"]
    assert (
        [c["text"] for c in clip["tokens"]]
        == [b["text"] for b in bert["tokens"]]
        == tok["sentences"]
    )
    for c in clip["tokens"]:  # CLIP: 77 posições, máscara = tokens reais
        assert c["length"] == 77 and c["n_real"] + c["n_padding"] == 77
        assert c["attention_mask_sum"] == c["n_real"] and c["tokens"][0] == clip["bos_token"]
        assert c["tokens"][-1] == clip["eos_token"]
    for b in bert["tokens"]:  # BERT: [CLS] ... [SEP] [PAD]..., máscara = tokens reais
        assert b["tokens"][0] == "[CLS]" and "[SEP]" in b["tokens"]
        assert sum(b["attention_mask"]) == b["n_real"] and len(b["attention_mask"]) == b["length"]
    assert any(b["n_padding"] > 0 for b in bert["tokens"]) and any(
        "##" in t for b in bert["tokens"] for t in b["tokens"]
    )
    # o experimento: sem a máscara, o BERT muda; o CLIP (atenção causal, leitura no EOS) não
    ce, be = clip["mask_experiment"], bert["mask_experiment"]
    assert ce["cos_alone_vs_padded_without_mask"] > 0.999
    assert be["cos_alone_vs_padded_with_mask"] > 0.999
    assert be["cos_alone_vs_padded_without_mask"] < 0.99
    text = section(report, "R20")
    low = text.lower()
    assert PENDING not in low and words(text) >= 250
    for term in (
        "padding",
        "attention mask",
        "bidirecional",
        "causal",
        "[cls]",
        "wordpiece",
        "bpe",
    ):
        assert term in low, f"a seção R20 devia explicar: {term}"
    assert "<|startoftext|>" in text and "[PAD]" in text, "a saída real dos dois tokenizadores"
