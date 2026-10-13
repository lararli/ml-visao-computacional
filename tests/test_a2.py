"""Testes da A2 (CLIP no ADS-16): sem rede, sem GPU, com um CLIP falso e um ADS-16 de brinquedo."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from visaocomp.a2.alignment import AlignmentProbe
from visaocomp.a2.config import A2Config, Query, load_a2_config
from visaocomp.a2.embedder import ClipEmbedder, l2_normalize
from visaocomp.a2.pipeline import (
    A2Pipeline,
    ThresholdCheckMissingError,
    load_threshold_check,
)
from visaocomp.a2.ranker import ObjectRanker, ThresholdCalibrator
from visaocomp.a2.report import A2Report, fill
from visaocomp.a2.sampler import Ads16Sampler, ContractError, allocate, stratified_sample
from visaocomp.a2.search import TextSearch
from visaocomp.a2.store import EmbeddingStore

DIM = 16
CATS = [f"Categoria {i}" for i in range(20)]


# ---- ADS-16 de brinquedo e CLIP falso -----------------------------------------------------------


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color).save(path)


def make_ads16(root: Path, per_folder: int = 3, users: int = 3, bad: bool = False) -> Path:
    """Cria `raw/ads16` no formato do ADS-16 (2 partes; pastas 1..20; usuários com fotos)."""
    raw = root / "raw" / "ads16"
    rng = np.random.default_rng(0)
    for part, folders in ((1, range(1, 11)), (2, range(11, 21))):
        base = raw / f"ADS16_Benchmark_part{part}" / f"ADS16_Benchmark_part{part}"
        for f in folders:
            for i in range(1, per_folder + 1):
                _png(
                    base / "Ads" / "Ads" / str(f) / f"{i}.png",
                    tuple(int(x) for x in rng.integers(0, 255, 3)),
                )
        for u in range(1, users + 1):
            if (u % 2) != (part % 2):
                continue
            user = base / "Corpus" / "Corpus" / f"U{u:04d}"
            for pol in ("POS", "NEG"):
                for i in range(1, 4):
                    _png(
                        user / f"U{u:04d}-IM-{pol}" / f"{i}.png",
                        tuple(int(x) for x in rng.integers(0, 255, 3)),
                    )
            with (user / f"U{u:04d}-RT.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, delimiter=";", quoting=csv.QUOTE_ALL)
                w.writerow([f"Cat{i}" for i in range(20)])
                w.writerow(CATS)
    if bad:
        bad_path = (
            raw / "ADS16_Benchmark_part1" / "ADS16_Benchmark_part1" / "Ads" / "Ads" / "1" / "9.png"
        )
        bad_path.write_bytes(b"isto nao e um png")
    return raw


class FakeProcessor:
    """Imagens viram a cor média; textos viram ids derivados do hash de cada palavra."""

    def __call__(self, images=None, text=None, return_tensors="pt", padding=True, truncation=True):
        if images is not None:
            arr = [
                torch.tensor(np.asarray(im, dtype=np.float32).mean((0, 1)) / 255) for im in images
            ]
            return {"pixel_values": torch.stack(arr)}
        ids = [
            [int(hashlib.md5(w.encode()).hexdigest(), 16) % 1000 + 1 for w in t.split()]
            for t in text
        ]
        width = max(len(i) for i in ids)
        mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids])
        padded = torch.tensor([i + [0] * (width - len(i)) for i in ids])
        return {"input_ids": padded, "attention_mask": mask}


class FakeModel:
    """Projeções fixas; o mesmo texto dá sempre o mesmo vetor."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(1)
        self.img = torch.randn(3, DIM, generator=g)
        self.txt = torch.randn(1000, DIM, generator=g)
        self.calls = 0

    def get_image_features(self, pixel_values):
        self.calls += 1
        return pixel_values @ self.img

    def get_text_features(self, input_ids, attention_mask):
        emb = self.txt[input_ids] * attention_mask.unsqueeze(-1)
        return emb.sum(1)


def make_embedder(batch_size: int = 4) -> ClipEmbedder:
    return ClipEmbedder("fake/clip", torch.device("cpu"), batch_size, FakeModel(), FakeProcessor())


def make_cfg(tmp_path: Path, sample_size: int = 20, n_check: int = 4) -> A2Config:
    descriptions = tuple(f"objeto {i}" for i in range(20))
    return A2Config(
        model="fake/clip",
        batch_size=4,
        seed=42,
        sample_size=sample_size,
        embedding_table="silver.ads16_embedding",
        top_k=5,
        examples_per_object=2,
        n_threshold_check=n_check,
        threshold_check_file="check.json",
        templates=("foto de {}",),
        descriptions=descriptions,
        queries=tuple(Query(f"q{i:02d}", f"consulta {i}", "nivel") for i in range(8)),
        tokenizer_demo={},
        path=tmp_path / "a2.yaml",
    )


# ---- configuração -------------------------------------------------------------------------------


def test_config_real_tem_20_descricoes_e_8_consultas():
    cfg = load_a2_config()
    assert len(cfg.descriptions) >= 20 and len(cfg.queries) >= 8
    for pd_desc in ("a car", "a person", "food product", "outdoor scenery", "text and logo"):
        assert pd_desc in cfg.descriptions, "as 5 descrições do PD têm de estar na lista"
    assert cfg.sample_size >= 500 and cfg.seed == 42


def test_config_recusa_menos_de_20_descricoes(tmp_path):
    doc = yaml.safe_load(load_a2_config().path.read_text(encoding="utf-8"))
    doc["descriptions"] = doc["descriptions"][:5]
    bad = tmp_path / "a2.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="20"):
        load_a2_config(bad)


# ---- amostra ------------------------------------------------------------------------------------


def test_allocate_soma_n_e_respeita_o_tamanho_do_estrato():
    out = allocate({"a": 10, "b": 5, "c": 1}, 8)
    assert sum(out.values()) == 8 and out["c"] <= 1 and out["a"] >= out["b"]
    assert allocate({"a": 3, "b": 2}, 99) == {"a": 3, "b": 2}


def test_amostra_estratificada_e_deterministica_com_a_mesma_seed():
    df = pd.DataFrame(
        {"image_path": [f"x/{i}" for i in range(100)], "stratum": ["a"] * 60 + ["b"] * 40}
    )
    one = stratified_sample(df, "stratum", 20, 42)
    two = stratified_sample(df, "stratum", 20, 42)
    other = stratified_sample(df, "stratum", 20, 7)
    assert list(one.image_path) == list(two.image_path)
    assert list(one.image_path) != list(other.image_path)
    assert one.stratum.value_counts().to_dict() == {"a": 12, "b": 8}


def test_sampler_mapa_categoria_e_contrato(tmp_path):
    s = Ads16Sampler(make_ads16(tmp_path))
    assert s.category_map()["1"] == "Categoria 0" and s.category_map()["20"] == "Categoria 19"
    report = s.validate()
    assert report["n_ads"] == 60 and set(report["ads_per_folder"]) == {str(i) for i in range(1, 21)}
    assert report["folders_with_other_than_15"], "o brinquedo tem 3 por pasta, não 15"


def test_sampler_completa_com_fotos_quando_faltam_anuncios(tmp_path):
    s = Ads16Sampler(make_ads16(tmp_path))
    sample = s.sample(70, 42)
    assert len(sample) == 70 and (sample.source == "ad").sum() == 60
    assert set(sample[sample.source == "pessoal"].stratum) <= {"pessoal/POS", "pessoal/NEG"}
    assert list(sample.image_path) == list(s.sample(70, 42).image_path)
    assert (
        sample.image_path.is_unique and sample.image_path.str.startswith(("Ads/", "Corpus/")).all()
    )


def test_sampler_estrutura_errada_diz_o_que_esperava(tmp_path):
    raw = make_ads16(tmp_path)
    import shutil

    shutil.rmtree(raw / "ADS16_Benchmark_part2" / "ADS16_Benchmark_part2" / "Ads" / "Ads" / "20")
    with pytest.raises(ContractError, match=r"1\.\.20.*faltam \['20'\]"):
        Ads16Sampler(raw).validate()
    with pytest.raises(ContractError, match="ADS16_Benchmark_part"):
        Ads16Sampler(tmp_path / "vazio").validate()


# ---- embedder -----------------------------------------------------------------------------------


def test_l2_normalize_da_norma_1_e_aceita_linha_zero():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8]) and np.allclose(out[1], 0)


def test_imagem_ilegivel_e_pulada_e_contada(tmp_path):
    good = tmp_path / "a.png"
    _png(good, (10, 20, 30))
    bad = tmp_path / "b.png"
    bad.write_bytes(b"lixo")
    vecs, ok, errors = make_embedder().embed_images([good, bad, good])
    assert ok == [0, 2] and vecs.shape == (2, DIM) and len(errors) == 1 and "b.png" in errors[0]
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_texto_o_mesmo_vetor_com_ou_sem_padding_no_lote():
    emb = make_embedder()
    alone = emb.embed_texts(["a car"])[0]
    batch = emb.embed_texts(["a car", "a red sports car on a road"])[0]
    assert np.allclose(alone, batch, atol=1e-6)


def test_ensembling_normaliza_a_media():
    emb = make_embedder()
    out = emb.embed_prompt_groups([["a b", "c d"], ["e f"]])
    assert out.shape == (2, DIM) and np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


# ---- DuckDB -------------------------------------------------------------------------------------


def _store(tmp_path: Path) -> EmbeddingStore:
    return EmbeddingStore(tmp_path / "catalog.duckdb", dim=DIM, lock_attempts=2, lock_wait=0.0)


def _meta(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": [f"Ads/1/{i}.png" for i in range(n)],
            "source": "ad",
            "folder": "1",
            "category": "Categoria 0",
        }
    )


def test_store_escrita_idempotente_e_busca_por_cosseno(tmp_path):
    store = _store(tmp_path)
    vecs = l2_normalize(np.random.default_rng(0).normal(size=(5, DIM)))
    assert store.add(_meta(5), vecs, "m") == 5
    assert store.add(_meta(5), vecs, "m") == 0, "a mesma chave não entra duas vezes"
    assert store.add(_meta(5), vecs, "outro-modelo") == 5, "a chave inclui o modelo"
    assert store.existing([f"Ads/1/{i}.png" for i in range(7)], "m") == {
        f"Ads/1/{i}.png" for i in range(5)
    }
    top = store.search(vecs[3], 2, "m")
    assert top.image_path[0] == "Ads/1/3.png" and top.score[0] == pytest.approx(1.0, abs=1e-5)
    only = store.search(vecs[3], 5, "m", paths=["Ads/1/0.png", "Ads/1/1.png"])
    assert set(only.image_path) == {"Ads/1/0.png", "Ads/1/1.png"}
    meta, matrix = store.load(["Ads/1/4.png", "Ads/1/0.png"], "m")
    assert list(meta.image_path) == ["Ads/1/4.png", "Ads/1/0.png"]
    assert np.allclose(matrix[0], vecs[4], atol=1e-6)


def test_store_dimensao_errada_e_erro_claro(tmp_path):
    with pytest.raises(ValueError, match="esperava vetores de 16"):
        _store(tmp_path).add(_meta(2), np.ones((2, 8)), "m")


def test_store_espera_quando_o_arquivo_esta_travado(tmp_path, monkeypatch):
    import duckdb

    real = duckdb.connect
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise duckdb.IOException("Could not set lock on file")
        return real(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", flaky)
    store = _store(tmp_path)
    store.ensure_table()
    assert calls["n"] == 2


# ---- ranking, limiar, alinhamento -------------------------------------------------------------


def test_ranker_conta_acima_do_limiar_e_ordena():
    sims = np.array([[0.30, 0.10], [0.28, 0.26], [0.10, 0.27], [0.12, 0.05]], dtype=np.float32)
    ranker = ObjectRanker(["a", "b"], 0.25)
    out = ranker.rank(sims, ["ad", "ad", "pessoal", "pessoal"])
    assert list(out.description) == ["a", "b"] or list(out.description) == ["b", "a"]
    a = out.set_index("description").loc["a"]
    assert (
        a.n_present == 2
        and a.frequency == 0.5
        and a.frequency_ad == 1.0
        and a.frequency_pessoal == 0.0
    )
    assert a.mean_score_present == pytest.approx(0.29)
    assert out.frequency.is_monotonic_decreasing
    assert [i for i, _ in ranker.examples(sims, "a", 2)] == [0, 1]
    with pytest.raises(ValueError):
        ranker.rank(np.zeros((2, 3)), ["ad", "ad"])


def test_similaridade_de_vetores_normalizados_e_o_cosseno():
    a, b = l2_normalize(np.array([[1.0, 0.0]])), l2_normalize(np.array([[1.0, 1.0]]))
    assert ObjectRanker.similarities(a, b)[0, 0] == pytest.approx(np.cos(np.pi / 4), abs=1e-6)


def test_calibrador_escolhe_o_limiar_que_mais_concorda():
    scores = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
    present = [False, False, False, True, True, True]
    out = ThresholdCalibrator.choose(scores, present)
    assert 0.24 < out["threshold"] <= 0.26 and out["accuracy"] == 1.0
    with pytest.raises(ValueError):
        ThresholdCalibrator.choose([], [])


def test_calibrador_pares_sao_deterministicos_e_um_por_imagem():
    sims = np.random.default_rng(0).normal(0.2, 0.03, size=(100, 8)).astype(np.float32)
    paths = [f"i{i}" for i in range(100)]
    desc = [f"d{j}" for j in range(8)]
    cal = ThresholdCalibrator(42, 20)
    one = cal.candidate_pairs(sims, paths, desc)
    assert one == cal.candidate_pairs(sims, paths, desc)
    assert len(one) == 20 and len({p["image_path"] for p in one}) == 20


def test_zero_shot_topk():
    sims = np.array([[0.9, 0.1, 0.0], [0.1, 0.2, 0.9]])
    truth = np.array([0, 1])
    assert AlignmentProbe.topk_accuracy(sims, truth, 1) == 0.5
    assert AlignmentProbe.topk_accuracy(sims, truth, 2) == 1.0


def test_pares_rotulados_faltando_sao_erro_claro(tmp_path):
    with pytest.raises(ThresholdCheckMissingError, match="--calibrate"):
        load_threshold_check(tmp_path / "nao-existe.json", 4)
    path = tmp_path / "c.json"
    path.write_text(
        json.dumps({"pairs": [{"image_path": "x", "description": "y", "present": None}]})
    )
    with pytest.raises(ThresholdCheckMissingError, match="sem rótulo"):
        load_threshold_check(path, 1)


def test_marcador_desconhecido_no_texto_e_erro():
    assert fill("n={{x}}", {"x": "3"}) == "n=3"
    with pytest.raises(KeyError, match="y"):
        fill("n={{y}}", {"x": "3"})


# ---- pipeline inteiro (smoke): CLIP falso, ADS-16 de brinquedo, DuckDB de verdade ----------------


def _write_check(path: Path, run_sims_pairs: list[dict]) -> None:
    pairs = [{**p, "present": i % 2 == 0, "note": ""} for i, p in enumerate(run_sims_pairs)]
    path.write_text(json.dumps({"rater": "teste", "criterion": "teste", "pairs": pairs}))


def _pipeline(tmp_path: Path, bad: bool = False) -> A2Pipeline:
    raw = make_ads16(tmp_path / "data.nosync", bad=bad)
    cfg = make_cfg(tmp_path, sample_size=70)
    pipe = A2Pipeline(
        cfg,
        tmp_path / "data.nosync",
        tmp_path / "outputs",
        make_embedder(),
        store=EmbeddingStore(tmp_path / "data.nosync" / "catalog.duckdb", dim=DIM),
        threshold_check=tmp_path / "check.json",
        ads16_root=raw,
    )
    return pipe


@pytest.mark.smoke
def test_pipeline_completo_e_segunda_execucao_sem_embeddings_novos(tmp_path, caplog):
    pipe = _pipeline(tmp_path, bad=True)
    cand = pipe.calibrate(tmp_path / "calib")
    doc = json.loads(cand.read_text())
    assert len(doc["pairs"]) == 4 and (tmp_path / "calib" / "threshold_sheet_1.png").exists()
    _write_check(tmp_path / "check.json", doc["pairs"])

    with caplog.at_level(logging.INFO):
        run = pipe.run()
    out = run.path / "a2"
    for name in (
        "ranking.csv", "top5.png", "queries.json", "threshold.json", "alignment.json",
        "category_map.json", "sample.csv",
    ):  # fmt: skip
        assert (out / name).exists(), name
    metrics = json.loads((run.path / "metrics.json").read_text())
    assert metrics["unreadable"] == 1 and metrics["n_embedded"] == 69
    assert len(pd.read_csv(out / "ranking.csv")) == 20
    queries = json.loads((out / "queries.json").read_text())
    assert len(queries) == 8 and all(len(q["results"]) == 5 for q in queries)
    manifest = json.loads((run.path / "manifest.json").read_text())
    assert len(manifest["sample_image_paths"]) == 70 and manifest["stage"] == "a2"

    caplog.clear()
    with caplog.at_level(logging.INFO):
        pipe.run()
    assert "embeddings: 0 novos, 69 reaproveitados, 1 ilegíveis" in caplog.text


@pytest.mark.smoke
def test_segunda_execucao_nao_chama_o_modelo_para_imagens(tmp_path):
    pipe = _pipeline(tmp_path)
    cand = pipe.calibrate(tmp_path / "calib")
    _write_check(tmp_path / "check.json", json.loads(cand.read_text())["pairs"])
    pipe.run()
    model = pipe.embedder.model
    before = model.calls
    pipe.run()
    assert model.calls == before, "a 2a execução não podia gerar embeddings de imagem"


@pytest.mark.smoke
def test_sem_arquivo_de_conferencia_o_run_para_cedo(tmp_path):
    pipe = _pipeline(tmp_path)
    with pytest.raises(ThresholdCheckMissingError):
        pipe.run()


def test_busca_devolve_k_resultados_ordenados(tmp_path):
    store = EmbeddingStore(tmp_path / "c.duckdb", dim=DIM)
    emb = make_embedder()
    paths = [tmp_path / f"{i}.png" for i in range(8)]
    for i, p in enumerate(paths):
        _png(p, (i * 30, 255 - i * 30, 40))
    vecs, ok, _ = emb.embed_images(paths)
    meta = pd.DataFrame(
        {
            "image_path": [f"Ads/1/{i}.png" for i in ok],
            "source": "ad",
            "folder": "1",
            "category": "c",
        }
    )
    store.add(meta, vecs, emb.model_name)
    res = TextSearch(emb, store, k=5).query("uma frase")
    assert len(res) == 5 and [r["rank"] for r in res] == [1, 2, 3, 4, 5]
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_relatorio_sem_texto_de_analise_marca_pendente(tmp_path):
    pipe = _pipeline(tmp_path)
    cand = pipe.calibrate(tmp_path / "calib")
    _write_check(tmp_path / "check.json", json.loads(cand.read_text())["pairs"])
    run = pipe.run()
    md = A2Report(run.path, pipe.cfg, None).build()
    assert "texto de análise pendente" in md and "## R17" in md and "## R20" in md


@pytest.mark.data
def test_rodar_de_novo_no_catalogo_real_nao_recalcula_embeddings(caplog, tmp_path):
    """Com o `make run-a2` já feito, as 600 imagens estão no DuckDB: 0 embeddings novos."""
    from visaocomp.paths import data_dir

    cfg = load_a2_config()

    class NoModel:
        model_name = cfg.model

        def embed_images(self, paths):
            raise AssertionError("tentou gerar embedding de imagem que já está no DuckDB")

        def embed_texts(self, texts):
            raise AssertionError("não devia chamar o modelo de texto aqui")

    pipe = A2Pipeline(cfg, data_dir(), tmp_path / "outputs", NoModel())
    sample = pipe.sampler.sample(cfg.sample_size, cfg.seed)
    with caplog.at_level(logging.INFO):
        counts = pipe.embed_sample(sample)
    assert counts["new"] == 0 and counts["reused"] + counts["unreadable"] == len(sample)
    assert counts["reused"] > 0, "rode `make run-a2` antes"
    assert f"embeddings: 0 novos, {counts['reused']} reaproveitados" in caplog.text
