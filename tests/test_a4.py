"""Unitários da A4: formas, passos de D e de G, replay buffer, dedup, caso, FID e métricas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from visaocomp.a4.config import CaseConfig, GanRunConfig, load_a4_config
from visaocomp.a4.data import (
    CaseBuilder,
    Deduplicator,
    ImageRecord,
    XrayDataset,
    hamming_matrix_rows,
    list_records,
)
from visaocomp.a4.fid import FidMeter
from visaocomp.a4.gan import ConditionalGan, GanTrainer, ReplayBuffer
from visaocomp.a4.recall import (
    GpuAugment,
    RecallExperiment,
    XrayClassifier,
    class_weights,
    classification_metrics,
    mean_ci,
)
from visaocomp.paths import repo_root

CPU = torch.device("cpu")


@pytest.fixture
def cfg():
    return load_a4_config(overrides={"gan": {"batch_size": 4, "ngf": 8, "ndf": 8}})


def params(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(
        not torch.equal(a, b.detach()) for a, b in zip(before, module.parameters(), strict=True)
    )


# ---------- GAN ----------


@pytest.mark.parametrize("sn", [False, True])
def test_formas_do_gerador_e_do_discriminador(sn):
    gan = ConditionalGan(nz=16, n_classes=3, ngf=8, ndf=8, use_spectral_norm=sn)
    z, y = torch.randn(5, 16), torch.tensor([0, 1, 2, 2, 0])
    img = gan.generator(z, y)
    assert img.shape == (5, 1, 64, 64) and img.min() >= -1 and img.max() <= 1
    assert gan.discriminator(img, y).shape == (5,)
    assert gan.sample(y).shape == (5, 1, 64, 64)


def test_passo_do_discriminador_so_muda_o_discriminador(cfg):
    torch.manual_seed(0)
    gan = ConditionalGan(cfg.gan.nz, 3, 8, 8)
    tr = GanTrainer(gan, cfg.gan, GanRunConfig("x"), CPU)
    g0, d0 = params(gan.generator), params(gan.discriminator)
    real = torch.rand(4, 1, 64, 64) * 2 - 1
    tr.discriminator_step(real, torch.tensor([0, 1, 2, 2]))
    assert changed(d0, gan.discriminator)
    assert not changed(g0, gan.generator)


def test_passo_do_gerador_so_muda_o_gerador(cfg):
    torch.manual_seed(0)
    gan = ConditionalGan(cfg.gan.nz, 3, 8, 8)
    tr = GanTrainer(gan, cfg.gan, GanRunConfig("x"), CPU)
    _, fake, lab = tr.discriminator_step(torch.rand(4, 1, 64, 64), torch.tensor([0, 1, 2, 2]))
    g0, d0 = params(gan.generator), params(gan.discriminator)
    tr.generator_step(fake, lab)
    assert changed(g0, gan.generator)
    assert not changed(d0, gan.discriminator)
    assert all(p.grad is None for p in gan.discriminator.parameters())


def test_loop_alterna_discriminador_e_gerador(cfg, tmp_path):
    gan = ConditionalGan(cfg.gan.nz, 3, 8, 8)
    tr = GanTrainer(gan, cfg.gan, GanRunConfig("x"), CPU, out_dir=tmp_path)
    x = torch.randint(0, 255, (12, 1, 64, 64), dtype=torch.uint8)
    y = torch.tensor([0, 1, 2] * 4)
    tr.fit(x, y, epochs=1)
    assert tr.step_order == ["D", "G"] * 3
    assert (tmp_path / "samples_epoch_001.png").exists()
    assert (tmp_path / "history.json").exists() and (tmp_path / "curves.png").exists()


def test_label_smoothing_muda_o_alvo_real(cfg):
    torch.manual_seed(0)
    gan = ConditionalGan(cfg.gan.nz, 3, 8, 8)
    real = torch.rand(4, 1, 64, 64)
    y = torch.tensor([0, 1, 2, 0])
    losses = []
    for smooth in (1.0, 0.9):
        torch.manual_seed(1)
        g = ConditionalGan(cfg.gan.nz, 3, 8, 8)
        g.load_state_dict(gan.state_dict())
        tr = GanTrainer(g, cfg.gan, GanRunConfig("x", label_smoothing=smooth), CPU)
        losses.append(tr.discriminator_step(real, y)[0]["loss_d"])
    assert losses[0] != losses[1]


def test_replay_buffer_enche_e_depois_mistura_antigas():
    buf = ReplayBuffer(capacity=4, prob=1.0, seed=0)
    first = torch.zeros(4, 1, 2, 2)
    out, _ = buf.push_and_pop(first, torch.zeros(4, dtype=torch.long))
    assert torch.equal(out, first) and len(buf) == 4  # enchendo: devolve as atuais
    new = torch.ones(4, 1, 2, 2)
    out, lab = buf.push_and_pop(new, torch.ones(4, dtype=torch.long))
    assert out.shape == new.shape and (out == 0).all() and (lab == 0).all()  # p=1: só antigas


def test_replay_buffer_desligado_devolve_o_lote():
    buf = ReplayBuffer(capacity=0, prob=0.5)
    x = torch.rand(3, 1, 2, 2)
    assert buf.push_and_pop(x, torch.zeros(3, dtype=torch.long))[0] is x


def test_estrategias_de_mitigacao_legiveis():
    rc = GanRunConfig("m", label_smoothing=0.9, spectral_norm=True, replay_size=8, replay_prob=0.5)
    assert len(rc.strategies) == 3 and GanRunConfig("s").strategies == []


# ---------- FID ----------


def test_frechet_de_conjuntos_iguais_e_zero_e_cresce_com_deslocamento():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 8))
    mu, s = FidMeter.stats(a)
    assert abs(FidMeter.frechet(mu, s, mu, s)["fid"]) < 1e-6
    shifted = FidMeter.frechet(mu, s, mu + 1.0, s)
    assert shifted["mean_term"] == pytest.approx(8.0) and abs(shifted["cov_term"]) < 1e-6
    collapsed = FidMeter.stats(np.repeat(a[:5], 100, axis=0))
    assert (
        FidMeter.frechet(mu, s, *collapsed)["cov_term"] > 1.0
    )  # colapso sobe o termo da covariância


# ---------- dedup e caso ----------


def save_img(path: Path, arr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def test_deduplicator_acha_copias_e_descarta_rotulos_conflitantes(tmp_path):
    rng = np.random.default_rng(0)
    base = [rng.integers(0, 255, (32, 32), dtype=np.uint8) for _ in range(4)]
    recs = [
        ImageRecord(save_img(tmp_path / "A/a0.png", base[0]), 0),
        ImageRecord(save_img(tmp_path / "A/a0_copia.png", base[0]), 0),
        ImageRecord(save_img(tmp_path / "A/a1.png", base[1]), 0),
        ImageRecord(save_img(tmp_path / "B/b2.png", base[2]), 1),
        ImageRecord(save_img(tmp_path / "A/a2_mesma_de_b2.png", base[2]), 0),
        ImageRecord(save_img(tmp_path / "B/b3.png", base[3]), 1),
    ]
    res = Deduplicator(cache_path=tmp_path / "cache.csv").run(recs)
    kept = {r.path.name for r in res.kept}
    assert kept == {"a0.png", "a1.png", "b3.png"}
    assert res.report["copias_removidas"] == 1
    assert res.report["removidas_por_conflito_de_rotulo"] == 2
    again = Deduplicator(cache_path=tmp_path / "cache.csv").run(recs)  # do cache
    assert {r.path.name for r in again.kept} == kept


def test_hamming():
    h = np.array([0b0, 0b1, 0b11, 0xFF], dtype=np.uint64)
    assert hamming_matrix_rows(h[:1], h).tolist() == [[0, 1, 2, 8]]


def test_imagem_corrompida_para_com_mensagem(tmp_path):
    bad = tmp_path / "C/bad.png"
    bad.parent.mkdir()
    bad.write_bytes(b"nao e imagem")
    with pytest.raises(ValueError, match="ilegível"):
        Deduplicator().hash_one(bad)
    with pytest.raises(ValueError, match="ilegível"):
        XrayDataset([ImageRecord(bad, 0)], 8).stack()


def test_classe_vazia_para_cedo(tmp_path):
    (tmp_path / "NORMAL").mkdir()
    with pytest.raises(ValueError, match="vazia"):
        list_records(tmp_path, ("NORMAL",))


def fake_records(n_per_class: int) -> tuple[list[ImageRecord], dict[str, int]]:
    recs = [ImageRecord(Path(f"/x/{c}/{i}.png"), c) for c in range(3) for i in range(n_per_class)]
    return recs, {str(r.path): i for i, r in enumerate(recs)}


def test_case_builder_conta_e_separa_sem_sobreposicao():
    case = CaseConfig(
        seed=42, test_per_class=5, train_counts={"N": 20, "P": 10, "C": 5}, val_fraction=0.2
    )
    recs, hashes = fake_records(40)
    split = CaseBuilder(case, ("N", "P", "C")).build(recs, hashes)
    counts = split.counts()
    assert counts["test"] == {"N": 5, "P": 5, "C": 5}
    assert counts["val"] == {"N": 4, "P": 2, "C": 1}
    assert counts["train"] == {"N": 16, "P": 8, "C": 4}
    f = split.frame
    assert set(f[f.split == "test"].path).isdisjoint(f[f.split != "test"].path)
    again = CaseBuilder(case, ("N", "P", "C")).build(recs, hashes).frame
    assert again.equals(f)  # mesma seed, mesmo split


def test_case_builder_para_se_faltar_imagem():
    case = CaseConfig(
        seed=0, test_per_class=5, train_counts={"N": 20, "P": 1, "C": 1}, val_fraction=0.2
    )
    recs, hashes = fake_records(10)
    with pytest.raises(ValueError, match="esperava ao menos 25"):
        CaseBuilder(case, ("N", "P", "C")).build(recs, hashes)


# ---------- classificador e estatística ----------


def test_augment_preserva_forma_e_e_determinista():
    x = torch.rand(6, 1, 32, 32)
    aug = GpuAugment()
    a = aug(x, torch.Generator().manual_seed(0))
    b = aug(x, torch.Generator().manual_seed(0))
    assert a.shape == x.shape and torch.equal(a, b) and a.min() >= 0 and a.max() <= 1


def test_classificador_aceita_cinza():
    model = XrayClassifier(3, pretrained=False)
    assert model(torch.rand(2, 1, 32, 32)).shape == (2, 3)


def test_class_weights_pesam_a_classe_rara():
    w = class_weights(torch.tensor([0] * 70 + [1] * 20 + [2] * 10), 3)
    assert w[2] > w[1] > w[0]
    with pytest.raises(ValueError):
        class_weights(torch.tensor([0, 0, 1]), 3)


def test_metricas_e_precisao_na_prevalencia():
    cm = np.array([[90, 5, 5], [10, 80, 10], [10, 10, 80]])
    m = classification_metrics(cm, ("NORMAL", "PNEUMONIA", "COVID"), 2, np.array([0.7, 0.2, 0.1]))
    assert m["recall_covid"] == pytest.approx(0.8)
    assert m["precision_covid"] == pytest.approx(80 / 95)
    # na prevalência do caso (10% COVID) a precisão cai: 0.08 / (0.035 + 0.02 + 0.08)
    assert m["ppv_covid_prevalencia_caso"] == pytest.approx(0.08 / 0.135)


def test_resumo_pareado_so_declara_melhora_com_intervalo_acima_de_zero():
    df = pd.DataFrame(
        {
            "condition": ["a"] * 3 + ["b"] * 3,
            "seed": [0, 1, 2] * 2,
            "recall_covid": [0.5, 0.6, 0.55, 0.52, 0.58, 0.6],
            "balanced_accuracy": [0.7] * 6,
            "accuracy": [0.7] * 6,
            "ppv_covid_prevalencia_caso": [0.5] * 6,
        }
    )
    s = RecallExperiment.summarize(df, ["a", "b"], "recall_covid")
    assert s["paired_difference"]["verdict"] == "inconclusivo"
    ci = mean_ci([1.0, 1.0, 1.0])
    assert ci["ci_low"] == ci["ci_high"] == 1.0


# ---------- dados reais: nenhuma imagem do teste no treino ----------


@pytest.mark.data
def test_nenhuma_imagem_do_teste_aparece_no_treino_por_hash():
    from visaocomp.a4.report import find_complete_run

    run = find_complete_run(repo_root() / "outputs")
    assert run is not None, "nenhuma execução completa da A4 em outputs/: rode `make run-a4`"
    split = pd.read_csv(run / "splits.csv", dtype={"phash": str})
    cfg = load_a4_config()
    test = split[split.split == "test"]
    rest = split[split.split != "test"]
    assert len(test) == 3 * cfg.case.test_per_class
    assert set(test.path).isdisjoint(rest.path)
    # recalcula o pHash dos arquivos (não confia no CSV) e compara todos os pares
    dedup = Deduplicator(cfg.dedup.hash_size, cfg.dedup.max_distance)
    h_test = dedup.hash_all([Path(p) for p in test.path])
    h_rest = dedup.hash_all([Path(p) for p in rest.path])
    assert [f"{h:016x}" for h in h_test] == test.phash.tolist()
    dist = hamming_matrix_rows(h_test, h_rest)
    assert int(dist.min()) > cfg.dedup.max_distance, "uma imagem do teste tem cópia no treino"
