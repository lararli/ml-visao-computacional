"""Treino multirrótulo da A1: perda com rótulo vazio, métricas, contrato dos dados e mapas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from a1_scratch_world import build_world
from visaocomp.a1.chips import (
    ChipCatalogReader,
    DataContractError,
    RegionMasker,
    augment,
    check_matrix,
    label_matrix,
)
from visaocomp.a1.train import (
    AttentionMapper,
    MaskedBCELoss,
    attribute_metrics,
    best_f1_threshold,
    positive_weights,
)

ATTRS = ("pool", "solar")


def test_perda_ignora_rotulo_vazio():
    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    got = MaskedBCELoss()(logits, targets)
    valid = torch.tensor([True, False, True, True])
    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.flatten()[valid], torch.tensor([1.0, 0.0, 1.0])
    )
    assert torch.allclose(got, ref)


def test_perda_sem_nenhum_rotulo_valido_da_zero_sem_nan():
    logits = torch.randn(3, 2, requires_grad=True)
    loss = MaskedBCELoss(torch.tensor([3.0, 10.0]))(logits, torch.full((3, 2), float("nan")))
    loss.backward()
    assert loss.item() == 0.0 and not torch.isnan(logits.grad).any()


def test_atributo_sem_positivo_no_lote_nao_gera_nan():
    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, float("nan")], [0.0, 0.0]])
    loss = MaskedBCELoss(torch.tensor([5.0, 5.0]))(logits, targets)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_peso_dos_positivos_conta_so_rotulos_validos():
    y = np.array([[1, 0], [0, 0], [0, np.nan], [0, 1]], dtype=np.float32)
    assert positive_weights(y).tolist() == [3.0, 2.0]


def test_metricas_ignoram_vazio_e_dao_pr_auc():
    y = np.array([1, 0, np.nan, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.99, 0.4, 0.6, 0.1])
    m = attribute_metrics(y, p, threshold=0.5)
    assert m["n"] == 5 and m["n_pos"] == 2
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert 0 < m["pr_auc"] <= 1 and m["prevalence"] == pytest.approx(0.4)


def test_metricas_sem_positivo_nao_quebram():
    m = attribute_metrics(np.array([0.0, 0.0]), np.array([0.1, 0.7]), 0.5)
    assert m["pr_auc"] is None and m["recall"] == 0.0 and m["confusion"]["fp"] == 1


def test_limiar_de_maior_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.35, 0.8])
    assert best_f1_threshold(y, p) == pytest.approx(0.35)
    assert best_f1_threshold(np.zeros(3), np.ones(3)) == 0.5


def test_leitor_do_catalogo_traz_rotulo_vazio_e_conferencia(tmp_path):
    cat = build_world(tmp_path)
    chips, pools = ChipCatalogReader(cat, ATTRS).read()
    assert len(chips) == 30 and len(pools) == 15
    lab = label_matrix(chips, ATTRS)
    assert np.isnan(lab[chips["chip_id"] == "chip004", 0]).all()
    chk = check_matrix(chips, ATTRS)
    test = chips["split"].eq("test").to_numpy()
    assert np.isnan(chk[~test]).all()
    assert np.isnan(chk[chips["chip_id"] == "chip029", 1]).all()  # "não dá" fica fora
    assert set(np.unique(chk[test][~np.isnan(chk[test])])) == {0.0, 1.0}


def test_contrato_recusa_split_vazio_e_vazio_sem_motivo():
    reader = ChipCatalogReader("x.duckdb", ATTRS)
    base = pd.DataFrame({
        "chip_id": ["a", "b", "c"], "path": ["p"] * 3, "split": ["train", "val", "test"],
        "x_center": [0.0] * 3, "y_center": [0.0] * 3, "epsg": [26914] * 3, "house_wkt": [None] * 3,
        "pool": [1, 0, 0], "solar": [1, 0, 0], "pool_reason": [None] * 3,
        "solar_reason": [None] * 3,
    })  # fmt: skip
    reader.validate(base)
    with pytest.raises(DataContractError, match="split test"):
        reader.validate(base.assign(split=["train", "val", "val"]))
    with pytest.raises(DataContractError, match="sem motivo"):
        reader.validate(base.assign(pool=[1, None, 0]))
    with pytest.raises(DataContractError, match="faltam colunas"):
        reader.validate(base.drop(columns=["solar"]))


def test_augmentation_so_gira_e_espelha():
    x = torch.rand(8, 3, 16, 16)
    y = augment(x, torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    for a, b in zip(x, y, strict=True):  # mesmo conteúdo, só reordenado e com brilho escalado
        ratio = b.sum() / a.sum()
        assert torch.allclose(
            torch.sort(a.flatten()).values * ratio, torch.sort(b.flatten()).values, atol=1e-4
        )


def test_mascaras_de_regiao_poem_telhado_no_centro_e_piscina_a_leste(tmp_path):
    cat = build_world(tmp_path)
    chips, pools = ChipCatalogReader(cat, ATTRS).read()
    masks = RegionMasker(96, 8, 0.6).masks(chips.head(2), pools)
    roof, pool = masks.roof[1], masks.pool[1]  # chip001 tem piscina
    assert roof.shape == (12, 12) and roof[0, 0] == 0.0
    # 16 x 16 px = 4 patches de 8 x 8 (folga: meio pixel e a ida e volta por lon/lat)
    assert roof.sum() == pytest.approx(4.0, abs=0.25)
    assert roof[5:7, 5:7].min() > 0.7 and roof[5:7, 5:7].sum() > 3.5
    assert pool[5:7, 8:10].sum() > 0 and pool[:, :6].sum() == 0
    assert masks.pool[0].sum() == 0  # chip000 não tem piscina


def test_razao_de_atencao_na_regiao():
    maps = np.zeros((2, 1, 2, 2))
    maps[:, 0, 0, 0] = 1.0  # toda a atenção no patch (0, 0)
    masks = np.zeros((2, 2, 2))
    masks[0, 0, 0] = 1.0  # região = 1/4 da imagem, recebe tudo
    r = AttentionMapper.region_lift(maps, masks)
    assert r["n"] == 1 and r["mean_heads"] == pytest.approx(4.0) and r["attention_share"] == 1.0


def test_mapa_ampliado_para_o_tamanho_da_imagem(tmp_path):
    mapper = AttentionMapper(96)
    maps = np.random.default_rng(0).random((3, 12, 12)).astype(np.float32)
    assert mapper.upsample(maps).shape == (3, 96, 96)
    img = np.zeros((3, 96, 96), dtype=np.uint8)
    out = mapper.figure(img, maps, "teste", tmp_path / "attention_x.png", highlight=1)
    assert out.stat().st_size > 1000
