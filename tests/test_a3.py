"""Testes unitários da A3: varredura, split, modelo, métricas, treino e o pipeline minúsculo."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from visaocomp.a3 import (
    A3Config,
    A3Pipeline,
    CnnClassifier,
    Evaluator,
    ImageFolderDataset,
    ImageScanner,
    Trainer,
    build_transform,
    compute_metrics,
    majority_baseline,
    stratified_split,
)
from visaocomp.device import pick_device

CPU = torch.device("cpu")


def small_cfg(**kw) -> A3Config:
    return A3Config(weights=None, crop=64, resize=72, epochs=2, batch_size=4, num_workers=0, **kw)


# -- varredura (contrato de dados) ---------------------------------------------------------
def test_scanner_acha_classes_pula_corrompida_e_tira_duplicata(image_folder):
    rep = ImageScanner(image_folder).scan()
    assert rep.classes == ["azul", "rara", "verde"]
    assert rep.kept_per_class == {"azul": 8, "rara": 4, "verde": 8}
    assert [s["path"] for s in rep.skipped] == ["azul/quebrada.bmp"]
    # em ordem alfabética, fica o primeiro (copia.png) e sai o outro
    assert rep.duplicates == [{"path": "verde/verde_0.png", "same_as": "verde/copia.png"}]
    assert rep.ignored_dirs == ["data"], "a cópia aninhada data/data não vira classe"
    assert rep.counts() == {
        "files_found": 22, "files_kept": 20, "files_skipped_unreadable": 1,
        "files_removed_duplicate": 1, "dirs_ignored": 1,
    }  # fmt: skip


def test_scanner_sem_dedupe_mantem_o_duplicado(image_folder):
    rep = ImageScanner(image_folder, dedupe=False).scan()
    assert len(rep.records) == 21 and rep.duplicates == []


def test_scanner_pasta_sem_imagens_para_cedo(tmp_path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(ValueError, match="nenhuma classe"):
        ImageScanner(tmp_path).scan()
    with pytest.raises(FileNotFoundError):
        ImageScanner(tmp_path / "nao-existe").scan()


def test_dataset_devolve_tensor_rgb_em_0_1_mesmo_com_png_rgba(tmp_path):
    from PIL import Image

    (tmp_path / "classe").mkdir()
    Image.new("RGBA", (20, 20), (255, 0, 0, 10)).save(tmp_path / "classe" / "um.png")
    rep = ImageScanner(tmp_path).scan()
    ds = ImageFolderDataset(tmp_path, rep.records, build_transform(72, 64))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0 and y == 0


# -- split ---------------------------------------------------------------------------------
def test_split_estratificado_sem_sobreposicao_e_deterministico():
    labels = [0] * 100 + [1] * 50 + [2] * 20
    a = stratified_split(labels, 0.70, 0.15, seed=42)
    b = stratified_split(labels, 0.70, 0.15, seed=42)
    c = stratified_split(labels, 0.70, 0.15, seed=7)
    assert a == b and a != c
    assert not set(a.train) & set(a.val) | set(a.train) & set(a.test) | set(a.val) & set(a.test)
    assert len(a.train) + len(a.val) + len(a.test) == len(labels)
    for k, n in ((0, 100), (1, 50), (2, 20)):
        in_train = sum(labels[i] == k for i in a.train)
        assert abs(in_train / n - 0.70) < 0.06, f"classe {k} fora da proporção"


def test_split_classe_minuscula_ainda_deixa_uma_imagem_no_treino():
    split = stratified_split([0] * 30 + [1] * 2 + [2], 0.70, 0.15, seed=1)
    for idx in (range(30, 32), [32]):
        assert any(i in split.train for i in idx)
    assert 32 in split.train


# -- modelo --------------------------------------------------------------------------------
def test_modelo_cabeca_nova_e_corpo_congelado():
    model = CnnClassifier(7, weights=None)
    assert model.head.out_features == 7
    body = [p for n, p in model.named_parameters() if not n.startswith("net.classifier.")]
    assert body and not any(p.requires_grad for p in body)
    counts = model.param_counts()
    assert counts["trainable"] == model.head.in_features * 7 + 7
    assert counts["head_out_features"] == 7 and counts["frozen"] > 4_000_000


def test_modelo_train_mantem_o_corpo_em_eval_e_a_saida_tem_a_forma_certa():
    model = CnnClassifier(3, weights=None)
    model.train()
    assert model.training and not model.net.features.training
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 3)
    assert out.requires_grad  # o gradiente chega na cabeça


def test_modelo_normaliza_dentro_e_exige_duas_classes():
    model = CnnClassifier(2, weights=None)
    assert torch.allclose(model.mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    with pytest.raises(ValueError):
        CnnClassifier(1, weights=None)


# -- métricas ------------------------------------------------------------------------------
def test_metricas_por_classe_nao_somem_na_media():
    labels = np.array([0, 0, 0, 0, 1, 1, 2])
    preds = np.array([0, 0, 0, 1, 1, 0, 0])
    m = compute_metrics(labels, preds, ["a", "b", "c", "d"])
    assert m["accuracy_global"] == pytest.approx(4 / 7)
    assert m["per_class"]["a"]["accuracy"] == pytest.approx(3 / 4)
    assert m["per_class"]["c"] == {
        "n": 1,
        "correct": 0,
        "accuracy": 0.0,
        "precision": None,
        "f1": 0.0,
    }
    assert m["per_class"]["d"]["n"] == 0 and m["per_class"]["d"]["accuracy"] is None
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2 + 0) / 3)
    assert m["confusion_matrix"][0] == [3, 1, 0, 0]
    assert m["top_confusions"][0] == {"true": "a", "pred": "b", "count": 1}


def test_baseline_classe_mais_comum():
    base = majority_baseline([0, 0, 1], np.array([0, 1, 1, 1]), ["a", "b"])
    assert base["class"] == "a" and base["accuracy"] == pytest.approx(0.25)


def test_pick_device_respeita_o_pedido():
    assert pick_device("cpu") == CPU
    assert pick_device().type in {"cuda", "mps", "cpu"}


# -- treino --------------------------------------------------------------------------------
def _loaders(root, cfg):
    rep = ImageScanner(root).scan()
    ds = ImageFolderDataset(root, rep.records, build_transform(cfg.resize, cfg.crop))
    split = stratified_split([r.label for r in rep.records], cfg.train_frac, cfg.val_frac, 42)
    mk = lambda idx, sh: DataLoader(ds.subset(idx), batch_size=4, shuffle=sh)  # noqa: E731
    return rep, mk(split.train, True), mk(split.val, False), mk(split.test, False)


def test_trainer_so_muda_a_cabeca_e_guarda_o_melhor_checkpoint(image_folder, tmp_path):
    cfg = small_cfg(lr=1e-2)
    rep, train, val, _ = _loaders(image_folder, cfg)
    torch.manual_seed(0)
    model = CnnClassifier(len(rep.classes), weights=None)
    body_before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    bn_before = model.net.features[0][1].running_mean.clone()
    head_before = model.head.weight.clone()
    hist = Trainer(model, cfg, CPU, checkpoint=tmp_path / "head.pt").fit(train, val)
    assert len(hist.train_loss) == len(hist.val_acc) == 2 and 1 <= hist.best_epoch <= 2
    assert all(np.isfinite(hist.train_loss))
    assert not torch.equal(head_before, model.head.weight)
    assert all(torch.equal(p, dict(model.named_parameters())[n]) for n, p in body_before.items())
    assert torch.equal(bn_before, model.net.features[0][1].running_mean), "BatchNorm mexeu"
    saved = torch.load(tmp_path / "head.pt")
    assert set(saved) == {"net.classifier.1.weight", "net.classifier.1.bias"}


def test_evaluator_preve_o_conjunto_inteiro(image_folder):
    cfg = small_cfg()
    rep, _, _, test = _loaders(image_folder, cfg)
    model = CnnClassifier(len(rep.classes), weights=None)
    pred = Evaluator(model, CPU).predict(test)
    assert len(pred.labels) == len(pred.preds) == len(test.dataset)
    assert np.isfinite(pred.loss) and (pred.confidence <= 1).all()


# -- pipeline minúsculo, em CPU, sem rede --------------------------------------------------
@pytest.mark.smoke
def test_pipeline_a3_inteiro_em_amostra_minuscula(image_folder, tmp_path):
    cfg = small_cfg(lr=1e-2)
    section = tmp_path / "reports" / "a3.md"
    pipe = A3Pipeline(cfg, image_folder, {"name": "fixture"}, tmp_path / "outputs", CPU, section)
    manifest_path = pipe.run()
    run_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "a3" and manifest["counts"]["files_skipped_unreadable"] == 1
    assert manifest["counts"]["files_removed_duplicate"] == 1
    assert manifest["skipped_files"][0]["path"] == "azul/quebrada.bmp"
    metrics = json.loads((run_dir / "a3" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["classes"] == ["azul", "rara", "verde"] and set(metrics["per_class"]) == set(
        metrics["classes"]
    )
    assert metrics["model"]["head_out_features"] == 3 and not metrics["model"]["body_requires_grad"]
    assert 0.0 <= metrics["accuracy_global"] <= 1.0 and len(metrics["history"]["val_acc"]) == 2
    assert (run_dir / "a3" / "curves.png").stat().st_size > 0
    assert (run_dir / "a3" / "confusion.png").stat().st_size > 0
    text = section.read_text(encoding="utf-8")
    assert "$" not in text and "Classes do dataset: azul, rara, verde." in text
    # um segundo run não sobrescreve o primeiro
    again = A3Pipeline(cfg, image_folder, {"name": "fixture"}, tmp_path / "outputs", CPU).run()
    assert again.parent != run_dir


def test_notebook_a3_e_um_orquestrador_fino_com_requisitos_no_topo():
    import nbformat

    from visaocomp.paths import repo_root

    nb = nbformat.read(repo_root() / "notebooks" / "A3_cnn_kaggle.ipynb", 4)
    first = nb.cells[0]
    assert first.cell_type == "markdown" and "Requisitos" in first.source
    assert "Memória" in first.source and "Tempo" in first.source
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "from visaocomp.a3.cli import run_a3" in code
    assert "class " not in code and "nn.Module" not in code, "a lógica mora no pacote"
