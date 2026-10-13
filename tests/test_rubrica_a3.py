"""Um teste por item da rubrica da A3 (CNN pré-treinada).

R01 e as partes de texto de R03 a R05 rodam sem dados e sem GPU (entram no `make ci`).
As partes que leem `outputs/` têm o marcador `data`: precisam de um `make run-a3` feito.
"""

from __future__ import annotations

import json
import re

import pytest

from visaocomp.a3 import CnnClassifier, ImageScanner
from visaocomp.paths import data_dir, repo_root
from visaocomp.run import latest_manifest

SECTION = repo_root() / "reports" / "sections" / "a3.md"
AUGMENTATIONS = {
    "flip horizontal",
    "rotação leve",
    "recorte aleatório",
    "color jitter",
    "grayscale",
    "variação de escala",
}


def _latest_a3():
    """O manifesto e o metrics.json do run mais recente da A3."""
    outputs = repo_root() / "outputs"
    manifest = latest_manifest(outputs, "a3")
    assert manifest is not None, "nenhum run da A3 em outputs/: rode `make run-a3`"
    a3_dir = outputs / manifest["run_id"] / "a3"
    metrics = json.loads((a3_dir / "metrics.json").read_text(encoding="utf-8"))
    return manifest, metrics, a3_dir


def _sections(text: str, level: str = "## ") -> dict[str, str]:
    """Quebra o markdown em {título: corpo} pelos títulos do nível dado."""
    parts = re.split(rf"^{re.escape(level)}(.+)$", text, flags=re.MULTILINE)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts), 2)}


@pytest.fixture(scope="module")
def section_text() -> str:
    assert SECTION.is_file(), "reports/sections/a3.md não existe: rode `make run-a3`"
    return SECTION.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def section_classes(section_text) -> list[str]:
    found = re.search(r"^Classes do dataset: (.+)\.$", section_text, flags=re.MULTILINE)
    assert found, "o texto precisa listar as classes na linha 'Classes do dataset: ...'"
    return [c.strip() for c in found.group(1).split(",")]


def _find(sections: dict[str, str], *words: str) -> str:
    for title, body in sections.items():
        if all(w in title.lower() for w in words):
            return body
    raise AssertionError(f"seção com {words} não encontrada; achei {list(sections)}")


# -- R01 -----------------------------------------------------------------------------------
def test_R01_cnn_congelada_novo_head(image_folder):
    """Cabeça com o número de classes achado nas pastas; nenhum parâmetro do corpo treina."""
    classes = ImageScanner(image_folder).scan().classes
    model = CnnClassifier(len(classes), weights=None)
    assert model.head.out_features == len(classes) == 3
    body = [p for n, p in model.named_parameters() if not n.startswith("net.classifier.")]
    assert body and not any(p.requires_grad for p in body)
    assert all(p.requires_grad for p in model.head.parameters())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"net.classifier.1.weight", "net.classifier.1.bias"}


@pytest.mark.data
def test_R01_run_real_tem_cabeca_nova_e_corpo_congelado():
    """O run real (pesos do ImageNet) gravou a cabeça e o corpo no metrics.json e no checkpoint."""
    _, metrics, a3_dir = _latest_a3()
    model = metrics["model"]
    classes = ImageScanner(data_dir() / "raw/images_dataset/data").scan().classes
    assert metrics["classes"] == classes and metrics["n_classes"] == len(classes)
    assert model["weights"] == "IMAGENET1K_V1"
    assert model["head_out_features"] == len(classes)
    assert model["body_requires_grad"] is False
    assert model["trainable"] == model["head_in_features"] * len(classes) + len(classes)
    assert (a3_dir / "best_head.pt").stat().st_size > 0


# -- R02 -----------------------------------------------------------------------------------
@pytest.mark.data
def test_R02_curvas_e_accuracy_por_classe():
    manifest, metrics, a3_dir = _latest_a3()
    classes = metrics["classes"]
    assert 0.0 <= metrics["accuracy_global"] <= 1.0
    assert set(metrics["per_class"]) == set(classes), "toda classe tem de aparecer por classe"
    for name, row in metrics["per_class"].items():
        assert {"n", "correct", "accuracy"} <= set(row), name
        assert row["n"] > 0 and 0.0 <= row["accuracy"] <= 1.0, name
    cm = metrics["confusion_matrix"]
    assert len(cm) == len(classes) and sum(map(sum, cm)) == metrics["split"]["test"]
    hist = metrics["history"]
    epochs = metrics["train"]["epochs"]
    assert all(len(hist[k]) == epochs for k in ("train_loss", "train_acc", "val_loss", "val_acc"))
    assert (a3_dir / "curves.png").stat().st_size > 1000
    assert manifest["counts"]["files_kept"] == sum(
        metrics["split"][s] for s in ("train", "val", "test")
    )
    assert manifest["sources"][0]["version"] == 1 and len(manifest["sources"][0]["sha256"]) == 64
    assert metrics["baseline"]["accuracy"] < metrics["accuracy_global"], "não bate o baseline"


# -- R03 -----------------------------------------------------------------------------------
def test_R03_tres_augmentations(section_text, section_classes):
    """Pelo menos 3 augmentations, cada uma com a classe onde pode distorcer."""
    block = _find(_sections(section_text), "augmentation")
    items = _sections(block, "### ")
    named = {t: b for t, b in items.items() if any(a in t.lower() for a in AUGMENTATIONS)}
    assert len(named) >= 3, f"preciso de 3 augmentations nomeadas; achei {list(items)}"
    for title, body in named.items():
        why = re.search(r"Por que ajuda: (.+)", body)
        bad = re.search(r"Pode distorcer: (.+)", body)
        assert why and len(why.group(1)) > 40, f"{title}: falta a justificativa"
        assert bad, f"{title}: falta 'Pode distorcer'"
        assert any(re.search(rf"\b{c}\b", bad.group(1)) for c in section_classes), (
            f"{title}: 'Pode distorcer' não cita nenhuma classe do dataset {section_classes}"
        )


@pytest.mark.data
def test_R03_classes_do_texto_sao_as_do_run():
    _, metrics, _ = _latest_a3()
    text = SECTION.read_text(encoding="utf-8")
    assert f"Classes do dataset: {', '.join(metrics['classes'])}." in text
    assert metrics["run_id"] in text, "o texto veio de outro run: rode `make run-a3`"


# -- R04 -----------------------------------------------------------------------------------
def test_R04_feature_extraction_vs_finetuning(section_text):
    body = _find(_sections(section_text), "feature extraction", "fine-tuning").lower()
    for word in ("feature extraction", "fine-tuning", "tamanho", "domínio", "imagenet"):
        assert word in body, f"R04: falta '{word}'"
    assert re.search(r"dataset (pequeno|grande)", body), "R04: falta o critério de tamanho"
    assert "a4" in body, "R04: falta o exemplo de domínio diferente (A4)"


# -- R05 -----------------------------------------------------------------------------------
def test_R05_escolha_do_modelo_t4(section_text):
    body = _find(_sections(section_text), "escolha do modelo")
    for word in ("EfficientNet-B0", "T4", "ResNet-50", "classes", "GFLOPs"):
        assert word in body, f"R05: falta '{word}'"
    assert re.search(r"\b\d+ classes\b", body), "R05: falta o número de classes"
    assert "$" not in body, "R05: sobrou um espaço do modelo sem preencher"


@pytest.mark.data
def test_R05_numeros_do_texto_batem_com_o_metrics_json():
    _, metrics, _ = _latest_a3()
    text = SECTION.read_text(encoding="utf-8")
    assert f"{metrics['accuracy_global'] * 100:.1f}%".replace(".", ",") in text
    assert f"{metrics['model']['trainable']:,}".replace(",", ".") in text
    assert f"{metrics['n_classes']} classes" in text
