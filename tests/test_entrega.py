"""Entrega: os 4 notebooks executados e a rubrica sem nenhum skip."""

import json

import pytest

from visaocomp.paths import repo_root

ROOT = repo_root()
NOTEBOOKS = [
    "A1_vision_transformers.ipynb",
    "A2_clip_ads16.ipynb",
    "A3_cnn_kaggle.ipynb",
    "A4_estudo_caso_raio_x.ipynb",
]


def test_rubrica_sem_skip():
    for path in sorted((ROOT / "tests").glob("test_rubrica_a*.py")):
        assert "pytest.skip(" not in path.read_text(encoding="utf-8"), path.name


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_executado_sem_erro(name):
    nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    first = "".join(nb["cells"][0]["source"])
    assert "Memória" in first and "Tempo" in first, f"{name}: requisitos no topo"
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["execution_count"] is not None for c in code), f"{name}: não executado"
    for c in code:
        for out in c.get("outputs", []):
            assert out["output_type"] != "error", f"{name}: célula com erro"
