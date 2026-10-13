"""Um teste por item da rubrica da A4.

R22, R23 e R24 leem a execução completa mais recente em outputs/ (`make run-a4`), por isso têm o
marcador `data` (ficam fora do `make test`). R21 e R26 leem reports/sections/a4.md, que está no git.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from visaocomp.a4.config import load_a4_config
from visaocomp.paths import repo_root

A4_MD = repo_root() / "reports" / "sections" / "a4.md"


def a4_run() -> Path:
    from visaocomp.a4.report import find_complete_run

    run = find_complete_run(repo_root() / "outputs")
    assert run is not None, "nenhuma execução completa da A4 em outputs/: rode `make run-a4`"
    return run


def section(md: str, title: str) -> str:
    """Texto de uma seção `## <title>` até a próxima `## `."""
    m = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", md, flags=re.M | re.S)
    assert m, f"a4.md não tem a seção '## {title}'"
    return m.group(1)


def test_R21_cinco_problemas_raio_x():
    sec = section(A4_MD.read_text(encoding="utf-8"), "Problemas do projeto anterior")
    problems = re.split(r"^### ", sec, flags=re.M)[1:]
    assert len(problems) >= 5, f"esperava 5+ problemas, achei {len(problems)}"
    for p in problems:
        title = p.splitlines()[0]
        assert "Impacto clínico" in p, f"problema sem impacto clínico: {title}"
        assert len(p.split()) > 40, f"problema sem justificativa: {title}"


@pytest.mark.data
def test_R22_gan_com_loop_adversarial():
    run = a4_run()
    gan = json.loads((run / "gan_metrics.json").read_text(encoding="utf-8"))
    cfg = load_a4_config()
    split = pd.read_csv(run / "splits.csv")
    n_train = int((split["split"] == "train").sum())
    batches = n_train // cfg.gan.batch_size
    assert gan["fid_reference"]["n_real"] == n_train, "a GAN treina só com o treino do caso"
    for name in ("sem_mitigacao", "com_mitigacao"):
        r = gan["runs"][name]
        assert r["n_d_steps"] == r["n_g_steps"] == r["epochs"] * batches, (
            "o loop tem de alternar D e G, um passo de cada por lote"
        )
        grids = sorted((run / "gan" / name).glob("samples_epoch_*.png"))
        assert len(grids) >= 3, f"{name}: faltam as grades samples_epoch_*.png"
        assert (run / "gan" / name / "generator_best.pt").exists()
    from visaocomp.a4.gan import GanTrainer

    assert hasattr(GanTrainer, "discriminator_step") and hasattr(GanTrainer, "generator_step")


@pytest.mark.data
def test_R23_instabilidade_e_mitigacao():
    run = a4_run()
    gan = json.loads((run / "gan_metrics.json").read_text(encoding="utf-8"))
    sem, com = gan["runs"]["sem_mitigacao"], gan["runs"]["com_mitigacao"]
    assert sem["strategies"] == [] and len(com["strategies"]) >= 1
    for r in (sem, com):
        assert isinstance(r["fid_final"], float) and r["fid_final"] > 0
        assert len(r["fid_curve"]) >= 2
        n = r["epochs"]
        assert all(len(r["curves"][k]) == n for k in ("loss_d", "loss_g", "d_real", "d_fake"))
        diag = r["diagnostico"]
        for k in ("discriminador_dominou", "fid_subiu_depois_do_melhor", "colapso_de_modo"):
            assert isinstance(diag[k], bool)
    assert isinstance(gan["comparacao"]["melhora_fid"], bool)
    assert (run / "gan_comparison.png").exists()
    sec = section(A4_MD.read_text(encoding="utf-8"), "Instabilidade e mitigação")
    assert "FID" in sec and "colapso" in sec.lower()


@pytest.mark.data
def test_R24_recall_covid_com_e_sem_sintetico():
    run = a4_run()
    cfg = load_a4_config()
    df = pd.read_csv(run / "recall" / "recall.csv")
    assert set(df["condition"]) == {"sem_sintetico", "com_sintetico"}
    for cond, sub in df.groupby("condition"):
        assert sorted(sub["seed"]) == sorted(cfg.classifier.seeds), cond
        assert len(sub) >= 3
    assert df["recall_covid"].between(0, 1).all()
    assert (df["n_test_covid"] == cfg.case.test_per_class).all()
    assert df["test_real_only"].all()
    com = df[df.condition == "com_sintetico"]
    sem = df[df.condition == "sem_sintetico"]
    assert (com["n_synthetic"] == cfg.synthetic.n_covid).all() and (sem["n_synthetic"] == 0).all()
    # o teste é o mesmo nas duas condições e só tem imagens reais do split de teste
    preds = pd.read_csv(run / "recall" / "test_predictions.csv")
    split = pd.read_csv(run / "splits.csv")
    test_paths = set(split.loc[split.split == "test", "path"])
    assert set(preds["path"]) == test_paths
    assert not preds["path"].str.contains("synthetic").any()
    summary = json.loads((run / "recall" / "recall_summary.json").read_text(encoding="utf-8"))
    for cond in ("sem_sintetico", "com_sintetico"):
        s = summary["conditions"][cond]["recall_covid"]
        assert s["ci_low"] <= s["mean"] <= s["ci_high"]
    assert summary["paired_difference"]["verdict"] in {"melhora", "piora", "inconclusivo"}


def test_R25_problemas_trafego_e_transfer():
    """4+ problemas do projeto de tráfego, cada um com impacto e abordagem; riscos do ImageNet."""
    md = (repo_root() / "reports" / "sections" / "a4_2.md").read_text(encoding="utf-8")
    probs = section(md, "Problemas, impacto operacional e como eu abordaria")
    items = re.split(r"^### ", probs, flags=re.M)[1:]
    assert len(items) >= 4, f"só {len(items)} problemas"
    for item in items:
        assert "Impacto" in item and "Como abordar" in item, item.splitlines()[0]
    risks = section(md, "Riscos do transfer learning do ImageNet para classificar fluxo")
    assert "ImageNet" in risks and len(re.findall(r"^\d\. ", risks, flags=re.M)) >= 3
    for term in ("chuva", "noite", "ângulo", "câmera"):
        assert term in md.lower(), term


def test_R26_plano_de_melhoria():
    sec = section(A4_MD.read_text(encoding="utf-8"), "Plano de melhoria integrado")
    for sub in ("Modelo", "Métrica", "Augmentation sintética", "Critério de adoção clínica"):
        m = re.search(rf"^### {sub}\s*$(.*?)(?=^### |\Z)", sec, flags=re.M | re.S)
        assert m, f"o plano não tem a parte '### {sub}'"
        assert len(m.group(1).split()) > 30, f"a parte '{sub}' está vazia"
    assert "recall" in sec.lower()
