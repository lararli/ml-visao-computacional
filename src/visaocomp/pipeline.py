"""O pipeline da entrega: uma atividade = um dataset = silver -> modelos -> avaliação -> gold.

`make pipeline` roda, para cada dataset, o treino e a avaliação (só se ainda não há resultado em
outputs/, ou com --force) e depois publica a gold no `catalog.duckdb`:

| Atividade | Dataset | Gold |
|---|---|---|
| A1 | austin_houses (NAIP + Austin) | gold.austin_house_attribute (+ _evidence) |
| A2 | ads16 (Kaggle ADS-16) | gold.ads16_ad_object |
| A3 | images (Kaggle images-dataset) | gold.images_prediction |
| A4.1 | chest_xray (Kaggle COVID, pneumonia, normal) | gold.xray_prediction |

E `gold.model_run`: todo modelo treinado ou usado, com dataset, versão (hash), métrica e run_id.
A escrita é MERGE pela chave (rodar de novo não duplica; um modelo novo não apaga o antigo).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from visaocomp.a1.finetune import file_hash
from visaocomp.a1.pipeline import KEYS, A1Catalog
from visaocomp.paths import data_dir, repo_root

log = logging.getLogger(__name__)

MODEL_RUN = "gold.model_run"
ADS16_OBJECT = "gold.ads16_ad_object"
IMAGES_PREDICTION = "gold.images_prediction"
XRAY_PREDICTION = "gold.xray_prediction"
KEYS.setdefault(MODEL_RUN, ["activity", "model", "model_version"])
KEYS.setdefault(ADS16_OBJECT, ["image_path", "description", "model_version"])
KEYS.setdefault(IMAGES_PREDICTION, ["sha256", "model_version"])
KEYS.setdefault(XRAY_PREDICTION, ["image", "condition", "seed", "model_version"])


@dataclass(frozen=True)
class ModelRun:
    """Uma linha de `gold.model_run`."""

    activity: str
    dataset: str
    model: str
    model_version: str
    run_id: str
    metric: str
    value: float
    metrics: dict[str, Any]
    output_dir: str

    def row(self) -> dict[str, Any]:
        return {
            "activity": self.activity,
            "dataset": self.dataset,
            "model": self.model,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "metric": self.metric,
            "value": float(self.value),
            "metrics_json": json.dumps(self.metrics, ensure_ascii=False),
            "output_dir": self.output_dir,
        }


def _latest(outputs: Path, pattern: str, required: str) -> Path | None:
    """O run mais recente que bate com `pattern` e tem o arquivo `required`."""
    runs = sorted(p for p in outputs.glob(pattern) if (p / required).exists())
    return runs[-1] if runs else None


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class DatasetPipeline(ABC):
    """Um dataset do pipeline: como treinar e como virar gold a partir de outputs/."""

    activity: str = ""
    dataset: str = ""
    scripts: tuple[list[str], ...] = ()

    def __init__(self, root: Path, outputs: Path) -> None:
        self.root = Path(root)
        self.outputs = Path(outputs)

    @abstractmethod
    def has_results(self) -> bool:
        """Verdadeiro se outputs/ já tem o resultado que a gold precisa."""

    @abstractmethod
    def gold(self) -> tuple[dict[str, pd.DataFrame], list[ModelRun]]:
        """Tabelas gold do dataset e as linhas de `gold.model_run`."""

    def train(self) -> None:
        """Roda os mesmos scripts do Makefile (treino, avaliação e a seção do relatório)."""
        for args in self.scripts:
            log.info("%s: %s", self.dataset, " ".join(args))
            subprocess.run([sys.executable, *args], cwd=self.root, check=True)

    def _src(self, df: pd.DataFrame, run_id: str) -> pd.DataFrame:
        df = df.copy()
        df["run_id"] = run_id
        df["_source"] = f"pipeline:{self.dataset}"
        return df


class AustinHousesPipeline(DatasetPipeline):
    """A1: ViT do zero, ViT-S pré-treinado e ResNet-18; a gold vem do modelo escolhido."""

    activity, dataset = "A1", "austin_houses"
    scripts = (["scripts/run_a1_scratch.py"], ["scripts/run_a1_pretrained.py"])

    def _runs(self) -> tuple[Path | None, Path | None]:
        scratch = _latest(self.outputs, "*-a1_scratch", "a1_scratch/metrics.json")
        pre = _latest(self.outputs, "*-a1_pretrained", "a1_pretrained/abstain.json")
        return scratch, pre

    def has_results(self) -> bool:
        return all(self._runs())

    def gold(self) -> tuple[dict[str, pd.DataFrame], list[ModelRun]]:
        # gold.austin_house_attribute e _evidence: o post do A1 (regra de "não dá" e GoldWriter)
        # grava por MERGE; aqui ele roda de novo para valer também num catálogo novo.
        from visaocomp.a1.pretrained_cli import post

        post(self.root, self.outputs)
        scratch, pre = self._runs()
        assert scratch is not None and pre is not None
        runs = []
        s = _json(scratch / "a1_scratch" / "metrics.json")
        runs.append(
            ModelRun(
                self.activity, self.dataset, "vit_scratch",
                f"vit_scratch-{file_hash(scratch / 'a1_scratch' / 'best.pt')}", s["run_id"],
                "pr_auc_medio_validacao", s["train"]["best_val_pr_auc_mean"],
                {"test": s["test"]["per_attribute"],
                 "test_reviewed": s["test_reviewed"]["per_attribute"]},
                str(scratch.relative_to(self.root)),
            )
        )  # fmt: skip
        m = _json(pre / "a1_pretrained" / "metrics.json")
        for key, mm in m["models"].items():
            runs.append(
                ModelRun(
                    self.activity, self.dataset, key, mm["model_version"], m["run_id"],
                    "pr_auc_medio_validacao", mm["train"]["best_val_pr_auc_mean"],
                    {"test": mm["test"]["per_attribute"],
                     "test_reviewed": mm["test_reviewed"]["per_attribute"]},
                    str(pre.relative_to(self.root)),
                )
            )  # fmt: skip
        return {}, runs


class Ads16Pipeline(DatasetPipeline):
    """A2: CLIP sem treino; a gold diz quais objetos aparecem em cada anúncio."""

    activity, dataset = "A2", "ads16"
    scripts = (["scripts/run_a2.py"],)

    def _run(self) -> Path | None:
        return _latest(self.outputs, "*-a2", "a2/scores.csv")

    def has_results(self) -> bool:
        return self._run() is not None

    def gold(self) -> tuple[dict[str, pd.DataFrame], list[ModelRun]]:
        run = self._run()
        assert run is not None
        out = run / "a2"
        manifest = _json(run / "manifest.json")
        thr = _json(out / "threshold.json")
        model = str(manifest.get("config", {}).get("model", "openai/clip-vit-base-patch32"))
        version = model.split("/")[-1]
        scores = pd.read_csv(out / "scores.csv")
        long = scores.melt(id_vars=["image_path", "source"], var_name="description",
                           value_name="score")  # fmt: skip
        long["rank"] = long.groupby("image_path")["score"].rank(ascending=False, method="first")
        long["present"] = long["score"] >= thr["threshold"]
        keep = long[long["present"] | (long["rank"] == 1)].copy()  # presentes, e sempre o top-1
        keep["rank"] = keep["rank"].astype(int)
        keep["threshold"] = thr["threshold"]
        keep["model_version"] = version
        table = self._src(keep, run.name)
        runs = [
            ModelRun(
                self.activity, self.dataset, "clip_zero_shot", version, run.name,
                "acerto_do_limiar_conferido", thr["accuracy"],
                {"threshold": thr["threshold"], "n_conferidos": thr.get("n_labeled")},
                str(run.relative_to(self.root)),
            )
        ]  # fmt: skip
        return {ADS16_OBJECT: table.reset_index(drop=True)}, runs


class ImagesPipeline(DatasetPipeline):
    """A3: EfficientNet-B0 congelada; a gold é a classe prevista de cada imagem do teste."""

    activity, dataset = "A3", "images"
    scripts = (["scripts/run_a3.py"],)

    def _run(self) -> Path | None:
        return _latest(self.outputs, "*-a3", "a3/predictions.csv")

    def has_results(self) -> bool:
        return self._run() is not None

    def gold(self) -> tuple[dict[str, pd.DataFrame], list[ModelRun]]:
        run = self._run()
        assert run is not None
        m = _json(run / "a3" / "metrics.json")
        version = f"{m['model']['arch']}-{file_hash(run / 'a3' / 'best_head.pt')}"
        pred = pd.read_csv(run / "a3" / "predictions.csv").rename(
            columns={"true": "true_class", "pred": "pred_class"}
        )
        pred["correct"] = pred["true_class"] == pred["pred_class"]
        pred["model_version"] = version
        runs = [
            ModelRun(
                self.activity, self.dataset, m["model"]["arch"], version, m["run_id"],
                "accuracy_global_teste", m["accuracy_global"],
                {k: m[k] for k in ("balanced_accuracy", "macro_f1", "per_class")},
                str(run.relative_to(self.root)),
            )
        ]  # fmt: skip
        return {IMAGES_PREDICTION: self._src(pred, m["run_id"])}, runs


class ChestXrayPipeline(DatasetPipeline):
    """A4.1: GAN condicional e ResNet-18 sem e com sintético; a gold é a previsão do teste real."""

    activity, dataset = "A4.1", "chest_xray"
    scripts = (["scripts/run_a4.py"],)

    def _run(self) -> Path | None:
        return _latest(self.outputs, "*-a4", "recall/test_predictions.csv")

    def has_results(self) -> bool:
        return self._run() is not None

    def gold(self) -> tuple[dict[str, pd.DataFrame], list[ModelRun]]:
        run = self._run()
        assert run is not None
        cfg = _json(run / "config.json")
        classes = list(cfg["classes"])
        pred = pd.read_csv(run / "recall" / "test_predictions.csv")
        pred["image"] = pred["path"].str.replace(r"^.*?/raw/", "", regex=True)  # relativo a raw/
        pred["true_class"] = pred["true"].map(lambda k: classes[int(k)])
        pred["pred_class"] = pred["pred"].map(lambda k: classes[int(k)])
        pred["model_version"] = "resnet18_" + pred["condition"] + "-" + run.name
        table = pred[["image", "condition", "seed", "true_class", "pred_class", "prob_covid",
                      "model_version"]]  # fmt: skip
        summary = _json(run / "recall" / "recall_summary.json")
        runs = [
            ModelRun(
                self.activity, self.dataset, f"resnet18_{cond}", f"resnet18_{cond}-{run.name}",
                run.name, "recall_covid_medio_teste_real", s["recall_covid"]["mean"], s,
                str(run.relative_to(self.root)),
            )
            for cond, s in summary["conditions"].items()
        ]  # fmt: skip
        gan = _json(run / "gan_metrics.json")
        for name, g in gan["runs"].items():
            best = min(p["fid"] for p in g["fid_curve"])
            runs.append(
                ModelRun(
                    self.activity, self.dataset, f"gan_{name}", f"gan_{name}-{run.name}",
                    run.name, "melhor_fid", best,
                    {"epochs": g["epochs"], "strategies": g["strategies"]},
                    str(run.relative_to(self.root)),
                )
            )  # fmt: skip
        return {XRAY_PREDICTION: self._src(table, run.name)}, runs


DATASETS: dict[str, type[DatasetPipeline]] = {
    "austin_houses": AustinHousesPipeline,
    "ads16": Ads16Pipeline,
    "images": ImagesPipeline,
    "chest_xray": ChestXrayPipeline,
}


class PipelineRunner:
    """Para cada dataset: treina se precisa (ou se forçado) e publica a gold."""

    def __init__(self, names: list[str] | None = None, root: Path | None = None) -> None:
        self.root = root or repo_root()
        outputs = self.root / "outputs"
        self.datasets = [DATASETS[n](self.root, outputs) for n in (names or list(DATASETS))]
        self.catalog = A1Catalog(data_dir() / "catalog.duckdb")

    def run(self, force: bool = False, gold_only: bool = False) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for ds in self.datasets:
            if not gold_only and (force or not ds.has_results()):
                ds.train()
            if not ds.has_results():
                raise RuntimeError(f"{ds.dataset}: sem resultado em outputs/; rode sem --gold-only")
            tables, runs = ds.gold()
            tables[MODEL_RUN] = pd.DataFrame([r.row() for r in runs]).assign(
                _source=f"pipeline:{ds.dataset}"
            )
            counts = self.catalog.merge_all(tables, prune=set())
            summary[ds.dataset] = {"activity": ds.activity, "gold": counts,
                                   "models": [r.model_version for r in runs]}  # fmt: skip
            log.info("%s (%s): gold %s", ds.dataset, ds.activity, counts)
        return summary
