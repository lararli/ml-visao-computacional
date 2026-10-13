"""SectionWriter: gera reports/sections/a3.md a partir do metrics.json (nenhum número digitado)."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights

TEMPLATE = Path(__file__).parent / "templates" / "a3_section.md"


def pct(value: float | None, digits: int = 1) -> str:
    """Formata 0,934 como '93,4%' (vírgula decimal); None vira 'sem imagens'."""
    return "sem imagens" if value is None else f"{value * 100:.{digits}f}%".replace(".", ",")


def millions(n: int) -> str:
    """Formata 5288548 como '5,3 milhões'."""
    return f"{n / 1e6:.1f} milhões".replace(".", ",")


def thousands(n: int) -> str:
    """Formata 1764 como '1.764' (ponto de milhar)."""
    return f"{n:,}".replace(",", ".")


class SectionWriter:
    """Preenche o modelo de texto da A3 com os números e as classes de um metrics.json."""

    def __init__(self, metrics: dict[str, Any], template: Path = TEMPLATE) -> None:
        self.m = metrics
        self.template = Path(template)

    def _per_class_table(self) -> str:
        rows = [
            "| Classe | Imagens no teste | Acertos | Accuracy | Precision | F1 |",
            "|---|---|---|---|---|---|",
        ]
        for name, v in self.m["per_class"].items():
            rows.append(
                f"| {name} | {v['n']} | {v['correct']} | {pct(v['accuracy'])} "
                f"| {pct(v['precision'])} | {pct(v['f1'])} |"
            )
        return "\n".join(rows)

    def _hard_classes(self, k: int = 2) -> str:
        ranked = sorted(
            ((v["accuracy"], n) for n, v in self.m["per_class"].items() if v["n"]),
        )[:k]
        return " e ".join(f"{n} ({pct(a)})" for a, n in ranked)

    def _confusions(self) -> str:
        top = self.m["top_confusions"][:3]
        if not top:
            return "o teste não teve nenhum erro"
        return "; ".join(
            f"{c['count']} {'imagem' if c['count'] == 1 else 'imagens'} de {c['true']} "
            f"{'prevista' if c['count'] == 1 else 'previstas'} como {c['pred']}"
            for c in top
        )

    def values(self) -> dict[str, str]:
        """Todos os valores que entram no texto, já formatados."""
        m = self.m
        eff = EfficientNet_B0_Weights.IMAGENET1K_V1.meta
        res = ResNet50_Weights.IMAGENET1K_V1.meta
        data, split, train, model = m["data"], m["split"], m["train"], m["model"]
        return {
            "run_id": m["run_id"],
            "classes_csv": ", ".join(m["classes"]),
            "n_classes": str(m["n_classes"]),
            "files_found": thousands(data["files_found"]),
            "files_kept": thousands(data["files_kept"]),
            "files_duplicate": str(data["files_removed_duplicate"]),
            "files_unreadable": str(data["files_skipped_unreadable"]),
            "n_train": thousands(split["train"]),
            "n_val": thousands(split["val"]),
            "n_test": thousands(split["test"]),
            "seed": str(split["seed"]),
            "epochs": str(train["epochs"]),
            "batch_size": str(train["batch_size"]),
            "lr": f"{train['lr']:g}".replace(".", ","),
            "device": train["device"],
            "train_seconds": f"{train['seconds']:.0f}",
            "best_epoch": str(train["best_epoch"]),
            "best_val_acc": pct(train["best_val_acc"]),
            "acc_global": pct(m["accuracy_global"]),
            "balanced_acc": pct(m["balanced_accuracy"]),
            "macro_f1": pct(m["macro_f1"]),
            "baseline_acc": pct(m["baseline"]["accuracy"]),
            "baseline_class": m["baseline"]["class"],
            "per_class_table": self._per_class_table(),
            "hard_classes": self._hard_classes(),
            "confusions": self._confusions(),
            "params_total": millions(model["total"]),
            "params_trainable": thousands(model["trainable"]),
            "head_in": thousands(model["head_in_features"]),
            "head_out": str(model["head_out_features"]),
            "eff_params": millions(eff["num_params"]),
            "eff_top1": f"{eff['_metrics']['ImageNet-1K']['acc@1']:.1f}".replace(".", ","),
            "eff_gflops": f"{eff['_ops']:.2f}".replace(".", ","),
            "res_params": millions(res["num_params"]),
            "res_top1": f"{res['_metrics']['ImageNet-1K']['acc@1']:.1f}".replace(".", ","),
            "res_gflops": f"{res['_ops']:.1f}".replace(".", ","),
        }

    def render(self) -> str:
        """O texto final; falha se algum espaço do modelo ficou sem valor."""
        return Template(self.template.read_text(encoding="utf-8")).substitute(self.values())

    def write(self, path: Path) -> Path:
        """Grava o texto em `path` (cria a pasta)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path
