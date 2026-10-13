"""ScratchReport: gera a parte "ViT do zero" de reports/sections/a1.md a partir do run.

Todo número vem do metrics.json do run. O texto de análise (escrito depois de olhar os mapas)
fica em `reports/a1_scratch_narrativa.yaml` e cita os números por `{{marcador}}`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

START = "<!-- a1_scratch:inicio -->"
END = "<!-- a1_scratch:fim -->"
HEADER = "# A1: Vision Transformer em casas de Austin\n"
PENDING = "(texto de análise pendente)"
NAMES = {"pool": "piscina", "solar": "solar"}
_MARK = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
TEMPLATE = Path(__file__).parent / "templates" / "a1_scratch_section.md"
TEXTS = ("r09_texto", "resultados_texto", "regioes_texto", "limites_texto")


def latest_run(outputs_root: Path) -> Path | None:
    """Pasta do run mais recente do ViT do zero que tem metrics.json."""
    runs = sorted(Path(outputs_root).glob("*/a1_scratch/metrics.json"))
    return runs[-1].parent.parent if runs else None


def fill(text: str, values: dict[str, str]) -> str:
    """Troca `{{nome}}` pelo valor; marcador desconhecido é erro (evita número digitado)."""

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"marcador desconhecido no texto de análise: {{{{{key}}}}}")
        return values[key]

    return _MARK.sub(sub, text)


def _f(x: float | None, digits: int = 3) -> str:
    return "n/d" if x is None else f"{x:.{digits}f}"


def _pct(x: float | None) -> str:
    return "n/d" if x is None else f"{100 * x:.0f}%"


class ScratchReport:
    """Monta o bloco do ViT do zero (R06 a R09, R11, R12) e grava entre os marcadores do a1.md."""

    def __init__(self, run_path: Path, narrative_path: Path | None = None) -> None:
        self.run_path = Path(run_path)
        self.metrics: dict[str, Any] = json.loads(
            (self.run_path / "a1_scratch" / "metrics.json").read_text(encoding="utf-8")
        )
        self.narrative: dict[str, str] = {}
        if narrative_path and Path(narrative_path).exists():
            self.narrative = yaml.safe_load(Path(narrative_path).read_text(encoding="utf-8")) or {}

    def values(self) -> dict[str, str]:
        """Todos os marcadores que o texto de análise pode usar."""
        m = self.metrics
        v: dict[str, str] = {
            "run_id": m["run_id"],
            "epochs": str(m["train"]["epochs"]),
            "best_epoch": str(m["train"]["best_epoch"]),
            "train_min": f"{m['train']['seconds'] / 60:.0f}",
            "params": f"{m['model']['params']:,}".replace(",", "."),
            "tokens": str(m["model"]["tokens"]),
            "n_test": str(m["test"]["n_chips"]),
            "n_reviewed": str(m["test_reviewed"]["n_chips"]),
            "perm_with_mean": _f(m["permutation"]["with_pe_mean_abs_prob_change"]),
            "perm_with_max": _f(m["permutation"]["with_pe_max_abs_prob_change"]),
            "perm_without_max": f"{m['permutation']['without_pe_max_abs_logit_change']:.1e}",
            "perm_n": str(m["permutation"]["n_images"]),
        }
        sets = {
            "test": m["test"]["per_attribute"],
            "rev": m["test_reviewed"]["per_attribute"],
            "revcat": m["test_reviewed"]["catalog_labels_same_chips"],
        }
        for name, per in sets.items():
            for a, row in per.items():
                for k in ("precision", "recall", "f1", "pr_auc", "prevalence"):
                    v[f"{name}_{a}_{k}"] = _f(row[k], 2)
                v[f"{name}_{a}_n"] = str(row["n"])
                v[f"{name}_{a}_n_pos"] = str(row["n_pos"])
        for region, row in m["regions"].items():
            v[f"{region}_n"] = str(row["n"])
            v[f"{region}_lift"] = _f(row["mean_heads"], 1)
            v[f"{region}_share"] = _pct(row["attention_share"])
            v[f"{region}_area"] = _pct(row["area_share"])
            if row["per_head"]:
                v[f"{region}_best_head_lift"] = _f(max(row["per_head"]), 1)
        for a, h in m["attention"]["head_used"].items():
            v[f"head_{a}"] = str(h)
        for key, ex in m["attention"]["examples"].items():
            v[f"ex_{key}_chip"] = ex["chip_id"]
            v[f"ex_{key}_kind"] = ex["kind"]
            v[f"ex_{key}_attr"] = NAMES[ex["attribute"]]
            for a, p in ex["probs"].items():
                v[f"ex_{key}_prob_{a}"] = _f(p, 2)
        return v

    def _text(self, key: str, values: dict[str, str]) -> str:
        raw = self.narrative.get(key)
        return fill(raw.strip(), values) if raw else PENDING

    def _metrics_table(self) -> str:
        m = self.metrics
        rows = [
            "| Atributo | Conjunto (verdade) | n | Positivos | Precision | Recall | F1 | PR-AUC "
            "| PR-AUC do acaso |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        sets = [
            ("teste inteiro (rótulo do catálogo)", m["test"]["per_attribute"]),
            ("amostra conferida (marcação a olho)", m["test_reviewed"]["per_attribute"]),
            (
                "amostra conferida (rótulo do catálogo)",
                m["test_reviewed"]["catalog_labels_same_chips"],
            ),
        ]
        for a in m["attributes"]:
            for name, per in sets:
                r = per[a]
                rows.append(
                    f"| {NAMES[a]} | {name} | {r['n']} | {r['n_pos']} | {_f(r['precision'], 2)} "
                    f"| {_f(r['recall'], 2)} | {_f(r['f1'], 2)} | {_f(r['pr_auc'], 2)} "
                    f"| {_f(r['prevalence'], 2)} |"
                )
        return "\n".join(rows)

    def _regions_table(self) -> str:
        reg = self.metrics["regions"]
        labels = {
            "roof_all": "telhado da casa, todo o teste",
            "roof_pred_solar": "telhado, recortes que o modelo diz solar",
            "pool_all": "piscina do OSM no recorte, todo o teste",
            "pool_pred_pool": "piscina do OSM, recortes que o modelo diz piscina",
        }
        rows = [
            "| Região | Recortes | Atenção do CLS na região | Área da região "
            "| Razão (média das cabeças) | Razão por cabeça (1 a 6) |",
            "|---|---|---|---|---|---|",
        ]
        for key, name in labels.items():
            r = reg[key]
            heads = " / ".join(_f(h, 1) for h in r["per_head"]) or "n/d"
            rows.append(
                f"| {name} | {r['n']} | {_pct(r['attention_share'])} | {_pct(r['area_share'])} "
                f"| {_f(r['mean_heads'], 1)} | {heads} |"
            )
        return "\n".join(rows)

    def block(self) -> str:
        """O markdown do bloco do ViT do zero, com os marcadores (template + números do run)."""
        m, v = self.metrics, self.values()
        t, mo = m["train"], m["model"]
        weights = zip(m["attributes"], t["pos_weight"] or [], strict=False)
        filled = {
            **v,
            "start": START,
            "end": END,
            **{
                k: str(mo[k])
                for k in ("heads", "patch", "dim", "depth", "num_outputs", "image_size")
            },
            "grid": str(mo["image_size"] // mo["patch"]),
            "n_train": str(m["split"]["train"]["recortes"]),
            "n_val": str(m["split"]["val"]["recortes"]),
            "n_test_split": str(m["split"]["test"]["recortes"]),
            "pos_weight": ", ".join(f"{NAMES[a]} {w:.1f}" for a, w in weights),
            **{
                k: str(t[k])
                for k in ("lr", "weight_decay", "warmup_epochs", "batch_size", "device")
            },
            "best_val_pr_auc_mean": _f(t["best_val_pr_auc_mean"]),
            "thresholds": ", ".join(f"{NAMES[a]} {_f(x, 2)}" for a, x in m["thresholds"].items()),
            "metrics_table": self._metrics_table(),
            "regions_table": self._regions_table(),
            "figures": "\n\n".join(f"![{f}](../figures/a1/{f})" for f in m["attention"]["figures"]),
            **{k: self._text(k, v) for k in TEXTS},
        }
        return fill(TEMPLATE.read_text(encoding="utf-8"), filled)

    def write(self, path: Path) -> Path:
        """Grava o bloco no a1.md: troca o que está entre os marcadores, ou cria o arquivo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        block = self.block()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if START in text and END in text:
                pre, rest = text.split(START, 1)
                post = rest.split(END, 1)[1]
                text = pre + block.rstrip("\n") + post
            else:
                text = text.rstrip("\n") + "\n\n" + block
        else:
            text = HEADER + "\n" + block
        path.write_text(text, encoding="utf-8")
        return path
