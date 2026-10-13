"""PretrainedPost: depois do treino dos pré-treinados, compara, decide (sim, não, não dá),
grava a gold e gera a parte "pré-treinado" de reports/sections/a1.md.

Roda sem treinar de novo: lê `outputs/<run>/a1_pretrained/{metrics.json,predictions.csv}` e o
metrics.json do ViT do zero. O texto de análise fica em `reports/a1_pretrained_narrativa.yaml`.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from visaocomp.a1.abstain import ABSTAIN, NO, YES, AbstainPolicy
from visaocomp.a1.gold import EVIDENCE, HOUSE_ATTRIBUTE, GoldWriter
from visaocomp.a1.scratch_report import NAMES, PENDING, _f, _pct, fill

log = logging.getLogger(__name__)

START = "<!-- a1_pretrained:inicio -->"
END = "<!-- a1_pretrained:fim -->"
TEMPLATE = Path(__file__).parent / "templates" / "a1_pretrained_section.md"
TEXTS = ("r13_texto", "r16_texto", "abstencao_texto", "r10_texto", "r14_texto", "r15_texto")
SCRATCH = "vit_scratch"
LABELS = {SCRATCH: "ViT do zero"}
EVAL_SETS = {
    "test": "teste inteiro (rótulo do catálogo)",
    "test_reviewed": "amostra conferida (marcação a olho)",
}


def _key(text: str) -> str:
    """Texto livre como nome de marcador: sem acento, espaço vira _ ("não dá" -> "nao_da")."""
    ascii_ = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", ascii_).strip("_").lower()


def _check(v: Any) -> float:
    return {"sim": 1.0, "não": 0.0}.get(v, np.nan) if isinstance(v, str) else np.nan


class PretrainedPost:
    """Comparação, abstenção, gold e o bloco do a1.md para um run dos pré-treinados."""

    def __init__(
        self,
        run_path: Path,
        scratch_run: Path,
        catalog_path: Path,
        narrative_path: Path | None = None,
        abstain_cfg: dict[str, float] | None = None,
    ) -> None:
        self.run_path = Path(run_path)
        self.out = self.run_path / "a1_pretrained"
        self.metrics = json.loads((self.out / "metrics.json").read_text(encoding="utf-8"))
        self.scratch = json.loads(
            (Path(scratch_run) / "a1_scratch" / "metrics.json").read_text(encoding="utf-8")
        )
        self.catalog_path = Path(catalog_path)
        self.narrative: dict[str, str] = {}
        if narrative_path and Path(narrative_path).exists():
            self.narrative = yaml.safe_load(Path(narrative_path).read_text(encoding="utf-8")) or {}
        self.abstain_cfg = abstain_cfg or {}
        self.attrs = tuple(self.metrics["attributes"])

    # -- comparação -------------------------------------------------------------------------------
    def _models(self) -> dict[str, dict[str, Any]]:
        s = self.scratch
        scratch = {
            "label": LABELS[SCRATCH],
            "params": s["model"]["params"],
            "train": s["train"],
            "test": s["test"],
            "test_reviewed": s["test_reviewed"],
        }
        return {SCRATCH: scratch, **self.metrics["models"]}

    def comparison(self) -> pd.DataFrame:
        """Uma linha por modelo x atributo x conjunto, com as mesmas métricas e o mesmo teste."""
        rows = []
        for key, m in self._models().items():
            for a in self.attrs:
                for set_key, set_name in EVAL_SETS.items():
                    r = m[set_key]["per_attribute"][a]
                    rows.append(
                        {
                            "model": key,
                            "label": m["label"],
                            "attribute": a,
                            "eval_set": set_name,
                            "n": r["n"],
                            "n_pos": r["n_pos"],
                            "precision": r["precision"],
                            "recall": r["recall"],
                            "f1": r["f1"],
                            "pr_auc": r["pr_auc"],
                            "prevalence": r["prevalence"],
                            "params": m["params"],
                            "train_min": m["train"]["seconds"] / 60,
                            "best_val_pr_auc_mean": m["train"]["best_val_pr_auc_mean"],
                        }
                    )
        return pd.DataFrame(rows)

    def chosen(self) -> str:
        """O modelo da gold: o de maior PR-AUC médio na validação entre os que têm predição."""
        models = self.metrics["models"]
        return max(models, key=lambda k: models[k]["train"]["best_val_pr_auc_mean"])

    # -- abstenção e gold -------------------------------------------------------------------------
    def decisions(self) -> tuple[AbstainPolicy, pd.DataFrame, pd.DataFrame]:
        """Política ajustada na validação; decisões do teste com contexto e rótulos."""
        key = self.chosen()
        preds = pd.read_csv(self.out / "predictions.csv")
        preds = preds[preds["model"] == key]
        writer = GoldWriter(self.catalog_path)
        ctx = writer.context(list(preds["chip_id"].unique()))
        wide = ctx.pivot_table(
            index="chip_id", columns="attribute", values="label", aggfunc="first", dropna=False
        )
        val = preds[preds["split"] == "val"].set_index("chip_id")
        thresholds = self.metrics["models"][key]["thresholds"]
        labels = {a: wide.reindex(val.index)[a].astype(float).to_numpy() for a in self.attrs}
        probs = {a: val[f"prob_{a}"].to_numpy() for a in self.attrs}
        policy = AbstainPolicy.fit(labels, probs, thresholds, **self.abstain_cfg)
        test = preds[preds["split"] == "test"]
        chip_ctx = ctx.drop_duplicates("chip_id")[["chip_id", "roof_tree_frac", "yard_tree_frac"]]
        dec = policy.decide_frame(test.merge(chip_ctx, on="chip_id"), self.attrs)
        return policy, dec, ctx

    @staticmethod
    def _stats(dec: pd.DataFrame, truth: pd.Series) -> dict[str, Any]:
        """Cobertura, contagens e acerto do "sim" e do "não" contra uma verdade (1, 0, NaN)."""
        out: dict[str, Any] = {"n": int(len(dec))}
        for d in (YES, NO, ABSTAIN):
            out[d] = int((dec["decision"] == d).sum())
        out["coverage"] = (out[YES] + out[NO]) / max(out["n"], 1)
        out["reasons"] = {
            str(k): int(v) for k, v in dec["reason"].dropna().value_counts().sort_index().items()
        }
        t = truth.to_numpy(dtype=float)
        yes, no = (dec["decision"] == YES).to_numpy(), (dec["decision"] == NO).to_numpy()
        ok_y, ok_n = yes & ~np.isnan(t), no & ~np.isnan(t)
        out["yes_precision"] = float(t[ok_y].mean()) if ok_y.any() else None
        out["no_npv"] = float(1 - t[ok_n].mean()) if ok_n.any() else None
        out["yes_n_truth"], out["no_n_truth"] = int(ok_y.sum()), int(ok_n.sum())
        return out

    def run(self, write_gold: bool = True) -> dict[str, Any]:
        """Grava comparison.csv, decisions.csv e abstain.json; e a gold, se pedido."""
        comp = self.comparison()
        comp.to_csv(self.out / "comparison.csv", index=False)
        policy, dec, ctx = self.decisions()
        dec.to_csv(self.out / "decisions.csv", index=False)
        lab = ctx.set_index(["chip_id", "attribute"])
        keyed = dec.set_index(["chip_id", "attribute"])
        catalog_truth = lab["label"].reindex(keyed.index).astype(float).reset_index(drop=True)
        checks = self._checks()
        review_truth = checks.reindex(keyed.index).reset_index(drop=True)
        stats: dict[str, Any] = {"model": self.chosen(), "policy": policy.to_dict(), "test": {}}
        for a in self.attrs:
            m = (dec["attribute"] == a).to_numpy()
            part = dec[m].reset_index(drop=True)
            stats["test"][a] = {
                "catalog": self._stats(part, catalog_truth[m].reset_index(drop=True)),
                "reviewed": self._stats(part, review_truth[m].reset_index(drop=True)),
            }
        model = self.metrics["models"][self.chosen()]
        if write_gold:
            writer = GoldWriter(self.catalog_path)
            tables = writer.tables(dec, ctx, model["model_version"], self.metrics["run_id"])
            stats["gold"] = writer.write(tables)
            stats["gold_tables"] = {
                HOUSE_ATTRIBUTE: len(tables[HOUSE_ATTRIBUTE]),
                EVIDENCE: len(tables[EVIDENCE]),
            }
            stats["agreement"] = {
                a: {
                    str(k): int(v)
                    for k, v in tables[EVIDENCE][tables[EVIDENCE]["attribute"] == a]["agreement"]
                    .value_counts()
                    .sort_index()
                    .items()
                }
                for a in self.attrs
            }
        (self.out / "abstain.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return stats

    def _checks(self) -> pd.Series:
        import duckdb

        con = duckdb.connect(str(self.catalog_path), read_only=True)
        try:
            df = con.execute("SELECT chip_id, attribute, first_check FROM silver.label").df()
        finally:
            con.close()
        df["truth"] = df["first_check"].map(_check)
        return df.set_index(["chip_id", "attribute"])["truth"]

    # -- relatório --------------------------------------------------------------------------------
    def values(self, stats: dict[str, Any]) -> dict[str, str]:
        """Marcadores que o texto de análise pode usar."""
        comp = self.comparison()
        v: dict[str, str] = {
            "run_id": self.metrics["run_id"],
            "chosen": self.chosen(),
            "chosen_label": self.metrics["models"][self.chosen()]["label"],
        }
        for r in comp.itertuples(index=False):
            tag = "test" if r.eval_set == EVAL_SETS["test"] else "rev"
            for k in ("precision", "recall", "f1", "pr_auc", "prevalence"):
                v[f"{r.model}_{tag}_{r.attribute}_{k}"] = _f(getattr(r, k), 2)
            v[f"{r.model}_params_m"] = f"{r.params / 1e6:.1f}"
            v[f"{r.model}_train_min"] = f"{r.train_min:.0f}"
            v[f"{r.model}_val_pr_auc"] = _f(r.best_val_pr_auc_mean, 3)
        for key, m in self.metrics["models"].items():
            v[f"{key}_epochs"] = str(m["train"]["epochs"])
            v[f"{key}_lr"] = str(m["train"]["lr"])
            v[f"{key}_head_in"] = str(m["head"]["in"])
            v[f"{key}_version"] = m["model_version"]
        for a, s in stats.get("test", {}).items():
            band = stats["policy"]["bands"][a]
            v[f"band_{a}_low"] = _f(band["low"], 2)
            v[f"band_{a}_high"] = _f(band["high"], 2)
            for truth in ("catalog", "reviewed"):
                t = s[truth]
                v[f"abs_{a}_{truth}_yes_precision"] = _f(t["yes_precision"], 2)
                v[f"abs_{a}_{truth}_no_npv"] = _f(t["no_npv"], 2)
            c = s["catalog"]
            v[f"abs_{a}_coverage"] = _pct(c["coverage"])
            for d, name in ((YES, "yes"), (NO, "no"), (ABSTAIN, "abstain")):
                v[f"abs_{a}_{name}"] = str(c[d])
            for reason, n in c["reasons"].items():
                v[f"abs_{a}_reason_{_key(reason)}"] = str(n)
        for a, ag in stats.get("agreement", {}).items():
            for k, n in ag.items():
                v[f"agree_{a}_{_key(k)}"] = str(n)
        if "gold_tables" in stats:
            v["gold_rows"] = str(stats["gold_tables"][HOUSE_ATTRIBUTE])
        return v

    def _text(self, key: str, values: dict[str, str]) -> str:
        raw = self.narrative.get(key)
        return fill(raw.strip(), values) if raw else PENDING

    def _comparison_table(self) -> str:
        comp = self.comparison()
        rows = [
            "| Atributo | Conjunto (verdade) | Modelo | Precision | Recall | F1 | PR-AUC | Acaso "
            "| Parâmetros | Treino (min) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in comp.itertuples(index=False):
            rows.append(
                f"| {NAMES[r.attribute]} | {r.eval_set} | {r.label} | {_f(r.precision, 2)} "
                f"| {_f(r.recall, 2)} | {_f(r.f1, 2)} | {_f(r.pr_auc, 2)} | {_f(r.prevalence, 2)} "
                f"| {r.params / 1e6:.1f} M | {r.train_min:.0f} |"
            )
        return "\n".join(rows)

    def _abstain_table(self, stats: dict[str, Any]) -> str:
        rows = [
            "| Atributo | Faixa de incerteza (p) | sim | não | não dá | Motivos | Cobertura "
            "| Acerto do sim (catálogo / conferida) | Acerto do não (catálogo / conferida) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for a, s in stats["test"].items():
            band = stats["policy"]["bands"][a]
            c, r = s["catalog"], s["reviewed"]
            reasons = ", ".join(f"{k} {n}" for k, n in c["reasons"].items()) or "nenhum"
            rows.append(
                f"| {NAMES[a]} | {_f(band['low'], 2)} a {_f(band['high'], 2)} | {c[YES]} | {c[NO]} "
                f"| {c[ABSTAIN]} | {reasons} | {_pct(c['coverage'])} "
                f"| {_f(c['yes_precision'], 2)} / {_f(r['yes_precision'], 2)} "
                f"| {_f(c['no_npv'], 2)} / {_f(r['no_npv'], 2)} |"
            )
        return "\n".join(rows)

    def block(self, stats: dict[str, Any]) -> str:
        v = self.values(stats)
        filled = {
            **v,
            "start": START,
            "end": END,
            "comparison_table": self._comparison_table(),
            "abstain_table": self._abstain_table(stats),
            **{k: self._text(k, v) for k in TEXTS},
        }
        return fill(TEMPLATE.read_text(encoding="utf-8"), filled)

    def write(self, path: Path, stats: dict[str, Any]) -> Path:
        """Troca o bloco entre os marcadores no a1.md (ou acrescenta no fim)."""
        path = Path(path)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        block = self.block(stats)
        if START in text and END in text:
            pre, rest = text.split(START, 1)
            text = pre + block.rstrip("\n") + rest.split(END, 1)[1]
        else:
            text = text.rstrip("\n") + "\n\n" + block
        path.write_text(text, encoding="utf-8")
        return path
