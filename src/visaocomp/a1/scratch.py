"""ScratchPipeline: do catálogo ao metrics.json e aos mapas de atenção do ViT do zero."""

from __future__ import annotations

import copy
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from visaocomp.a1.chips import (
    ChipCatalogReader,
    RegionMasker,
    check_matrix,
    label_matrix,
    load_images,
)
from visaocomp.a1.scratch_config import ScratchConfig
from visaocomp.a1.train import (
    AttentionMapper,
    ChipClassifier,
    MultiLabelTrainer,
    Prediction,
    best_f1_threshold,
    channel_stats,
    multilabel_metrics,
    positive_weights,
)
from visaocomp.device import seed_everything
from visaocomp.run import RunDir
from visaocomp.vit import VisionTransformer, shuffle_patches

log = logging.getLogger(__name__)

STAGE = "a1_scratch"


def permutation_check(model: ChipClassifier, images: torch.Tensor, seed: int) -> dict[str, float]:
    """Embaralha os patches: com posição a saída muda; sem posição (zerada) não muda."""
    gen = torch.Generator().manual_seed(seed)
    vit = model.vit
    perm = torch.randperm(vit.cfg.num_patches, generator=gen)
    dev = next(model.parameters()).device
    x = images.to(dev).float() / 255.0
    xs = shuffle_patches(x, vit.cfg.patch, perm.to(dev))
    no_pos = copy.deepcopy(model)
    with torch.no_grad():
        no_pos.vit.pos_embed.zero_()
    model.eval()
    no_pos.eval()
    with torch.no_grad():
        with_pe = (torch.sigmoid(model(x)) - torch.sigmoid(model(xs))).abs()
        without = (no_pos(x) - no_pos(xs)).abs()
    return {
        "n_images": int(len(images)),
        "with_pe_mean_abs_prob_change": float(with_pe.mean()),
        "with_pe_max_abs_prob_change": float(with_pe.max()),
        "without_pe_max_abs_logit_change": float(without.max()),
    }


class ScratchPipeline:
    """Lê o catálogo, treina o ViT do zero, avalia no teste uma vez e grava tudo em outputs/."""

    def __init__(
        self,
        cfg: ScratchConfig,
        catalog_path: Path,
        data_root: Path,
        outputs_root: Path,
        device: torch.device,
        figures_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.catalog_path = Path(catalog_path)
        self.data_root = Path(data_root)
        self.outputs_root = Path(outputs_root)
        self.device = device
        self.figures_dir = figures_dir

    # -- partes ---------------------------------------------------------------------------------
    @staticmethod
    def _counts(chips: pd.DataFrame, attributes: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in ("train", "val", "test"):
            part = chips[chips["split"] == s]
            out[s] = {"recortes": int(len(part))}
            for a in attributes:
                vc = part[a].map(lambda v: "vazio" if pd.isna(v) else str(int(v))).value_counts()
                out[s][a] = {k: int(v) for k, v in vc.sort_index().items()}
        return out

    def _examples(
        self, test: pd.DataFrame, probs: np.ndarray, checks: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        """Piscina e solar: o acerto mais confiante na amostra conferida.

        Erro: o erro mais confiante (classe errada no limiar, contra a marcação a olho)."""
        attrs = self.cfg.attributes
        out: dict[str, dict[str, Any]] = {}
        for j, a in enumerate(attrs):
            pos = np.flatnonzero(checks[:, j] == 1)
            if len(pos):
                i = int(pos[np.argmax(probs[pos, j])])
                out[a] = {"row": i, "attribute": a, "kind": "acerto"}
        thr = np.asarray(self._thresholds)[None, :]
        wrong = (probs >= thr).astype(float) != np.nan_to_num(checks, nan=-1.0)
        err = np.abs(probs - np.nan_to_num(checks, nan=0.0))
        err[np.isnan(checks) | ~wrong] = -1.0
        i, j = np.unravel_index(int(np.argmax(err)), err.shape)
        out["erro"] = {
            "row": int(i),
            "attribute": attrs[j],
            "kind": "falso positivo" if checks[i, j] == 0 else "falso negativo",
        }
        for ex in out.values():
            r = ex["row"]
            ex.update(
                chip_id=str(test.iloc[r]["chip_id"]),
                probs={a: float(probs[r, k]) for k, a in enumerate(attrs)},
                catalog_label={
                    a: None if pd.isna(test.iloc[r][a]) else int(test.iloc[r][a]) for a in attrs
                },
                reviewed={a: test.iloc[r][f"{a}_check"] for a in attrs},
            )
        return out

    def _regions(self, test: pd.DataFrame, pools: pd.DataFrame, pred: Prediction) -> dict[str, Any]:
        """Quanto da atenção do CLS cai no telhado e na piscina, contra a área deles."""
        masks = RegionMasker(self.cfg.image_size, self.cfg.patch, self.cfg.pixel_m).masks(
            test, pools
        )
        maps = pred.cls_attention
        assert maps is not None
        thr = self._thresholds
        pool_j, solar_j = self.cfg.attributes.index("pool"), self.cfg.attributes.index("solar")
        pred_solar = pred.probs[:, solar_j] >= thr[solar_j]
        pred_pool = pred.probs[:, pool_j] >= thr[pool_j]
        lift = AttentionMapper.region_lift
        return {
            "roof_all": lift(maps, masks.roof),
            "roof_pred_solar": lift(maps[pred_solar], masks.roof[pred_solar]),
            "pool_all": lift(maps, masks.pool),
            "pool_pred_pool": lift(maps[pred_pool], masks.pool[pred_pool]),
        }

    def _figures(
        self,
        out_dir: Path,
        test_images: torch.Tensor,
        pred: Prediction,
        examples: dict[str, dict[str, Any]],
        heads: dict[str, int],
    ) -> list[str]:
        mapper = AttentionMapper(self.cfg.image_size)
        names = {"pool": "piscina", "solar": "solar"}
        files = []
        for key, ex in examples.items():
            r, a = ex["row"], ex["attribute"]
            probs = ", ".join(f"{names[k]} {v:.2f}" for k, v in ex["probs"].items())
            title = (
                f"{key}: {ex['kind']} de {names[a]} (chip {ex['chip_id']}); probabilidades: {probs}"
            )
            path = out_dir / f"attention_{key}.png"
            mapper.figure(test_images[r].numpy(), pred.cls_attention[r], title, path, heads[a])
            files.append(path.name)
        return files

    def _curves(self, out_dir: Path, history: dict[str, Any]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        ep = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(ep, history["train_loss"], label="treino")
        axes[0].plot(ep, history["val_loss"], label="validação")
        axes[0].set(xlabel="época", ylabel="BCE com peso", title="perda")
        for a, vals in history["val_pr_auc"].items():
            axes[1].plot(ep, [np.nan if v is None else v for v in vals], label=a)
        axes[1].axvline(history["best_epoch"], color="gray", ls="--", lw=1, label="melhor época")
        axes[1].set(xlabel="época", ylabel="PR-AUC", title="PR-AUC na validação")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=110)
        plt.close(fig)

    # -- execução -------------------------------------------------------------------------------
    def run(self) -> Path:
        """Executa o ViT do zero inteiro e devolve o caminho do manifest.json."""
        t_start = time.perf_counter()
        cfg = self.cfg
        attrs = cfg.attributes
        seed_everything(cfg.seed)
        log.info("A1 do zero: início, catálogo %s", self.catalog_path)
        chips, pools = ChipCatalogReader(self.catalog_path, attrs).read()
        images = load_images(list(chips["path"]), self.data_root, cfg.image_size)
        labels = label_matrix(chips, attrs)
        checks = check_matrix(chips, attrs)
        idx = {s: np.flatnonzero(chips["split"].to_numpy() == s) for s in ("train", "val", "test")}
        log.info("recortes lidos: %d imagens %s", len(images), tuple(images.shape[1:]))

        run = RunDir(self.outputs_root, STAGE)
        out_dir = run.path / STAGE
        out_dir.mkdir()
        mean, std = channel_stats(images[idx["train"]])
        model = ChipClassifier(VisionTransformer(cfg.vit()), mean, std)
        pos_weight = positive_weights(labels[idx["train"]]) if cfg.pos_weight else None
        trainer = MultiLabelTrainer(model, cfg, self.device, pos_weight, out_dir / "best.pt")
        t_train = time.perf_counter()
        history = trainer.fit(
            images[idx["train"]], labels[idx["train"]], images[idx["val"]], labels[idx["val"]]
        )
        train_seconds = time.perf_counter() - t_train

        val = trainer.predict(images[idx["val"]], labels[idx["val"]])
        self._thresholds = [
            best_f1_threshold(
                labels[idx["val"], j][~np.isnan(labels[idx["val"], j])],
                val.probs[~np.isnan(labels[idx["val"], j]), j],
            )
            for j in range(len(attrs))
        ]
        test_images = images[idx["test"]]
        pred = trainer.predict(test_images, labels[idx["test"]], attention_layer=-1)
        test = chips.iloc[idx["test"]].reset_index(drop=True)
        t_lab, t_chk = labels[idx["test"]], checks[idx["test"]]
        reviewed = test[[f"{a}_check" for a in attrs]].notna().any(axis=1).to_numpy()
        regions = self._regions(test, pools, pred)
        heads = {
            "pool": int(np.argmax(regions["pool_all"]["per_head"]))
            if regions["pool_all"]["n"]
            else 0,
            "solar": int(np.argmax(regions["roof_all"]["per_head"]))
            if regions["roof_all"]["n"]
            else 0,
        }
        examples = self._examples(test, pred.probs, t_chk)
        figures = self._figures(out_dir, test_images, pred, examples, heads)
        perm = permutation_check(trainer.model, test_images[:64], cfg.seed)

        metrics: dict[str, Any] = {
            "run_id": run.run_id,
            "attributes": list(attrs),
            "model": {
                **cfg.vit().to_dict(),
                "params": model.vit.param_count(),
                "tokens": cfg.vit().num_patches + 1,
                "normalization": {"mean": mean, "std": std},
            },
            "train": {
                **{k: v for k, v in cfg.to_dict().items() if k not in ("examples",)},
                "device": str(self.device),
                "pos_weight": None if pos_weight is None else [float(v) for v in pos_weight],
                "seconds": train_seconds,
                "best_epoch": history.best_epoch,
                "best_val_pr_auc_mean": history.best_val_pr_auc_mean,
            },
            "thresholds": dict(zip(attrs, self._thresholds, strict=True)),
            "test": {
                "label_source": "silver.label (rótulo automático da montagem; vazio fica fora)",
                "n_chips": int(len(test)),
                "test_loss": pred.loss,
                "per_attribute": multilabel_metrics(t_lab, pred.probs, attrs, self._thresholds),
            },
            "test_reviewed": {
                "label_source": "silver.label.first_check (conferência a olho do conferencia_inicial.csv: "  # noqa: E501
                "sim = 1, não = 0; 'não dá' fica fora)",
                "n_chips": int(reviewed.sum()),
                "per_attribute": multilabel_metrics(
                    t_chk[reviewed], pred.probs[reviewed], attrs, self._thresholds
                ),
                "catalog_labels_same_chips": multilabel_metrics(
                    t_lab[reviewed], pred.probs[reviewed], attrs, self._thresholds
                ),
            },
            "regions": regions,
            "attention": {
                "layer": "última (6 de 6)" if cfg.depth == 6 else f"última ({cfg.depth})",
                "what": "linha do token CLS: peso de atenção do CLS para cada um dos patches",
                "head_used": {a: h + 1 for a, h in heads.items()},
                "examples": examples,
                "figures": figures,
            },
            "permutation": perm,
            "split": self._counts(chips, attrs),
            "history": history.to_dict(),
        }
        run.write_json(f"{STAGE}/metrics.json", metrics)
        self._curves(out_dir, metrics["history"])
        if self.figures_dir is not None:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            for name in [*figures, "curves.png"]:
                shutil.copy2(out_dir / name, self.figures_dir / name)
        manifest = run.write_manifest(
            [{"name": "catalog.duckdb", "path": str(self.catalog_path),
              "tables": ["silver.chip", "silver.label", "silver.house", "silver.osm_pool"]}],
            {"recortes": int(len(chips)), **{s: int(len(i)) for s, i in idx.items()},
             "conferidos_no_teste": int(reviewed.sum())},
            extra={
                "config": cfg.to_dict(),
                "device": str(self.device),
                "timings_seconds": {
                    "train": round(train_seconds, 1),
                    "total": round(time.perf_counter() - t_start, 1),
                },
            },
        )  # fmt: skip
        log.info(
            "A1 do zero: fim, PR-AUC no teste %s; resultados em %s",
            ", ".join(f"{a} {m['pr_auc']}" for a, m in metrics["test"]["per_attribute"].items()),
            out_dir,
        )
        return manifest
