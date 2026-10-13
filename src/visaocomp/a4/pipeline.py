"""A4 de ponta a ponta: caso, GAN sem e com mitigação, FID, sintético e recall com e sem."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from visaocomp.a4.config import A4Config, GanRunConfig  # noqa: E402
from visaocomp.a4.data import CaseBuilder, CaseSplit, Deduplicator, XrayDataset, list_records  # noqa: E402
from visaocomp.a4.fid import FidMeter  # noqa: E402
from visaocomp.a4.gan import ConditionalGan, GanTrainer  # noqa: E402
from visaocomp.a4.recall import ClassifierTrainer, Condition, RecallExperiment  # noqa: E402
from visaocomp.run import RunDir  # noqa: E402

log = logging.getLogger(__name__)

STAGE = "a4"


def pick_device(prefer: str | None = None) -> torch.device:
    """Dispositivo automático: cuda, depois mps, depois cpu (ou o pedido)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def open_run(outputs_root: Path, run_id: str) -> RunDir:
    """Reabre uma pasta de execução existente (para --resume); não cria pasta nova."""
    path = Path(outputs_root) / run_id
    if not path.is_dir():
        raise FileNotFoundError(f"{path} não existe")
    run = RunDir.__new__(RunDir)
    run.stage, run.outputs_root, run.run_id, run.path = STAGE, Path(outputs_root), run_id, path
    run.created_at = datetime.now(UTC).isoformat(timespec="seconds")
    return run


def instability_flags(history: dict[str, Any], diversity: float) -> dict[str, Any]:
    """Sinais de instabilidade lidos das curvas: divergência, oscilação e colapso de modo."""
    n = len(history["loss_d"])
    tail = max(1, n // 5)
    d_fake_tail = float(np.mean(history["d_fake"][-tail:]))
    d_real_tail = float(np.mean(history["d_real"][-tail:]))
    lg = np.asarray(history["loss_g"][-tail:])
    fids = [f["fid"] for f in history["fid"]]
    best, final = (min(fids), fids[-1]) if fids else (float("nan"), float("nan"))
    return {
        "d_real_final": d_real_tail,
        "d_fake_final": d_fake_tail,
        "loss_g_final": float(lg.mean()),
        "loss_g_cv_final": float(lg.std() / lg.mean()) if lg.mean() > 0 else 0.0,
        "discriminador_dominou": bool(d_fake_tail < 0.1 and d_real_tail > 0.9),
        "fid_subiu_depois_do_melhor": bool(fids and (final - best) / best > 0.10),
        "diversidade_covid": diversity,
        "colapso_de_modo": bool(diversity < 0.5),
    }


def mean_pairwise_distance(feats: np.ndarray, max_n: int = 600) -> float:
    """Distância euclidiana média entre pares (mede variedade num conjunto)."""
    f = torch.from_numpy(feats[:max_n]).float()
    d = torch.cdist(f, f)
    n = len(f)
    return float(d.sum() / (n * (n - 1))) if n > 1 else 0.0


class A4Pipeline:
    """Executa a A4 numa pasta outputs/<run_id>/; cada etapa pula o que já está pronto."""

    def __init__(
        self,
        cfg: A4Config,
        raw_dir: Path,
        run: RunDir,
        device: torch.device,
        feature_extractor: nn.Module | None = None,
        cache_dir: Path | None = None,
        source_info: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.raw_dir = Path(raw_dir)
        self.run_dir = run
        self.out = run.path
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.source_info = source_info or {}
        self._extractor = feature_extractor
        self.durations: dict[str, float] = {}
        self.dedup_report: dict[str, Any] = {}

    # ---------- etapas ----------

    def stage_case(self) -> CaseSplit:
        """Dedup por pHash, teste real separado primeiro, caso 840/240/120 do resto."""
        path = self.out / "splits.csv"
        report_path = self.out / "dedup.json"
        if path.exists() and report_path.exists():
            self.dedup_report = json.loads(report_path.read_text(encoding="utf-8"))
            return CaseSplit.load(path, self.cfg.classes)
        records = list_records(self.raw_dir, self.cfg.classes)
        cache = self.cache_dir / f"phash_{self.cfg.dedup.hash_size}.csv" if self.cache_dir else None
        dedup = Deduplicator(self.cfg.dedup.hash_size, self.cfg.dedup.max_distance, cache).run(
            records
        )
        split = CaseBuilder(self.cfg.case, self.cfg.classes).build(dedup.kept, dedup.hashes)
        self.dedup_report = {
            **dedup.report,
            "removidas_por_classe": {
                self.cfg.classes[k]: v for k, v in dedup.report["removidas_por_classe"].items()
            },
            "por_classe_antes": {
                c: sum(1 for r in records if r.label == i) for i, c in enumerate(self.cfg.classes)
            },
            "por_classe_depois": {
                c: sum(1 for r in dedup.kept if r.label == i)
                for i, c in enumerate(self.cfg.classes)
            },
        }
        report_path.write_text(json.dumps(self.dedup_report, indent=2, ensure_ascii=False), "utf-8")
        split.save(path)
        return split

    def _fid_meter(self) -> FidMeter:
        if self._extractor is None:
            from visaocomp.a4.fid import InceptionFeatures

            self._extractor = InceptionFeatures()
        return FidMeter(self._extractor, self.device, size=self.cfg.gan.image_size)

    def stage_gan(self, split: CaseSplit) -> dict[str, Any]:
        """Treina a GAN sem e com mitigação (só no treino do caso) e grava gan_metrics.json."""
        g = self.cfg.gan
        train_x, train_y = XrayDataset(split.records("train"), g.image_size).stack()
        meter = self._fid_meter()
        real_feats = meter.features(train_x)
        reference = meter.reference(real_feats)
        pos = self.cfg.positive_index
        real_pos_feats = real_feats[(train_y == pos).numpy()]
        fid_labels = train_y[
            torch.randperm(len(train_y), generator=torch.Generator().manual_seed(0))
        ]
        fid_labels = fid_labels.repeat(int(np.ceil(g.fid_n / len(fid_labels))))[: g.fid_n]

        def fid_fn(gan: ConditionalGan) -> dict[str, float]:
            fake = gan.sample(fid_labels, gen=torch.Generator().manual_seed(123))
            return meter.score(reference, fake)

        runs: dict[str, Any] = {}
        for rc in g.runs:
            run_out = self.out / "gan" / rc.name
            run_out.mkdir(parents=True, exist_ok=True)
            hist_path = run_out / "history.json"
            t0 = time.perf_counter()
            if hist_path.exists():
                history = json.loads(hist_path.read_text(encoding="utf-8"))
                log.info("gan %s: já treinada nesta execução, pulando", rc.name)
            else:
                torch.manual_seed(g.seed)
                gan = self._make_gan(rc)
                trainer = GanTrainer(gan, g, rc, self.device, run_out, fid_fn)
                history = trainer.fit(train_x, train_y).to_dict()
            self.durations[f"gan_{rc.name}"] = round(time.perf_counter() - t0, 1)
            runs[rc.name] = self._evaluate_gan(rc, history, run_out, meter, real_pos_feats)
        names = [rc.name for rc in g.runs]
        chosen = min(names, key=lambda n: runs[n]["fid_best"])
        doc: dict[str, Any] = {
            "fid_reference": {
                "real": "treino do caso",
                "n_real": len(train_y),
                "n_fake": g.fid_n,
                "resolucao": g.image_size,
                "extrator": type(self._extractor).__name__,
            },
            "runs": runs,
            "chosen_for_synthetic": chosen,
            "regra_de_escolha": "a variante com o menor FID (melhor checkpoint) gera o sintético",
        }
        if len(names) == 2:
            a, b = runs[names[0]], runs[names[1]]
            doc["comparacao"] = {
                "of": f"{names[1]} vs {names[0]}",
                "fid_final": [a["fid_final"], b["fid_final"]],
                "fid_best": [a["fid_best"], b["fid_best"]],
                "melhora_fid": bool(b["fid_best"] < a["fid_best"]),
                "melhora_fid_final": bool(b["fid_final"] < a["fid_final"]),
            }
        (self.out / "gan_metrics.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._plot_gan_comparison(doc)
        return doc

    def _make_gan(self, rc: GanRunConfig) -> ConditionalGan:
        g = self.cfg.gan
        return ConditionalGan(
            g.nz, len(self.cfg.classes), g.ngf, g.ndf, g.image_size, rc.spectral_norm
        )

    def _evaluate_gan(
        self,
        rc: GanRunConfig,
        history: dict[str, Any],
        run_out: Path,
        meter: FidMeter,
        real_pos_feats: np.ndarray,
    ) -> dict[str, Any]:
        """FID da COVID e diversidade no melhor checkpoint; sinais de instabilidade das curvas."""
        pos = self.cfg.positive_index
        gan = self._make_gan(rc).to(self.device)
        gan.generator.load_state_dict(
            torch.load(run_out / "generator_best.pt", map_location=self.device)
        )
        labels = torch.full((self.cfg.synthetic.n_covid,), pos, dtype=torch.long)
        fake_feats = meter.features(gan.sample(labels, gen=torch.Generator().manual_seed(321)))
        mu_r, s_r = meter.stats(real_pos_feats)
        mu_f, s_f = meter.stats(fake_feats)
        fid_pos = meter.frechet(mu_r, s_r, mu_f, s_f)
        diversity = mean_pairwise_distance(fake_feats) / mean_pairwise_distance(real_pos_feats)
        fids = history["fid"]
        best = min(fids, key=lambda f: f["fid"])
        grids = sorted(p.name for p in run_out.glob("samples_epoch_*.png"))
        return {
            "strategies": rc.strategies,
            "config": rc.__dict__,
            "epochs": len(history["loss_d"]),
            "n_d_steps": history["n_d_steps"],
            "n_g_steps": history["n_g_steps"],
            "seconds": round(history["seconds"], 1),
            "fid_curve": fids,
            "fid_final": fids[-1]["fid"],
            "fid_final_terms": {k: fids[-1][k] for k in ("mean_term", "cov_term")},
            "fid_best": best["fid"],
            "fid_best_epoch": best["epoch"],
            "fid_best_terms": {k: best[k] for k in ("mean_term", "cov_term")},
            "fid_covid_best": fid_pos["fid"],
            "fid_covid_n": [len(real_pos_feats), len(fake_feats)],
            "curves": {k: history[k] for k in ("loss_d", "loss_g", "d_real", "d_fake")},
            "diagnostico": instability_flags(history, diversity),
            "sample_grids": grids,
        }

    def _plot_gan_comparison(self, doc: dict[str, Any]) -> None:
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
        for name, r in doc["runs"].items():
            ep = np.arange(1, len(r["curves"]["loss_d"]) + 1)
            ax[0].plot(ep, r["curves"]["loss_d"], label=f"D, {name}")
            ax[0].plot(ep, r["curves"]["loss_g"], ls="--", label=f"G, {name}")
            ax[1].plot(ep, r["curves"]["d_fake"], label=f"D(G(z)), {name}")
            ax[2].plot(
                [f["epoch"] for f in r["fid_curve"]],
                [f["fid"] for f in r["fid_curve"]],
                marker="o",
                label=name,
            )
        ax[0].set(title="Perda por época", xlabel="época")
        ax[1].axhline(0.5, color="gray", lw=0.8, ls="--")
        ax[1].set(
            title="D(G(z)): quanto o discriminador acredita nas falsas", xlabel="época", ylim=(0, 1)
        )
        ax[2].set(title="FID durante o treino (menor é melhor)", xlabel="época")
        for a in ax:
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self.out / "gan_comparison.png", dpi=110)
        plt.close(fig)

    def stage_synthetic(self, gan_doc: dict[str, Any]) -> torch.Tensor:
        """Gera n COVID com o melhor checkpoint da variante escolhida; grava PNGs e uma grade."""
        s = self.cfg.synthetic
        folder = self.out / "synthetic"
        files = sorted(folder.glob("covid_*.png")) if folder.exists() else []
        if len(files) == s.n_covid:
            arrs = [np.asarray(Image.open(p).convert("L")) for p in files]
            return torch.from_numpy(np.stack(arrs)).unsqueeze(1)
        folder.mkdir(parents=True, exist_ok=True)
        name = gan_doc["chosen_for_synthetic"]
        rc = next(r for r in self.cfg.gan.runs if r.name == name)
        gan = self._make_gan(rc).to(self.device)
        gan.generator.load_state_dict(
            torch.load(self.out / "gan" / name / "generator_best.pt", map_location=self.device)
        )
        labels = torch.full((s.n_covid,), self.cfg.positive_index, dtype=torch.long)
        imgs = gan.sample(labels, gen=torch.Generator().manual_seed(s.seed))
        u8 = (imgs * 255).round().to(torch.uint8)
        for i, img in enumerate(u8):
            Image.fromarray(img[0].numpy()).save(folder / f"covid_{i:04d}.png")
        save_image(imgs[:64], self.out / "synthetic_grid.png", nrow=8)
        log.info("sintético: %d COVID geradas pela variante %s", s.n_covid, name)
        return u8

    def stage_recall(self, split: CaseSplit, synthetic: torch.Tensor) -> dict[str, Any]:
        """Treina o classificador sem e com sintético, 3 seeds, e mede no teste real."""
        c = self.cfg.classifier
        rec_dir = self.out / "recall"
        if (rec_dir / "recall_summary.json").exists():
            return json.loads((rec_dir / "recall_summary.json").read_text(encoding="utf-8"))
        size = c.image_size
        train_x, train_y = XrayDataset(split.records("train"), size).stack()
        val = XrayDataset(split.records("val"), size).stack()
        test_records = split.records("test")
        test = XrayDataset(test_records, size).stack()
        syn = F.interpolate(
            synthetic.float(), size=(size, size), mode="bilinear", align_corners=False
        )
        syn = syn.round().clamp(0, 255).to(torch.uint8)
        syn_y = torch.full((len(syn),), self.cfg.positive_index, dtype=torch.long)
        conditions = [
            Condition("sem_sintetico", train_x, train_y),
            Condition(
                "com_sintetico", torch.cat([train_x, syn]), torch.cat([train_y, syn_y]), len(syn)
            ),
        ]
        counts = np.array([self.cfg.case.train_counts[k] for k in self.cfg.classes], dtype=float)
        trainer = ClassifierTrainer(
            c, self.cfg.classes, self.cfg.positive_index, counts / counts.sum(), self.device
        )
        exp = RecallExperiment(trainer, c.seeds, rec_dir)
        exp.run(conditions, val, test, [str(r.path) for r in test_records])
        self._plot_false_negatives(rec_dir, test[0])
        return json.loads((rec_dir / "recall_summary.json").read_text(encoding="utf-8"))

    def _plot_false_negatives(self, rec_dir: Path, test_x: torch.Tensor) -> None:
        """Análise de erros: COVID do teste que o classificador não reconheceu (primeira seed)."""
        import pandas as pd

        preds = pd.read_csv(rec_dir / "test_predictions.csv")
        pos = self.cfg.positive_index
        seed0 = self.cfg.classifier.seeds[0]
        conds = list(dict.fromkeys(preds["condition"]))
        fig, axes = plt.subplots(len(conds), 8, figsize=(13, 1.9 * len(conds)), squeeze=False)
        for row, cond in enumerate(conds):
            sub = preds[(preds["condition"] == cond) & (preds["seed"] == seed0)].reset_index(
                drop=True
            )
            fn = sub[(sub["true"] == pos) & (sub["pred"] != pos)]
            for col in range(8):
                ax = axes[row][col]
                ax.axis("off")
                if col < len(fn):
                    i = int(fn.index[col])
                    ax.imshow(test_x[i, 0].numpy(), cmap="gray")
                    ax.set_title(
                        f"previsto: {self.cfg.classes[int(fn.iloc[col]['pred'])]}", fontsize=7
                    )
            axes[row][0].text(
                -0.1,
                0.5,
                f"{cond}\n({len(fn)} FN)",
                transform=axes[row][0].transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )
        fig.suptitle(
            f"COVID do teste real classificadas como outra classe (seed {seed0})", fontsize=9
        )
        fig.tight_layout()
        fig.savefig(rec_dir / "falsos_negativos.png", dpi=110)
        plt.close(fig)

    # ---------- execução ----------

    def run(self) -> dict[str, Any]:
        """Roda todas as etapas e grava o manifesto (só no fim, quando tudo deu certo)."""
        t0 = time.perf_counter()
        if not (self.out / "config.json").exists():
            self.run_dir.write_json("config.json", self.cfg.raw)
        log.info("a4: início, execução %s, dispositivo %s", self.run_dir.run_id, self.device)
        t = time.perf_counter()
        split = self.stage_case()
        self.durations["caso"] = round(time.perf_counter() - t, 1)
        gan_doc = self.stage_gan(split)
        t = time.perf_counter()
        synthetic = self.stage_synthetic(gan_doc)
        self.durations["sintetico"] = round(time.perf_counter() - t, 1)
        t = time.perf_counter()
        summary = self.stage_recall(split, synthetic)
        self.durations["classificador"] = round(time.perf_counter() - t, 1)
        self.durations["total"] = round(time.perf_counter() - t0, 1)
        counts = {
            "dedup": self.dedup_report,
            "caso": split.counts(),
            "sintetico_covid": int(len(synthetic)),
        }
        self.run_dir.write_manifest(
            sources=[self.source_info] if self.source_info else [],
            counts=counts,
            extra={
                "config_sha256": self.cfg.sha256,
                "device": str(self.device),
                "durations_s": self.durations,
                "gan_escolhida": gan_doc["chosen_for_synthetic"],
                "status": "completo",
            },
        )
        log.info(
            "a4: fim em %.0f s, veredito do recall: %s",
            self.durations["total"],
            summary.get("paired_difference", {}).get("verdict"),
        )
        return {"gan": gan_doc, "recall": summary, "counts": counts}
