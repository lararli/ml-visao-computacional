"""A3Pipeline: do diretório de imagens ao metrics.json, em outputs/<run_id>/."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from visaocomp.a3.config import A3Config
from visaocomp.a3.dataset import (
    ImageFolderDataset,
    ImageScanner,
    ScanReport,
    Split,
    build_transform,
    stratified_split,
)
from visaocomp.a3.evaluator import Evaluator, compute_metrics, majority_baseline
from visaocomp.a3.model import CnnClassifier
from visaocomp.a3.plots import plot_confusion, plot_curves
from visaocomp.a3.report import SectionWriter
from visaocomp.a3.trainer import Trainer
from visaocomp.device import seed_everything
from visaocomp.run import RunDir

log = logging.getLogger(__name__)


class A3Pipeline:
    """Varre, divide, treina a cabeça, avalia no teste uma vez e grava os resultados."""

    def __init__(
        self,
        cfg: A3Config,
        image_root: Path,
        source_info: dict[str, Any],
        outputs_root: Path,
        device: torch.device,
        section_path: Path | None = None,
        max_errors: int = 12,
    ) -> None:
        self.cfg = cfg
        self.image_root = Path(image_root)
        self.source_info = source_info
        self.outputs_root = Path(outputs_root)
        self.device = device
        self.section_path = section_path
        self.max_errors = max_errors

    def _loader(self, ds: ImageFolderDataset, shuffle: bool) -> DataLoader:
        gen = torch.Generator().manual_seed(self.cfg.seed)
        workers = self.cfg.num_workers
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            persistent_workers=workers > 0,
            generator=gen if shuffle else None,
        )

    def _errors(
        self, ds: ImageFolderDataset, pred: Any, classes: list[str]
    ) -> list[dict[str, Any]]:
        wrong = np.flatnonzero(pred.labels != pred.preds)
        wrong = wrong[np.argsort(-pred.confidence[wrong])][: self.max_errors]
        return [
            {
                "path": ds.records[i].path,
                "true": classes[pred.labels[i]],
                "pred": classes[pred.preds[i]],
                "confidence": float(pred.confidence[i]),
            }
            for i in wrong
        ]

    def run(self) -> Path:
        """Executa a A3 inteira e devolve o caminho do manifest.json."""
        t_start = time.perf_counter()
        cfg = self.cfg
        seed_everything(cfg.seed)
        log.info("A3: início, imagens em %s", self.image_root)
        scan: ScanReport = ImageScanner(self.image_root, cfg.extensions, cfg.dedupe).scan()
        classes = scan.classes
        labels = [r.label for r in scan.records]
        split: Split = stratified_split(labels, cfg.train_frac, cfg.val_frac, cfg.seed)
        full = ImageFolderDataset(
            self.image_root, scan.records, build_transform(cfg.resize, cfg.crop)
        )
        train_ds, val_ds, test_ds = (full.subset(s) for s in (split.train, split.val, split.test))

        run = RunDir(self.outputs_root, "a3")
        a3_dir = run.path / "a3"
        a3_dir.mkdir()
        model = CnnClassifier(len(classes), cfg.weights, cfg.dropout)
        trainer = Trainer(model, cfg, self.device, checkpoint=a3_dir / "best_head.pt")
        t_train = time.perf_counter()
        history = trainer.fit(self._loader(train_ds, True), self._loader(val_ds, False))
        train_seconds = time.perf_counter() - t_train

        pred = Evaluator(model, self.device).predict(self._loader(test_ds, False))
        test = compute_metrics(pred.labels, pred.preds, classes)
        per_split = {
            name: {
                c: sum(1 for i in idx if scan.records[i].label == k) for k, c in enumerate(classes)
            }
            for name, idx in (("train", split.train), ("val", split.val), ("test", split.test))
        }
        metrics: dict[str, Any] = {
            "run_id": run.run_id,
            "classes": classes,
            "n_classes": len(classes),
            **{k: test[k] for k in ("accuracy_global", "balanced_accuracy", "macro_f1")},
            "test_loss": pred.loss,
            "per_class": test["per_class"],
            "confusion_matrix": test["confusion_matrix"],
            "top_confusions": test["top_confusions"],
            "baseline": majority_baseline([labels[i] for i in split.train], pred.labels, classes),
            "errors": self._errors(test_ds, pred, classes),
            "model": {
                "arch": cfg.arch,
                "weights": cfg.weights,
                **model.param_counts(),
                "body_requires_grad": any(
                    p.requires_grad
                    for n, p in model.named_parameters()
                    if not n.startswith("net.classifier.")
                ),
            },  # fmt: skip
            "split": {
                "seed": cfg.seed,
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
                "per_class": per_split,
            },  # fmt: skip
            "train": {
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "optimizer": cfg.optimizer,
                "device": str(self.device),
                "best_epoch": history.best_epoch,
                "seconds": train_seconds,
                "best_val_acc": history.val_acc[history.best_epoch - 1],
            },  # fmt: skip
            "history": history.to_dict(),
            "data": scan.counts(),
        }
        run.write_json("a3/metrics.json", metrics)
        pd.DataFrame(
            {
                "path": [r.path for r in test_ds.records],
                "sha256": [r.sha256 for r in test_ds.records],
                "true": [classes[k] for k in pred.labels],
                "pred": [classes[k] for k in pred.preds],
                "confidence": pred.confidence,
            }
        ).to_csv(a3_dir / "predictions.csv", index=False)  # vira gold.images_prediction
        plot_curves(metrics["history"], a3_dir / "curves.png")
        plot_confusion(metrics["confusion_matrix"], classes, a3_dir / "confusion.png")
        if self.section_path is not None:
            SectionWriter(metrics).write(self.section_path)
        manifest = run.write_manifest(
            [self.source_info],
            {
                **scan.counts(),
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
            },
            extra={
                "config": cfg.to_dict(),
                "device": str(self.device),
                "skipped_files": scan.skipped,
                "removed_duplicates": scan.duplicates,
                "ignored_dirs": scan.ignored_dirs,
                "timings_seconds": {
                    "train": round(train_seconds, 1),
                    "total": round(time.perf_counter() - t_start, 1),
                },
            },
        )
        log.info(
            "A3: fim, accuracy global %.4f no teste (%d imagens); resultados em %s",
            metrics["accuracy_global"], len(split.test), a3_dir,
        )  # fmt: skip
        return manifest
