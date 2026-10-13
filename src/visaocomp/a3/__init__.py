"""A3: classificador com CNN pré-treinada e corpo congelado."""

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
from visaocomp.a3.pipeline import A3Pipeline
from visaocomp.a3.report import SectionWriter
from visaocomp.a3.trainer import History, Trainer

__all__ = [
    "A3Config",
    "A3Pipeline",
    "CnnClassifier",
    "Evaluator",
    "History",
    "ImageFolderDataset",
    "ImageScanner",
    "ScanReport",
    "SectionWriter",
    "Split",
    "Trainer",
    "build_transform",
    "compute_metrics",
    "majority_baseline",
    "stratified_split",
]
