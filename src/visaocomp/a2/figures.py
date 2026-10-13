"""FigureBuilder: grades de imagens com legenda (top 5 objetos, consultas, conferência)."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

Cell = tuple[str, str]  # (image_path, legenda)


class FigureBuilder:
    """Monta figuras a partir de `image_path` lógicos; a imagem ilegível vira um quadro cinza."""

    def __init__(self, resolve: Mapping[str, Path], thumb: int = 256) -> None:
        self.resolve = resolve
        self.thumb = thumb

    def _load(self, key: str) -> Image.Image:
        try:
            with Image.open(self.resolve[key]) as im:
                im = im.convert("RGB")
                im.thumbnail((self.thumb, self.thumb))
                return im.copy()
        except (KeyError, OSError, ValueError):
            return Image.new("RGB", (self.thumb, self.thumb), "#cccccc")

    def grid(
        self,
        rows: Sequence[Sequence[Cell]],
        path: Path,
        row_labels: Sequence[str] | None = None,
        title: str | None = None,
        cell: float = 2.1,
    ) -> Path:
        """Uma linha por item de `rows`; cada célula mostra a imagem e a legenda."""
        nrows, ncols = len(rows), max(len(r) for r in rows)
        label_w = 1.9 if row_labels else 0.0
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * cell + label_w, nrows * (cell + 0.45) + (0.5 if title else 0.1)),
            squeeze=False,
        )
        for i, row in enumerate(rows):
            for j in range(ncols):
                ax = axes[i][j]
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if j < len(row):
                    key, caption = row[j]
                    ax.imshow(self._load(key))
                    ax.set_title(caption, fontsize=7, pad=2)
                else:
                    ax.axis("off")
            if row_labels:
                axes[i][0].set_ylabel(
                    "\n".join(textwrap.wrap(row_labels[i], 18)),
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                    labelpad=6,
                )
        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path

    def sheet(self, cells: Sequence[Cell], path: Path, ncols: int = 5) -> Path:
        """Folha de conferência: as células em linhas de `ncols`."""
        rows = [list(cells[i : i + ncols]) for i in range(0, len(cells), ncols)]
        return self.grid(rows, path, cell=2.3)
