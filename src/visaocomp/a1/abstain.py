"""AbstainPolicy: sim, não ou "não dá para dizer", e todo "não dá" com motivo.

Os motivos seguem uma ordem fixa: sem prédio, árvore, não visível de cima, incerto. A faixa de
"incerto" é escolhida na validação: abaixo dela o "não" acerta pelo menos `min_npv`, acima dela o
"sim" acerta pelo menos `min_precision`. Entre as duas, o modelo não decide.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

YES, NO, ABSTAIN = "sim", "não", "não dá"
REASON_NO_BUILDING = "sem prédio"
REASON_TREE = "árvore"
REASON_NOT_VISIBLE = "não visível de cima"
REASON_UNCERTAIN = "incerto"
# Ofícios que a foto de cima não mostra (o PD fala em casas; cortados da POC, mas a regra fica).
NOT_VISIBLE_FROM_ABOVE = frozenset({"fence", "cladding", "window", "door"})


@dataclass(frozen=True)
class Band:
    """Faixa de incerteza de um atributo: p <= low -> não; p >= high -> sim; entre -> não dá."""

    low: float
    high: float
    threshold: float

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError(f"faixa inválida: low {self.low} > high {self.high}")


def _band(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    min_precision: float,
    min_npv: float,
    min_support: int,
) -> Band:
    """Escolhe low e high na validação; sem rótulo (NaN) fica fora."""
    ok = ~np.isnan(y)
    y, p = y[ok], p[ok]
    high = np.inf
    for t in np.unique(p[p >= threshold]):
        sel = p >= t
        if sel.sum() < min_support:
            break
        if y[sel].mean() >= min_precision:
            high = float(t)
            break
    low = -np.inf
    for t in np.unique(p[p <= threshold])[::-1]:
        sel = p <= t
        if sel.sum() < min_support:
            break
        if 1.0 - y[sel].mean() >= min_npv:
            low = float(t)
            break
    if high == np.inf and low == -np.inf:
        low = high = threshold
    return Band(low=low, high=max(high, low), threshold=threshold)


class AbstainPolicy:
    """Decide por atributo com a ordem de motivos; a faixa vem de `fit` na validação."""

    def __init__(
        self,
        bands: dict[str, Band],
        roof_tree_frac: float = 0.5,
        yard_tree_frac: float = 0.75,
    ) -> None:
        self.bands = bands
        self.roof_tree_frac = roof_tree_frac
        self.yard_tree_frac = yard_tree_frac

    @classmethod
    def fit(
        cls,
        labels: dict[str, np.ndarray],
        probs: dict[str, np.ndarray],
        thresholds: dict[str, float],
        min_precision: float = 0.6,
        min_npv: float = 0.95,
        min_support: int = 20,
        **tree: float,
    ) -> AbstainPolicy:
        """Uma faixa por atributo, escolhida nos rótulos e probabilidades da validação."""
        bands = {
            a: _band(labels[a], probs[a], thresholds[a], min_precision, min_npv, min_support)
            for a in probs
        }
        return cls(bands, **tree)

    def _tree(self, attribute: str, roof_frac: float | None, yard_frac: float | None) -> bool:
        def over(v: float | None, limit: float) -> bool:
            return v is not None and not pd.isna(v) and v > limit

        if attribute == "solar":
            return over(roof_frac, self.roof_tree_frac)
        if attribute == "pool":
            return over(yard_frac, self.yard_tree_frac)
        return False

    def decide(
        self,
        attribute: str,
        prob: float | None,
        has_building: bool = True,
        roof_tree_frac: float | None = None,
        yard_tree_frac: float | None = None,
    ) -> tuple[str, str | None]:
        """(decisão, motivo); o motivo só existe quando a decisão é "não dá"."""
        if not has_building:
            return ABSTAIN, REASON_NO_BUILDING
        if self._tree(attribute, roof_tree_frac, yard_tree_frac):
            return ABSTAIN, REASON_TREE
        if attribute in NOT_VISIBLE_FROM_ABOVE:
            return ABSTAIN, REASON_NOT_VISIBLE
        band = self.bands.get(attribute)
        if band is None or prob is None or pd.isna(prob):
            return ABSTAIN, REASON_UNCERTAIN
        if prob >= band.high and band.high > band.low:
            return YES, None
        if prob <= band.low and band.high > band.low:
            return NO, None
        if band.high == band.low:  # sem faixa: o limiar decide
            return (YES, None) if prob >= band.threshold else (NO, None)
        return ABSTAIN, REASON_UNCERTAIN

    def decide_frame(self, chips: pd.DataFrame, attributes: tuple[str, ...]) -> pd.DataFrame:
        """Uma linha por recorte x atributo: chip_id, attribute, probability, decision, reason.

        `chips` precisa de chip_id, prob_<atributo>, roof_tree_frac e yard_tree_frac; a coluna
        `has_building` é opcional (padrão: verdadeiro, porque todo recorte nasce de uma casa).
        """
        rows = []
        for r in chips.itertuples(index=False):
            for a in attributes:
                prob = getattr(r, f"prob_{a}")
                decision, reason = self.decide(
                    a,
                    prob,
                    bool(getattr(r, "has_building", True)),
                    getattr(r, "roof_tree_frac", None),
                    getattr(r, "yard_tree_frac", None),
                )
                rows.append(
                    {
                        "chip_id": r.chip_id,
                        "attribute": a,
                        "probability": float(prob),
                        "decision": decision,
                        "reason": reason,
                    }
                )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bands": {a: asdict(b) for a, b in self.bands.items()},
            "roof_tree_frac": self.roof_tree_frac,
            "yard_tree_frac": self.yard_tree_frac,
            "reason_order": [REASON_NO_BUILDING, REASON_TREE, REASON_NOT_VISIBLE, REASON_UNCERTAIN],
        }
