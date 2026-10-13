"""AbstainPolicy: ordem dos motivos, faixa escolhida na validação e todo "não dá" com motivo."""

import numpy as np
import pandas as pd
import pytest

from visaocomp.a1.abstain import (
    ABSTAIN,
    NO,
    REASON_NO_BUILDING,
    REASON_NOT_VISIBLE,
    REASON_TREE,
    REASON_UNCERTAIN,
    YES,
    AbstainPolicy,
    Band,
)


@pytest.fixture
def policy() -> AbstainPolicy:
    return AbstainPolicy({"pool": Band(0.2, 0.8, 0.5), "solar": Band(0.3, 0.9, 0.6)})


def test_ordem_dos_motivos(policy):
    # sem prédio vem antes de árvore, que vem antes de incerto
    assert policy.decide("pool", 0.99, has_building=False, yard_tree_frac=0.9) == (
        ABSTAIN,
        REASON_NO_BUILDING,
    )
    assert policy.decide("pool", 0.99, yard_tree_frac=0.9) == (ABSTAIN, REASON_TREE)
    assert policy.decide("solar", 0.99, roof_tree_frac=0.6) == (ABSTAIN, REASON_TREE)
    assert policy.decide("fence", 0.99) == (ABSTAIN, REASON_NOT_VISIBLE)
    assert policy.decide("pool", 0.5) == (ABSTAIN, REASON_UNCERTAIN)


def test_arvore_usa_o_telhado_no_solar_e_o_quintal_na_piscina(policy):
    assert policy.decide("solar", 0.95, roof_tree_frac=0.1, yard_tree_frac=0.99) == (YES, None)
    assert policy.decide("pool", 0.95, roof_tree_frac=0.99, yard_tree_frac=0.1) == (YES, None)


def test_faixa_decide_sim_e_nao(policy):
    assert policy.decide("pool", 0.85) == (YES, None)
    assert policy.decide("pool", 0.1) == (NO, None)
    assert policy.decide("pool", None) == (ABSTAIN, REASON_UNCERTAIN)


def test_fit_escolhe_faixa_que_contem_o_limiar():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(float)  # bem calibrado
    y[:50] = np.nan
    pol = AbstainPolicy.fit(
        {"pool": y}, {"pool": p}, {"pool": 0.5}, min_precision=0.8, min_npv=0.9, min_support=20
    )
    b = pol.bands["pool"]
    assert b.low <= 0.5 <= b.high
    sel = (p >= b.high) & ~np.isnan(y)
    assert y[sel].mean() >= 0.8
    sel = (p <= b.low) & ~np.isnan(y)
    assert 1 - y[sel].mean() >= 0.9


def test_fit_sem_faixa_possivel_cai_no_limiar():
    p = np.linspace(0, 1, 100)
    y = np.zeros(100)  # ninguém é positivo: o sim nunca chega a 60%
    y[::2] = 1  # metade positiva em todo lugar: nem sim nem não alcançam a meta
    pol = AbstainPolicy.fit({"solar": y}, {"solar": p}, {"solar": 0.5})
    b = pol.bands["solar"]
    assert b.low == b.high == 0.5
    assert pol.decide("solar", 0.7) == (YES, None)


def test_todo_nao_da_tem_motivo(policy):
    chips = pd.DataFrame(
        {
            "chip_id": ["a", "b", "c", "d"],
            "prob_pool": [0.9, 0.5, 0.1, 0.95],
            "prob_solar": [0.95, 0.5, 0.1, 0.2],
            "roof_tree_frac": [0.0, 0.0, 0.0, 0.9],
            "yard_tree_frac": [0.0, 0.0, 0.0, 0.9],
        }
    )
    dec = policy.decide_frame(chips, ("pool", "solar"))
    assert len(dec) == 8 and set(dec["decision"]) <= {YES, NO, ABSTAIN}
    assert dec.loc[dec["decision"] == ABSTAIN, "reason"].notna().all()
    assert dec.loc[dec["decision"] != ABSTAIN, "reason"].isna().all()
    d = dec.set_index(["chip_id", "attribute"])
    assert d.loc[("d", "pool"), "reason"] == REASON_TREE
