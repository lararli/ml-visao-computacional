"""Unitários da A1: fontes, lote/casa/endereço, ofício, recorte, NDVI, rótulo, split
e catálogo. Fixtures pequenas e sintéticas, sem rede."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from a1_world import (
    LOTS,
    WorldSession,
    address_features,
    buildings_frame,
    osm_elements,
    parcel_features,
    permit_rows,
    stac_doc,
    world_sources,
    write_rasters,
)
from visaocomp.a1.build import (
    REASON_OUTSIDE,
    REASON_TREE,
    AreaSplitter,
    ChipMaker,
    HouseBuilder,
    LabelBuilder,
    NaipRaster,
    NdviReasoner,
    PermitTradeClassifier,
    ReviewSampler,
    chip_id_for,
    osm_pool_geometries,
)
from visaocomp.a1.pipeline import A1Catalog
from visaocomp.a1.review import ReviewPage
from visaocomp.a1.sources import (
    DONE,
    SKIP_MSG,
    ArcgisLayerSource,
    RetryingClient,
    load_a1_config,
    query_key,
    validate_arcgis,
)
from visaocomp.data import DownloadFailedError

CFG = load_a1_config()


@pytest.fixture
def world():
    """Lotes, endereços, prédios e piscinas do mundo sintético, já montados."""
    hb = HouseBuilder(CFG.house)
    parcels, _ = hb.parcels(parcel_features())
    addresses, astats = hb.addresses(address_features(), parcels)
    buildings, _ = hb.buildings(buildings_frame(), parcels)
    houses, lots = hb.houses(parcels, addresses, buildings)
    pools = osm_pool_geometries(osm_elements())
    pools["parcel_id"] = hb._containing([g.representative_point() for g in pools["geometry"]],
                                        parcels)  # fmt: skip
    return {"hb": hb, "parcels": parcels, "addresses": addresses, "astats": astats,
            "buildings": buildings, "houses": houses, "lots": lots, "pools": pools}  # fmt: skip


def reasoner() -> NdviReasoner:
    nd = CFG.ndvi
    return NdviReasoner(nd["veg_threshold"], nd["roof_tree_frac"], nd["yard_tree_frac"])


# -- fontes ---------------------------------------------------------------------------------
def test_cliente_repete_em_erro_e_nunca_devolve_resposta_de_erro():
    session = WorldSession(fail_first=2)  # 504, depois {"error": ...}, depois ok
    client = RetryingClient(session, retries=3, backoff_s=0, sleep=lambda s: None)
    resp = client.call(
        "GET", "https://x/parcel/query", validate_arcgis,
        params={"resultOffset": 0, "resultRecordCount": 10},
    )  # fmt: skip
    assert "error" not in resp.json() and len(resp.json()["features"]) == len(LOTS)
    assert len(session.calls) == 3


def test_cliente_desiste_depois_das_tentativas():
    client = RetryingClient(WorldSession(fail_first=9), retries=2, backoff_s=0, sleep=lambda s: 0)
    with pytest.raises(DownloadFailedError):
        client.call("GET", "https://x/parcel/query", validate_arcgis,
                    params={"resultOffset": 0, "resultRecordCount": 10})  # fmt: skip


def test_arcgis_pagina_grava_cru_e_nao_consulta_de_novo(tmp_path, caplog):
    session = WorldSession(fail_first=1)
    client = RetryingClient(session, retries=3, backoff_s=0, sleep=lambda s: None)
    src = ArcgisLayerSource("parcels", tmp_path, client, "https://x/parcel/query",
                            ["PID_10"], "OBJECTID_1", (0, 0, 1, 1), page_size=3)  # fmt: skip
    assert src.download() is True
    pages = sorted(src.folder.glob("page_*.json"))
    assert len(pages) == 3  # 7 lotes em páginas de 3
    assert all("error" not in p.read_text() for p in pages)
    done = json.loads((src.folder / DONE).read_text())
    assert done["n_records"] == len(LOTS) and done["query"] == src.query()
    assert len(src.features()) == len(LOTS)
    n = len(session.calls)
    import logging

    with caplog.at_level(logging.INFO):
        assert src.download() is False
    assert len(session.calls) == n and SKIP_MSG in caplog.text
    (pages[0]).write_text("{}")  # arquivo alterado: verify falha
    assert not src.verify().ok


def test_chave_da_consulta_e_estavel():
    assert query_key({"a": 1, "b": [1, 2]}) == query_key({"b": [1, 2], "a": 1})
    assert query_key({"a": 1}) != query_key({"a": 2})


# -- licenças -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("NEW POOL DECK", {"pool"}),
        ("Swimming pool and spa", {"pool"}),
        ("DEMO EXISTING POOL", set()),
        ("Remove pool and fill in", set()),
        ("Install rooftop solar PV system", {"solar"}),
        ("Photovoltaic panels on roof", {"solar"}),
        ("Install solar screens on windows", set()),
        ("Carpool parking", set()),
        ("New pool and solar panels", {"pool", "solar"}),
        (None, set()),
    ],
)
def test_oficio_por_palavra(text, expected):
    assert PermitTradeClassifier(CFG.trades).classify(text) == expected


def test_licenca_sem_lote_fica_fora_e_e_contada():
    clf = PermitTradeClassifier(CFG.trades)
    rows = permit_rows() + [permit_rows()[0] | {"issue_date": "2018-01-01T00:00:00.000"}]
    df, stats = clf.events(rows, set(LOTS))
    assert df["permit_number"].is_unique
    assert set(df["permit_number"]) == {"2019-1 BP", "2020-2 EP"}  # demolição sem ofício
    assert stats["sem_lote"] == 1 and stats["com_oficio"] == 3
    assert df.loc[df["permit_number"] == "2019-1 BP", "issue_date"].item() == "2019-05-01"


# -- lote, endereço e casa ------------------------------------------------------------------
def test_casa_e_o_maior_predio_do_lote(world):
    h = world["houses"].set_index("parcel_id")
    assert h.loc["0000000001", "house_id"] == "casa-0000000001"  # não a garagem
    assert h.loc["0000000001", "n_buildings"] == 2


def test_lote_sem_endereco_ou_sem_predio_nao_vira_casa(world):
    pids = set(world["houses"]["parcel_id"])
    assert "0000000003" not in pids and "0000000004" not in pids
    assert world["lots"]["sem_endereco"] == 1 and world["lots"]["sem_predio"] == 1
    assert world["houses"]["address"].ne("").all() and world["houses"]["house_id"].is_unique
    assert world["houses"]["parcel_id"].is_unique


def test_endereco_ignora_hidrante(world):
    assert world["astats"]["tipo_endereco"] == 6  # 7 pontos, 1 hidrante
    assert world["houses"].set_index("parcel_id").loc["0000000001", "address"] == "101 TESTE LN"


def test_pid_repetido_vira_um_lote():
    feats = parcel_features()
    feats.append(feats[0] | {"properties": feats[0]["properties"] | {"OBJECTID_1": 99}})
    df, stats = HouseBuilder(CFG.house).parcels(feats)
    assert df["parcel_id"].is_unique and stats["pid_repetido"] == 1


def test_multifamiliar_e_area_fora_da_faixa_saem(world):
    hb = world["hb"]
    many = pd.concat([world["addresses"]] * 1, ignore_index=True)
    extra = many[many["parcel_id"] == "0000000001"].iloc[[0] * 5].copy()
    extra["address_id"] = range(900, 905)
    houses, lots = hb.houses(world["parcels"], pd.concat([many, extra]), world["buildings"])
    assert "0000000001" not in set(houses["parcel_id"]) and lots["multifamiliar"] == 1


# -- recorte e NDVI -------------------------------------------------------------------------
def test_ndvi_e_motivo_arvore():
    r = reasoner()
    red, nir = np.array([[30, 120]], np.uint8), np.array([[200, 60]], np.uint8)
    v = r.ndvi(red, nir)
    assert v[0, 0] > 0.7 and v[0, 1] < 0
    assert r.veg_fraction(red, nir, np.array([[1, 1]])) == 0.5
    assert r.veg_fraction(red, nir, np.array([[0, 0]])) is None
    assert r.reason("solar", 0.9, 0.0) == REASON_TREE
    assert r.reason("solar", 0.4, 0.99) is None
    assert r.reason("pool", 0.0, 0.9) == REASON_TREE


def test_recorte_96px_vizinha_na_borda_sem_foto_e_reaproveita(world, tmp_path):
    paths = write_rasters(tmp_path / "naip")
    rasters = [NaipRaster(p.stem, 2022, p) for p in paths]
    maker = ChipMaker(rasters, tmp_path / "processed", reasoner(), 96, 0.01)
    chips, stats = maker.make(world["houses"], world["parcels"], world["pools"])
    assert stats["sem_foto"] == 1 and stats["criados"] == len(world["houses"]) - 1
    c = chips.set_index("parcel_id")
    assert "0000000007" not in c.index
    assert c.loc["0000000005", "item_id"] == "item_b"  # a quadrícula A não cobre a janela
    from PIL import Image

    img = Image.open(tmp_path / c.loc["0000000001", "path"].replace("processed/", "processed/"))
    assert img.size == (96, 96) and img.mode == "RGB"
    nir = Image.open(tmp_path / c.loc["0000000001", "nir_path"])
    assert nir.mode == "L"
    assert c.loc["0000000002", "roof_tree_frac"] > 0.9  # árvore sobre a casa 2
    assert c.loc["0000000001", "roof_tree_frac"] == 0.0
    assert chips["chip_id"].is_unique
    assert c.loc["0000000001", "chip_id"] == chip_id_for("item_a", "casa-0000000001", 2022)
    again, stats2 = maker.make(world["houses"], world["parcels"], world["pools"])
    assert stats2["criados"] == 0 and stats2["reaproveitados"] == stats["criados"]
    pd.testing.assert_frame_equal(chips, again)


# -- rótulo, split e amostra ----------------------------------------------------------------
def test_rotulos_com_evidencia_motivo_e_fora_do_recorte(world, tmp_path):
    paths = write_rasters(tmp_path / "naip")
    maker = ChipMaker([NaipRaster(p.stem, 2022, p) for p in paths], tmp_path / "p", reasoner())
    chips, _ = maker.make(world["houses"], world["parcels"], world["pools"])
    permits, _ = PermitTradeClassifier(CFG.trades).events(permit_rows(), set(LOTS))
    labels = LabelBuilder(reasoner()).build(chips, permits, world["pools"])
    assert not labels.duplicated(["chip_id", "attribute"]).any()
    lab = labels.merge(chips[["chip_id", "parcel_id"]]).set_index(["parcel_id", "attribute"])
    assert lab.loc[("0000000001", "pool"), "label"] == 1
    assert lab.loc[("0000000001", "pool"), "label_source"] == "osm+permit"
    assert pd.isna(lab.loc[("0000000002", "solar"), "label"])
    assert lab.loc[("0000000002", "solar"), "reason"] == REASON_TREE
    assert lab.loc[("0000000002", "solar"), "has_evidence"]
    assert lab.loc[("0000000006", "pool"), "reason"] == REASON_OUTSIDE
    assert lab.loc[("0000000005", "pool"), "label"] == 0  # licença de demolição não conta
    assert lab.loc[("0000000005", "pool"), "label_source"] == "nenhuma"


def test_split_por_celula_nao_divide_celula_e_e_deterministico():
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(0, 8000, 3000), rng.uniform(0, 8000, 3000)
    sp = AreaSplitter(1000, {"train": 0.7, "val": 0.15, "test": 0.15}, 42)
    cells = [sp.cell_of(x, y) for x, y in zip(xs, ys, strict=True)]
    a, b = sp.assign(cells), sp.assign(cells)
    assert a == b and set(a.values()) == {"train", "val", "test"}
    split = pd.Series([a[c] for c in cells])
    frac = split.value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8 and 0.08 < frac["test"] < 0.22
    df = pd.DataFrame({"cell": cells, "split": split})
    assert (df.groupby("cell")["split"].nunique() == 1).all()
    assert sp.cell_of(1999.9, 0) == "E1_N0" and sp.cell_of(2000, 0) == "E2_N0"


def test_amostra_respeita_cotas_e_completa_estrato_pequeno():
    n = 400
    wide = pd.DataFrame(
        {"chip_id": [f"c{i:04d}" for i in range(n)],
         "pool": pd.array([1] * 30 + [0] * 340 + [None] * 30, dtype="Int64"),
         "solar": pd.array([0] * 30 + [1] * 10 + [0] * 330 + [0] * 30, dtype="Int64")}
    )  # fmt: skip
    s = ReviewSampler({"pool_pos": 60, "solar_pos": 50, "empty": 20, "neg": 70}, 42).sample(wide)
    assert len(s) == 200 and s["chip_id"].is_unique
    assert (s["stratum"] == "pool_pos").sum() == 30 and (s["stratum"] == "solar_pos").sum() == 10
    s2 = ReviewSampler({"pool_pos": 60, "solar_pos": 50, "empty": 20, "neg": 70}, 42).sample(wide)
    pd.testing.assert_frame_equal(s, s2)


# -- catálogo e conferência -----------------------------------------------------------------
def test_merge_no_catalogo_e_idempotente_e_poda(tmp_path):
    cat = A1Catalog(tmp_path / "c.duckdb", wait_s=0, tries=1)
    df = pd.DataFrame({"chip_id": ["a", "a", "b"], "attribute": ["pool", "solar", "pool"],
                       "label": [1, 0, None]})  # fmt: skip
    assert cat.merge_all({"silver.label": df}, prune={"silver.label"}) == {"silver.label": 3}
    assert cat.merge_all({"silver.label": df}, prune={"silver.label"}) == {"silver.label": 3}
    assert cat.merge_all({"silver.label": df.iloc[:2]}, prune={"silver.label"}) == {
        "silver.label": 2
    }
    with pytest.raises(ValueError, match="repetida"):
        cat.merge_all({"silver.label": pd.concat([df, df])}, prune=set())


def test_pagina_de_revisao_valida_e_mostra_cada_recorte(tmp_path):
    from PIL import Image

    (tmp_path / "processed" / "chips").mkdir(parents=True)
    Image.new("RGB", (96, 96)).save(tmp_path / "processed" / "chips" / "c1.png")
    sample = pd.DataFrame({"chip_id": ["c1"], "path": ["processed/chips/c1.png"],
                           "stratum": ["neg"]})  # fmt: skip
    inicial = pd.DataFrame(
        [{"chip_id": "c1", "address": "1 RUA", "parcel_id": "1", "house_id": "h",
          "label_pool": "0", "label_solar": None, "reason_pool": None, "reason_solar": "arvore",
          "inicial_pool": "sim", "inicial_pool_reason": "piscina azul", "inicial_solar": "não dá",
          "inicial_solar_reason": "árvore", "observacao": None}]
    )  # fmt: skip
    html = ReviewPage(tmp_path).render(sample, inicial)
    assert html.count('class="card"') == 1 and "1 RUA" in html and "data:image/png" in html
    assert "vazio (arvore)" in html
    with pytest.raises(ValueError, match="inválidos"):
        ReviewPage(tmp_path).render(sample, inicial.assign(inicial_pool="talvez"))


def test_stac_doc_tem_um_item_por_raster(tmp_path):
    doc = stac_doc(write_rasters(tmp_path))
    assert [f["id"] for f in doc["features"]] == ["item_a", "item_b"]


def test_world_sources_nao_usa_rede(tmp_path):
    srcs = world_sources(tmp_path / "d", WorldSession(), tmp_path / "r")
    assert set(srcs) == {"naip", "parcels", "addresses", "buildings", "osm_pools", "permits"}


def test_merge_aceita_coluna_nova_e_tipo_novo(tmp_path):
    cat = A1Catalog(tmp_path / "c.duckdb", wait_s=0, tries=1)
    df = pd.DataFrame({"chip_id": ["a"], "attribute": ["pool"], "first_check": [None]})
    cat.merge_all({"silver.label": df}, prune=set())
    df2 = df.assign(first_check=["não dá"], lara_check=["sim"])
    cat.merge_all({"silver.label": df2}, prune=set())
    import duckdb

    con = duckdb.connect(str(tmp_path / "c.duckdb"), read_only=True)
    row = con.execute("SELECT first_check, lara_check FROM silver.label").fetchone()
    con.close()
    assert row == ("não dá", "sim")
