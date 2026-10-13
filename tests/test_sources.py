"""Testes unitários de config, DataSource/KaggleSource e Catalog (sem rede)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace

import pytest

from conftest import FakeSession, make_zip
from visaocomp.catalog import Catalog
from visaocomp.config import load_data_config, record_sha256
from visaocomp.data import DownloadFailedError, KaggleSource, SourceChangedError
from visaocomp.data.sources import MARKER, READY_MSG
from visaocomp.paths import data_dir, repo_root


def _source(cfg, tmp_path, body, **kw):
    return KaggleSource(cfg, tmp_path / "dados", session=FakeSession(body, **kw), backoff=0)


def test_primeira_execucao_cria_data_dir_e_extrai(source_cfg, tmp_path, zip_bytes):
    src = _source(source_cfg, tmp_path, zip_bytes)
    assert not (tmp_path / "dados").exists()
    assert src.download() is True
    assert (tmp_path / "dados" / "raw" / "fonte" / "a" / "um.txt").read_bytes() == b"um"
    assert src.new_sha256 == hashlib.sha256(zip_bytes).hexdigest()
    result = src.verify()
    assert result.ok and result.n_files == 3
    assert src.session.calls == [source_cfg.url]
    assert "datasetVersionNumber=1" in source_cfg.url


def test_segunda_execucao_nao_baixa_e_avisa(source_cfg, tmp_path, zip_bytes, caplog):
    sha = hashlib.sha256(zip_bytes).hexdigest()
    cfg = replace(source_cfg, sha256=sha)
    _source(cfg, tmp_path, zip_bytes).download()
    offline = _source(cfg, tmp_path, None)  # qualquer uso da rede dá erro
    with caplog.at_level(logging.INFO):
        assert offline.download() is False
    assert READY_MSG in caplog.text
    assert offline.session.calls == []


def test_download_interrompido_descarta_parcial_e_refaz(source_cfg, tmp_path, zip_bytes):
    src = _source(source_cfg, tmp_path, zip_bytes, cut_at=20)
    src.retries = 2
    with pytest.raises(DownloadFailedError):
        src.download()
    assert len(src.session.calls) == 2
    assert not list((tmp_path / "dados").rglob("*.part"))
    assert not src.archive_path.exists()
    ok = _source(source_cfg, tmp_path, zip_bytes)
    assert ok.download() is True and ok.verify().ok


def test_tamanho_menor_que_o_anunciado_e_rejeitado(source_cfg, tmp_path, zip_bytes):
    src = _source(source_cfg, tmp_path, zip_bytes, lie_length=True)
    src.retries = 1
    with pytest.raises(DownloadFailedError, match="truncado"):
        src.download()
    assert not src.archive_path.exists()


def test_checksum_diferente_para_com_erro_e_nao_sobrescreve(source_cfg, tmp_path, zip_bytes):
    sha = hashlib.sha256(zip_bytes).hexdigest()
    cfg = replace(source_cfg, sha256=sha)
    _source(cfg, tmp_path, zip_bytes).download()
    outro = make_zip({"novo.txt": b"mudou"})
    src = _source(cfg, tmp_path, outro)
    with pytest.raises(SourceChangedError, match="a fonte mudou"):
        src.download(force=True)
    assert hashlib.sha256(src.archive_path.read_bytes()).hexdigest() == sha
    assert (src.dest_dir / "a" / "um.txt").exists()


def test_zip_com_caminho_suspeito_e_rejeitado(source_cfg, tmp_path):
    ruim = make_zip({"../fora.txt": b"x"})
    src = _source(source_cfg, tmp_path, ruim)
    with pytest.raises(DownloadFailedError, match="suspeito"):
        src.download()
    assert not (tmp_path / "dados" / "raw" / "fora.txt").exists()


def test_verify_detecta_arquivo_apagado_e_sem_zip(source_cfg, tmp_path, zip_bytes):
    assert _source(source_cfg, tmp_path, None).verify().ok is False
    sha = hashlib.sha256(zip_bytes).hexdigest()
    src = _source(replace(source_cfg, sha256=sha), tmp_path, zip_bytes)
    src.download()
    (src.dest_dir / "tres.txt").unlink()
    assert not src.verify().ok
    assert src.verify().reason == "pasta extraída não bate com o marcador"


def test_load_falha_sem_dados_e_files_ignora_marcador(source_cfg, tmp_path, zip_bytes):
    src = _source(source_cfg, tmp_path, zip_bytes)
    with pytest.raises(FileNotFoundError, match="make data"):
        src.load()
    src.download()
    src.cfg = replace(src.cfg, sha256=src.new_sha256)
    assert len(src.files()) == 3
    assert all(p.name != MARKER for p in src.files())


def test_config_real_tem_tres_fontes_com_versao_fixada():
    cfg = load_data_config()
    assert [s.name for s in cfg.sources] == ["ads16", "images_dataset", "covid_xray"]
    assert {f"{s.owner}/{s.dataset}" for s in cfg.sources} == {
        "groffo/ads16-dataset",
        "pavansanagapati/images-dataset",
        "sachinkumar413/covid-pneumonia-normal-chest-xray-images",
    }
    assert all(s.version == 1 and s.dest.startswith("raw/") for s in cfg.sources)


def test_record_sha256_so_muda_a_fonte_certa_e_mantem_comentarios(tmp_path):
    yaml_text = (
        "# comentário\nsources:\n  - name: a\n    sha256: null\n    dest: x\n"
        "  - name: b\n    sha256: null\n    dest: y\n"
    )
    path = tmp_path / "d.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    record_sha256(path, "b", "ab" * 32)
    out = path.read_text(encoding="utf-8")
    assert out.startswith("# comentário") and out.count("sha256: null") == 1
    assert f"sha256: {'ab' * 32}" in out
    with pytest.raises(KeyError):
        record_sha256(path, "zzz", "00")


def test_data_dir_usa_env_e_padrao(monkeypatch, tmp_path):
    monkeypatch.delenv("VISAOCOMP_DATA_DIR", raising=False)
    assert data_dir() == repo_root() / "data.nosync"
    monkeypatch.setenv("VISAOCOMP_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    assert data_dir("rel") == repo_root() / "rel"


def test_catalog_cria_schemas_vazios_e_e_idempotente(tmp_path):
    path = tmp_path / "novo" / "catalog.duckdb"
    with Catalog(path) as cat:
        assert cat.ensure_schemas() == ["bronze", "silver", "gold"]
        assert cat.ensure_schemas() == ["bronze", "silver", "gold"]
        n = (
            cat.connect()
            .execute(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema IN ('bronze','silver','gold')"
            )
            .fetchone()[0]
        )
        assert n == 0
    assert path.exists()


def test_catalog_recusa_nome_de_schema_invalido(tmp_path):
    with pytest.raises(ValueError):
        Catalog(tmp_path / "c.duckdb", ("bronze; drop",))
