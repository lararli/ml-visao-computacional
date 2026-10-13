"""Testes unitários do RunDir e do manifesto."""

from __future__ import annotations

import json

import pytest

from visaocomp.run import RunDir, git_info, latest_manifest


def test_runs_nunca_se_sobrescrevem(tmp_path):
    a = RunDir(tmp_path / "outputs", "data", run_id="r1")
    b = RunDir(tmp_path / "outputs", "data", run_id="r1")
    assert a.path != b.path and a.path.is_dir() and b.path.is_dir()
    assert b.run_id == "r1-2"


def test_manifesto_tem_fonte_versao_sha_contagens_e_commit(tmp_path):
    run = RunDir(tmp_path, "data")
    fonte = {"name": "x", "version": 1, "sha256": "ab" * 32, "n_files": 3}
    path = run.write_manifest([fonte], {"files": 3})
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert path == run.path / "manifest.json"
    assert doc["stage"] == "data" and doc["run_id"] == run.run_id
    assert doc["sources"] == [fonte] and doc["counts"] == {"files": 3}
    assert set(doc["git"]) == {"commit", "dirty"}


def test_write_json_nao_sobrescreve(tmp_path):
    run = RunDir(tmp_path, "x")
    run.write_json("metrics.json", {"a": 1})
    with pytest.raises(FileExistsError):
        run.write_json("metrics.json", {"a": 2})


def test_primeira_execucao_cria_a_pasta_de_saida(tmp_path):
    assert not (tmp_path / "fundo" / "outputs").exists()
    RunDir(tmp_path / "fundo" / "outputs", "data")
    assert (tmp_path / "fundo" / "outputs").is_dir()


def test_latest_manifest_pega_o_mais_recente_da_etapa(tmp_path):
    assert latest_manifest(tmp_path, "data") is None
    RunDir(tmp_path, "data", run_id="a").write_manifest([], {"n": 1})
    RunDir(tmp_path, "train", run_id="b").write_manifest([], {"n": 2})
    RunDir(tmp_path, "data", run_id="c").write_manifest([], {"n": 3})
    assert latest_manifest(tmp_path, "data")["counts"] == {"n": 3}


def test_git_info_fora_de_repo_nao_quebra(tmp_path):
    info = git_info(tmp_path)
    assert info["commit"] == "sem-commit" or len(info["commit"]) == 40
