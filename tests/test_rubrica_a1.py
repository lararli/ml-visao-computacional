"""Um teste por item da rubrica da A1 (ViT).

R06, R08, R09 e R11 testam os módulos feitos à mão (sem dados, sem GPU) e o texto em
reports/sections/a1.md, que está no git. As partes que leem o treino em outputs/ têm o marcador
`data`: precisam de um `make run-a1-scratch` feito.
"""

import json
import re

import pandas as pd
import pytest
import torch
from torch import nn

from visaocomp.a1.finetune import PretrainedVit, ResnetBaseline
from visaocomp.a1.pretrained_report import START as PRETRAINED_START
from visaocomp.a1.scratch_report import START, ScratchReport, latest_run
from visaocomp.paths import repo_root
from visaocomp.vit import (
    MultiHeadAttention,
    PatchEmbedding,
    ScaledDotProductAttention,
    TransformerEncoderBlock,
    VisionTransformer,
    ViTConfig,
    shuffle_patches,
)

A1_MD = repo_root() / "reports" / "sections" / "a1.md"
FIGURES = repo_root() / "reports" / "figures" / "a1"
PENDING = "(texto de análise pendente)"


def _a1_section(title_start: str) -> str:
    """Corpo de uma seção `### <título...>` da parte do ViT do zero no a1.md."""
    assert A1_MD.is_file(), "reports/sections/a1.md não existe: rode `make run-a1-scratch`"
    text = A1_MD.read_text(encoding="utf-8")
    assert START in text, "o a1.md não tem a parte do ViT do zero"
    m = re.search(rf"^### {re.escape(title_start)}.*?$(.*?)(?=^### |^<!-- |\Z)", text, re.M | re.S)
    assert m, f"a1.md não tem a seção '### {title_start}...'"
    body = m.group(1)
    assert PENDING not in body, f"texto de análise pendente em '{title_start}'"
    return body


def _latest_scratch() -> tuple[dict, object]:
    """metrics.json e a pasta do run mais recente do ViT do zero."""
    run = latest_run(repo_root() / "outputs")
    assert run is not None, "nenhum run do ViT do zero em outputs/: rode `make run-a1-scratch`"
    metrics = json.loads((run / "a1_scratch" / "metrics.json").read_text(encoding="utf-8"))
    return metrics, run


def _latest_pretrained() -> tuple[dict, pd.DataFrame]:
    """metrics.json e comparison.csv do run mais recente dos pré-treinados."""
    runs = sorted(
        p for p in (repo_root() / "outputs").glob("*-a1_pretrained")
        if (p / "a1_pretrained" / "comparison.csv").exists()
    )  # fmt: skip
    assert runs, "nenhum run dos pré-treinados em outputs/: rode `make run-a1-pretrained`"
    out = runs[-1] / "a1_pretrained"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    return metrics, pd.read_csv(out / "comparison.csv")


def _pretrained_section(title_start: str) -> str:
    assert PRETRAINED_START in A1_MD.read_text(encoding="utf-8"), "a1.md sem a parte pré-treinada"
    return _a1_section(title_start)


def test_R06_attention_e_multihead_do_zero():
    """SDPA à mão (bate com o PyTorch); MHA com Q, K, V próprios por cabeça, concatenação e W_O."""
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    out, w = ScaledDotProductAttention()(q, k, v)
    assert torch.allclose(out, nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-6)
    mha = MultiHeadAttention(dim=192, num_heads=6).eval()
    assert isinstance(mha, nn.Module) and len(mha.heads) == 6
    projs = [p for h in mha.heads for p in (h.q, h.k, h.v)]
    assert len({id(p.weight) for p in projs}) == 18
    assert all(p.in_features == 192 and p.out_features == 32 for p in projs)
    x = torch.randn(2, 145, 192)
    cat, weights = mha.concat_heads(x)
    assert cat.shape == (2, 145, 192) and weights.shape == (2, 6, 145, 145)
    with torch.no_grad():
        mha.heads[2].q.weight.mul_(3.0)  # mexe só na cabeça 3: só a fatia dela muda
    cat2, _ = mha.concat_heads(x)
    same = torch.isclose(cat, cat2, atol=1e-6).all(dim=(0, 1)).reshape(6, 32).all(dim=1)
    assert same.tolist() == [True, True, False, True, True, True]
    assert mha(x)[0].shape == (2, 145, 192)


def test_R07_heatmap_de_atencao():
    """Mapas sobre exemplos do domínio (piscina, solar e um erro) e a interpretação por escrito."""
    body = _a1_section("Mapas de atenção")
    figs = re.findall(r"!\[[^\]]*\]\(\.\./figures/a1/(attention_[a-z]+\.png)\)", body)
    assert {"attention_pool.png", "attention_solar.png", "attention_erro.png"} <= set(figs)
    for name in figs:
        assert (FIGURES / name).stat().st_size > 5000, name
    words = len(re.findall(r"\w+", body))
    assert words > 250, f"interpretação curta demais ({words} palavras)"
    for word in ("telhado", "piscina", "cabeça"):
        assert word in body.lower(), f"a interpretação não fala de '{word}'"


@pytest.mark.data
def test_R07_mapas_do_run_sao_a_linha_do_cls():
    """Os mapas do run existem e a linha do CLS foi tirada da última camada."""
    metrics, run = _latest_scratch()
    att = metrics["attention"]
    assert "CLS" in att["what"] and att["layer"].startswith("última")
    for name in att["figures"]:
        assert (run / "a1_scratch" / name).stat().st_size > 5000, name
    assert set(att["examples"]) == {"pool", "solar", "erro"}
    assert att["examples"]["erro"]["kind"] in {"falso positivo", "falso negativo"}


def test_R08_encoder_block_completo():
    """Pré-LayerNorm, FFN de duas camadas (GELU), residual; e é o bloco usado pelo ViT."""
    torch.manual_seed(0)
    block = TransformerEncoderBlock(dim=192, num_heads=6, mlp_ratio=4).eval()
    lin = [m for m in block.ffn.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin] == [(192, 768), (768, 192)]
    assert isinstance(block.norm1, nn.LayerNorm) and isinstance(block.norm2, nn.LayerNorm)
    x = torch.randn(1, 145, 192)
    with torch.no_grad():
        for m in (block.attn.out, block.ffn.fc2):
            m.weight.zero_()
            m.bias.zero_()
        assert torch.allclose(block(x)[0], x)  # só o residual sobra
    vit = VisionTransformer(ViTConfig())
    assert len(vit.blocks) == 6
    assert all(isinstance(b, TransformerEncoderBlock) for b in vit.blocks)


def test_R09_positional_encoding():
    """Sem posição, embaralhar os patches não muda o logit (atenção não sabe a ordem); com, muda."""
    torch.manual_seed(0)
    cfg = dict(image_size=32, patch=8, dim=24, depth=2, heads=3, dropout=0.0)
    no_pos = VisionTransformer(ViTConfig(**cfg, use_pos=False)).eval()
    with_pos = VisionTransformer(ViTConfig(**cfg, use_pos=True)).eval()
    with_pos.load_state_dict(no_pos.state_dict())
    with torch.no_grad():
        for m in (no_pos, with_pos):
            for p in m.parameters():
                if p.ndim >= 2:
                    p.copy_(
                        torch.randn_like(p, generator=torch.Generator().manual_seed(p.numel()))
                        * 0.3
                    )
    x = torch.rand(2, 3, 32, 32)
    xs = shuffle_patches(x, 8, torch.randperm(16, generator=torch.Generator().manual_seed(1)))
    with torch.no_grad():
        assert torch.allclose(no_pos(x), no_pos(xs), atol=1e-5)
        assert (with_pos(x) - with_pos(xs)).abs().max() > 1e-2
    body = _a1_section("Positional encoding")
    for word in ("ordem", "permut", "CLS"):
        assert word in body, f"a explicação do R09 não fala de '{word}'"


@pytest.mark.data
def test_R09_numeros_da_permutacao_no_texto_batem_com_o_run():
    metrics, run = _latest_scratch()
    perm = metrics["permutation"]
    assert perm["without_pe_max_abs_logit_change"] < 1e-4
    assert perm["with_pe_mean_abs_prob_change"] > 1e-3
    values = ScratchReport(run).values()
    body = _a1_section("Positional encoding")
    assert values["perm_with_mean"] in body and values["perm_without_max"] in body


def test_R10_bert_vs_vit_pretreino():
    """Texto: MLM (e NSP) do BERT contra a classificação supervisionada do ViT; o que maximizam."""
    body = _pretrained_section("BERT e ViT")
    for term in ("Masked Language Modeling", "Next Sentence Prediction", "ImageNet", "CLS",
                 "maximiza", "MAE"):  # fmt: skip
        assert term in body, term
    assert len(body.split()) > 200


def test_R11_vit_completo_patch_cls_pe():
    """Patch embedding (conv com stride do patch), CLS e posição aprendíveis; imagem -> 2 logits."""
    vit = VisionTransformer(ViTConfig()).eval()
    assert isinstance(vit.patch_embed, PatchEmbedding)
    assert vit.patch_embed.proj.kernel_size == vit.patch_embed.proj.stride == (8, 8)
    params = dict(vit.named_parameters())
    assert params["cls_token"].shape == (1, 1, 192) and params["cls_token"].requires_grad
    assert params["pos_embed"].shape == (1, 145, 192) and params["pos_embed"].requires_grad
    with torch.no_grad():
        logits = vit(torch.rand(3, 3, 96, 96))
    assert logits.shape == (3, 2) and torch.isfinite(logits).all()


def test_R12_vit_do_zero_treinado_e_mapas():
    """O texto descreve as regiões que aparecem nos mapas, com números medidos no teste."""
    body = _a1_section("Mapas de atenção")
    assert "Razão" in body and "telhado" in body and "piscina do OSM" in body
    assert len(re.findall(r"\w+", body)) > 250


@pytest.mark.data
def test_R12_run_do_zero_tem_metricas_mapas_e_regioes():
    metrics, run = _latest_scratch()
    out = run / "a1_scratch"
    t, hist = metrics["train"], metrics["history"]
    assert len(hist["train_loss"]) == t["epochs"] > 0 and 1 <= t["best_epoch"] <= t["epochs"]
    assert (out / "best.pt").stat().st_size > 1_000_000 and (out / "curves.png").exists()
    assert metrics["model"]["params"] > 1_000_000 and metrics["attributes"] == ["pool", "solar"]
    for key in ("test", "test_reviewed"):
        per = metrics[key]["per_attribute"]
        assert set(per) == {"pool", "solar"}, key
        for a, row in per.items():
            assert {"precision", "recall", "f1", "pr_auc", "prevalence"} <= set(row), (key, a)
            assert row["pr_auc"] is not None and row["n_pos"] > 0, (key, a)
    for a, row in metrics["test"]["per_attribute"].items():  # baseline: palpite ao acaso
        assert row["pr_auc"] > row["prevalence"], f"{a}: não bate o acaso no teste"
    assert metrics["test_reviewed"]["n_chips"] == 200
    reg = metrics["regions"]
    assert reg["roof_all"]["n"] > 1000 and reg["pool_all"]["n"] > 0
    assert len(reg["roof_all"]["per_head"]) == metrics["model"]["heads"]
    values = ScratchReport(run).values()
    body = _a1_section("Mapas de atenção")
    for k in ("roof_all_lift", "pool_all_lift"):
        assert values[k] in body, f"o número {k} não está no texto"
    assert metrics["run_id"] in A1_MD.read_text(encoding="utf-8")


@pytest.mark.data
def test_R13_vit_pretreinado_vs_zero():
    """Cabeça trocada por 2 saídas; ajustado nos mesmos recortes; tabela contra o ViT do zero."""
    for cls in (PretrainedVit, ResnetBaseline):
        model = cls(2, pretrained=False)
        assert model.head.out_features == 2
        with torch.no_grad():
            assert model.eval()(torch.rand(2, 3, 96, 96)).shape == (2, 2)
    metrics, comp = _latest_pretrained()
    vit = metrics["models"]["vit_pretrained"]
    assert vit["head"]["out"] == 2 and vit["train"]["epochs"] >= 1
    assert vit["model_version"].startswith("vit_pretrained-")
    test = comp[comp["eval_set"].str.startswith("teste inteiro")]
    assert set(test["model"]) == {"vit_scratch", "vit_pretrained", "resnet18"}
    assert test.groupby("attribute")["n"].nunique().eq(1).all(), "teste diferente entre modelos"
    body = _pretrained_section("Tabela comparativa")
    for label in test["label"].unique():
        assert label in body, label


def test_R14_deit_e_swin():
    """Texto: o que DeiT (dados, destilação) e Swin (janelas, hierarquia) resolvem."""
    body = _pretrained_section("DeiT e Swin")
    for term in ("DeiT", "destilação", "Swin", "janela", "quadrado", "escala"):
        assert term in body, term


@pytest.mark.data
def test_R15_vit_vs_cnn_quando():
    """Texto: quando o ViT ganha e quando a CNN é melhor, com os números do nosso domínio."""
    body = _pretrained_section("Quando o ViT supera")
    for term in ("CNN", "ViT", "localidade", "dados"):
        assert term in body, term
    _, comp = _latest_pretrained()
    test = comp[comp["eval_set"].str.startswith("teste inteiro")]
    assert any(f"{v:.2f}" in body for v in test["pr_auc"]), "R15 não cita os números da tabela"


@pytest.mark.data
def test_R16_tabela_comparativa_e_escolha():
    """Tabela do zero x pré-treinado (e CNN) com as mesmas métricas; escolha justificada."""
    _, comp = _latest_pretrained()
    for set_name in comp["eval_set"].unique():
        part = comp[comp["eval_set"] == set_name]
        assert len(part) == 6, set_name  # 3 modelos x 2 atributos
        assert part[["precision", "recall", "f1", "pr_auc"]].notna().all().all()
    body = _pretrained_section("Tabela comparativa")
    assert "| Modelo |" in body
    assert "escolh" in body.lower(), "a escolha de arquitetura não está justificada"
