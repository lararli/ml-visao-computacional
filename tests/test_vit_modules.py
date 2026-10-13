"""Módulos do ViT feito à mão: formas, cabeças independentes, residual e o teste de permutação."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from visaocomp.vit import (
    MultiHeadAttention,
    PatchEmbedding,
    ScaledDotProductAttention,
    TransformerEncoderBlock,
    VisionTransformer,
    ViTConfig,
    shuffle_patches,
)

TINY = ViTConfig(image_size=32, patch=8, dim=24, depth=2, heads=3, dropout=0.0, num_outputs=2)


def _vit(**kw) -> VisionTransformer:
    torch.manual_seed(0)
    cfg = ViTConfig(**{**TINY.to_dict(), **kw})
    return VisionTransformer(cfg).eval()


def _strong(model: VisionTransformer) -> VisionTransformer:
    """Pesos grandes: a atenção deixa de ser quase uniforme, como depois de treinar."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.3)
    return model


# -- atenção ------------------------------------------------------------------------------------
def test_sdpa_formas_pesos_somam_um_e_bate_com_o_pytorch():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 5, 8), torch.randn(2, 7, 8), torch.randn(2, 7, 4)
    out, w = ScaledDotProductAttention()(q, k, v)
    assert out.shape == (2, 5, 4) and w.shape == (2, 5, 7)
    assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-6)
    ref = nn.functional.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)


def test_sdpa_mascara_zera_o_que_nao_pode_ver():
    q = k = v = torch.randn(1, 4, 8)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    _, w = ScaledDotProductAttention()(q, k, v, causal)
    assert torch.all(w[0][~causal] == 0)


def test_multihead_tem_projecoes_independentes_por_cabeca():
    mha = MultiHeadAttention(dim=24, num_heads=3)
    assert len(mha.heads) == 3 and mha.head_dim == 8
    params = set()
    for head in mha.heads:
        for proj in (head.q, head.k, head.v):
            assert isinstance(proj, nn.Linear) and proj.in_features == 24
            assert proj.out_features == 8
            params.add(id(proj.weight))
    assert len(params) == 9, "cada cabeça precisa das suas próprias matrizes Q, K e V"
    assert mha.out.in_features == mha.out.out_features == 24


def test_multihead_concatena_as_cabecas_e_cada_cabeca_so_mexe_na_sua_fatia():
    torch.manual_seed(0)
    mha = MultiHeadAttention(dim=24, num_heads=3).eval()
    x = torch.randn(2, 6, 24)
    cat, w = mha.concat_heads(x)
    assert cat.shape == (2, 6, 24) and w.shape == (2, 3, 6, 6)
    out, w2 = mha(x)
    assert out.shape == (2, 6, 24) and torch.equal(w, w2)
    with torch.no_grad():
        mha.heads[1].v.weight.add_(1.0)
    cat2, _ = mha.concat_heads(x)
    assert torch.allclose(cat[..., :8], cat2[..., :8])
    assert torch.allclose(cat[..., 16:], cat2[..., 16:])
    assert not torch.allclose(cat[..., 8:16], cat2[..., 8:16])


def test_multihead_recusa_dimensao_que_nao_divide():
    with pytest.raises(ValueError, match="não divide"):
        MultiHeadAttention(dim=10, num_heads=3)


# -- bloco encoder ------------------------------------------------------------------------------
def test_encoder_block_tem_layernorm_ffn_de_duas_camadas_e_residual():
    torch.manual_seed(0)
    block = TransformerEncoderBlock(dim=24, num_heads=3, mlp_ratio=4).eval()
    assert isinstance(block.norm1, nn.LayerNorm) and isinstance(block.norm2, nn.LayerNorm)
    linears = [m for m in block.ffn.modules() if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(24, 96), (96, 24)]
    assert isinstance(block.ffn.act, nn.GELU)
    x = torch.randn(2, 5, 24)
    y, w = block(x)
    assert y.shape == x.shape and w.shape == (2, 3, 5, 5)
    with torch.no_grad():  # sem atenção e sem FFN, sobra só o caminho residual
        for lin in (block.attn.out, block.ffn.fc2):
            lin.weight.zero_()
            lin.bias.zero_()
    y0, _ = block(x)
    assert torch.allclose(y0, x)


# -- patches, CLS, posição e ViT ---------------------------------------------------------------
def test_patch_embedding_e_convolucao_com_stride_do_patch_em_ordem_de_leitura():
    torch.manual_seed(0)
    pe = PatchEmbedding(image_size=96, patch=8, in_channels=3, dim=192)
    assert pe.proj.kernel_size == (8, 8) and pe.proj.stride == (8, 8)
    x = torch.randn(2, 3, 96, 96)
    tokens = pe(x)
    assert tokens.shape == (2, 144, 192)
    perm = torch.randperm(144)
    assert torch.allclose(pe(shuffle_patches(x, 8, perm)), tokens[:, perm], atol=1e-5)


def test_vit_do_brief_recebe_imagem_e_devolve_dois_logits():
    model = VisionTransformer(ViTConfig()).eval()  # 96, patch 8, dim 192, 6 blocos, 6 cabeças
    assert isinstance(model.cls_token, nn.Parameter) and model.cls_token.requires_grad
    assert model.cls_token.shape == (1, 1, 192)
    assert model.pos_embed.shape == (1, 145, 192) and model.pos_embed.requires_grad
    assert len(model.blocks) == 6 and all(len(b.attn.heads) == 6 for b in model.blocks)
    with torch.no_grad():
        logits, weights = model.forward_with_attention(torch.rand(2, 3, 96, 96))
    assert logits.shape == (2, 2)
    assert len(weights) == 6 and weights[-1].shape == (2, 6, 145, 145)


def test_vit_mesma_seed_mesma_saida():
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(_vit()(x), _vit()(x))


def test_permutacao_sem_posicao_so_embaralha_com_posicao_muda():
    """Sem posição, a atenção não sabe a ordem: embaralhar os patches só embaralha os tokens de
    saída, e o CLS (que olha para todos) dá o mesmo logit. Com posição, o logit muda."""
    x = torch.rand(2, 3, 32, 32)
    perm = torch.tensor([3, 0, 15, 7, 1, 2, 9, 4, 5, 6, 8, 10, 11, 12, 14, 13])
    xs = shuffle_patches(x, 8, perm)

    no_pos = _strong(_vit(use_pos=False))
    with torch.no_grad():
        seq, _ = no_pos.encode(x)
        seq_s, _ = no_pos.encode(xs)
        assert torch.allclose(seq_s[:, 1:], seq[:, 1:][:, perm], atol=1e-5)
        assert torch.allclose(no_pos(xs), no_pos(x), atol=1e-5)

    with_pos = _strong(_vit(use_pos=True))  # mesmos pesos, agora somando a posição
    with torch.no_grad():
        assert (with_pos(xs) - with_pos(x)).abs().max() > 1e-2
