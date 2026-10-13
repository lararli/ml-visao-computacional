"""FID (Fréchet Inception Distance): distância entre a distribuição das reais e a das geradas."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionFeatures(nn.Module):
    """Inception-v3 do ImageNet, congelada, com `fc` trocada pela identidade: 2048 números."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [N, 1 ou 3, H, W] em [0, 1] -> [N, 2048]."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


class FidMeter:
    """Calcula o FID com um extrator de características congelado (Inception por padrão)."""

    def __init__(
        self, extractor: nn.Module, device: torch.device, batch_size: int = 64, size: int = 64
    ) -> None:
        self.extractor = extractor.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.size = size

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> np.ndarray:
        """Características de um conjunto. images [N, C, H, W], uint8 ou float em [0, 1]."""
        out = []
        for i in range(0, len(images), self.batch_size):
            x = images[i : i + self.batch_size].to(self.device).float()
            x = x / 255.0 if images.dtype == torch.uint8 else x
            if x.shape[-1] != self.size:
                # reais e geradas passam pela mesma resolução antes da Inception
                x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
            out.append(self.extractor(x).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

    @staticmethod
    def stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Média e covariância das características."""
        return feats.mean(axis=0), np.cov(feats, rowvar=False)

    @staticmethod
    def psd_sqrt(sigma: np.ndarray) -> np.ndarray:
        """Raiz quadrada de uma matriz simétrica semidefinida positiva (via autovalores)."""
        vals, vecs = np.linalg.eigh(sigma)
        return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T

    @classmethod
    def frechet(
        cls,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        sqrt_sigma1: np.ndarray | None = None,
    ) -> dict[str, float]:
        """FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)), separado nos dois termos.

        Tr((S1 S2)^(1/2)) = Tr((S1^(1/2) S2 S1^(1/2))^(1/2)): mesma conta, mas simétrica e estável.
        """
        diff = mu1 - mu2
        root1 = cls.psd_sqrt(sigma1) if sqrt_sigma1 is None else sqrt_sigma1
        inner = root1 @ sigma2 @ root1
        tr_covmean = float(
            np.sqrt(np.clip(np.linalg.eigvalsh((inner + inner.T) / 2), 0, None)).sum()
        )
        mean_term = float(diff @ diff)
        cov_term = float(np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
        return {"fid": mean_term + cov_term, "mean_term": mean_term, "cov_term": cov_term}

    def reference(self, real_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Média, covariância e raiz da covariância das reais (calculadas uma vez)."""
        mu, sigma = self.stats(real_feats)
        return mu, sigma, self.psd_sqrt(sigma)

    def score(
        self, reference: tuple[np.ndarray, np.ndarray, np.ndarray], fake_images: torch.Tensor
    ) -> dict[str, float]:
        """FID entre a referência real (de `reference`) e um conjunto de imagens geradas."""
        mu_r, s_r, root_r = reference
        mu_f, s_f = self.stats(self.features(fake_images))
        return self.frechet(mu_r, s_r, mu_f, s_f, sqrt_sigma1=root_r)
