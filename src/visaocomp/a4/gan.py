"""GAN condicional (DCGAN com rótulo), replay buffer e o loop adversarial."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.utils.parametrizations import spectral_norm  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from visaocomp.a4.config import GanConfig, GanRunConfig  # noqa: E402

log = logging.getLogger(__name__)


def init_dcgan_weights(module: nn.Module) -> None:
    """Inicialização do artigo da DCGAN (Radford et al.): normal(0, 0.02)."""
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class ConditionalGenerator(nn.Module):
    """Gerador: ruído z + embedding da classe -> imagem 1 x 64 x 64 em [-1, 1]."""

    def __init__(self, nz: int, n_classes: int, ngf: int = 64, emb_dim: int = 50) -> None:
        super().__init__()
        self.nz = nz
        self.embed = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + emb_dim, ngf * 8, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),  # 64x64
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """z [N, nz], labels [N] -> imagens [N, 1, 64, 64]."""
        x = torch.cat([z, self.embed(labels)], dim=1)
        return self.net(x[:, :, None, None])


class ConditionalDiscriminator(nn.Module):
    """Discriminador: imagem + mapa da classe -> logit (real > 0). Spectral norm opcional."""

    def __init__(
        self, n_classes: int, ndf: int = 64, image_size: int = 64, use_spectral_norm: bool = False
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.embed = nn.Embedding(n_classes, image_size * image_size)

        def conv(cin: int, cout: int, k: int, s: int, p: int) -> nn.Module:
            layer = nn.Conv2d(cin, cout, k, s, p, bias=False)
            return spectral_norm(layer) if use_spectral_norm else layer

        def norm(c: int) -> nn.Module:
            # com spectral norm a normalização já é feita nos pesos; BatchNorm sai
            return nn.Identity() if use_spectral_norm else nn.BatchNorm2d(c)

        self.net = nn.Sequential(
            conv(2, ndf, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2, True),
            conv(ndf, ndf * 2, 4, 2, 1),  # 16x16
            norm(ndf * 2),
            nn.LeakyReLU(0.2, True),
            conv(ndf * 2, ndf * 4, 4, 2, 1),  # 8x8
            norm(ndf * 4),
            nn.LeakyReLU(0.2, True),
            conv(ndf * 4, ndf * 8, 4, 2, 1),  # 4x4
            norm(ndf * 8),
            nn.LeakyReLU(0.2, True),
            conv(ndf * 8, 1, 4, 1, 0),  # 1x1
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """imagens [N, 1, 64, 64], labels [N] -> logits [N]."""
        cond = self.embed(labels).view(-1, 1, self.image_size, self.image_size)
        return self.net(torch.cat([images, cond], dim=1)).view(-1)


class ConditionalGan(nn.Module):
    """Gerador e discriminador condicionais, com amostragem por classe."""

    def __init__(
        self,
        nz: int,
        n_classes: int,
        ngf: int = 64,
        ndf: int = 64,
        image_size: int = 64,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("a arquitetura DCGAN desta entrega gera 64 x 64")
        self.nz = nz
        self.n_classes = n_classes
        self.generator = ConditionalGenerator(nz, n_classes, ngf)
        self.discriminator = ConditionalDiscriminator(n_classes, ndf, image_size, use_spectral_norm)
        self.generator.apply(init_dcgan_weights)
        self.discriminator.apply(init_dcgan_weights)

    def noise(
        self, n: int, device: torch.device, gen: torch.Generator | None = None
    ) -> torch.Tensor:
        """Ruído z ~ N(0, 1), [n, nz]."""
        return torch.randn(n, self.nz, generator=gen, device="cpu").to(device)

    @torch.no_grad()
    def sample(
        self, labels: torch.Tensor, gen: torch.Generator | None = None, batch: int = 256
    ) -> torch.Tensor:
        """Gera imagens em [0, 1] para os rótulos dados (modo de avaliação)."""
        device = next(self.parameters()).device
        was_training = self.generator.training
        self.generator.eval()
        out = []
        for i in range(0, len(labels), batch):
            lab = labels[i : i + batch].to(device)
            out.append((self.generator(self.noise(len(lab), device, gen), lab) + 1) / 2)
        self.generator.train(was_training)
        return torch.cat(out).clamp(0, 1).cpu()


class ReplayBuffer:
    """Guarda imagens falsas antigas e as devolve ao discriminador com probabilidade p."""

    def __init__(self, capacity: int, prob: float = 0.5, seed: int = 0) -> None:
        self.capacity = capacity
        self.prob = prob
        self.rng = np.random.default_rng(seed)
        self.images: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def push_and_pop(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coloca o lote no buffer e devolve um lote do mesmo tamanho, misturando antigas."""
        if self.capacity <= 0 or self.prob <= 0:
            return images, labels
        images, labels = images.detach(), labels.detach()
        if self.images is None:
            self.images = torch.empty((self.capacity, *images.shape[1:]), device=images.device)
            self.labels = torch.empty(self.capacity, dtype=labels.dtype, device=labels.device)
        assert self.labels is not None
        out_img, out_lab = images.clone(), labels.clone()
        for i in range(len(images)):
            if self.size < self.capacity:
                self.images[self.size] = images[i]
                self.labels[self.size] = labels[i]
                self.size += 1
            elif self.rng.random() < self.prob:
                j = int(self.rng.integers(self.capacity))
                out_img[i], out_lab[i] = self.images[j].clone(), self.labels[j].clone()
                self.images[j], self.labels[j] = images[i], labels[i]
        return out_img, out_lab


@dataclass
class GanHistory:
    """Curvas por época de um treino da GAN."""

    loss_d: list[float] = field(default_factory=list)
    loss_g: list[float] = field(default_factory=list)
    d_real: list[float] = field(default_factory=list)
    d_fake: list[float] = field(default_factory=list)
    fid: list[dict[str, float]] = field(default_factory=list)
    n_d_steps: int = 0
    n_g_steps: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Histórico como dicionário."""
        return self.__dict__.copy()


class GanTrainer:
    """Treina a GAN alternando um passo do discriminador e um do gerador por lote."""

    def __init__(
        self,
        gan: ConditionalGan,
        cfg: GanConfig,
        run_cfg: GanRunConfig,
        device: torch.device,
        out_dir: Path | None = None,
        fid_fn: Any = None,
    ) -> None:
        self.gan = gan.to(device)
        self.cfg = cfg
        self.run_cfg = run_cfg
        self.device = device
        self.out_dir = Path(out_dir) if out_dir else None
        self.fid_fn = fid_fn
        self.opt_d = torch.optim.Adam(gan.discriminator.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.opt_g = torch.optim.Adam(gan.generator.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.bce = nn.BCEWithLogitsLoss()
        self.buffer = ReplayBuffer(run_cfg.replay_size, run_cfg.replay_prob, seed=cfg.seed)
        self.step_order: list[str] = []
        self.history = GanHistory()
        self.gen = torch.Generator().manual_seed(cfg.seed)

    def discriminator_step(
        self, real: torch.Tensor, labels: torch.Tensor
    ) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        """Um passo do discriminador: real -> alvo (com smoothing), falsa -> 0. Só D muda."""
        d, g = self.gan.discriminator, self.gan.generator
        n = len(real)
        fake_labels = torch.randint(0, self.gan.n_classes, (n,), generator=self.gen).to(self.device)
        fake = g(self.gan.noise(n, self.device, self.gen), fake_labels)
        d_fake_in, d_fake_lab = self.buffer.push_and_pop(fake.detach(), fake_labels)
        self.opt_d.zero_grad(set_to_none=True)
        out_real = d(real, labels)
        out_fake = d(d_fake_in, d_fake_lab)
        target_real = torch.full_like(out_real, self.run_cfg.label_smoothing)
        loss = self.bce(out_real, target_real) + self.bce(out_fake, torch.zeros_like(out_fake))
        loss.backward()
        self.opt_d.step()
        self.step_order.append("D")
        self.history.n_d_steps += 1
        stats = {
            "loss_d": float(loss.detach()),
            "d_real": float(torch.sigmoid(out_real).mean().detach()),
            "d_fake": float(torch.sigmoid(out_fake).mean().detach()),
        }
        return stats, fake, fake_labels

    def generator_step(self, fake: torch.Tensor, fake_labels: torch.Tensor) -> dict[str, float]:
        """Um passo do gerador: perda não saturante (D(G(z)) -> 1). Só G muda."""
        self.opt_g.zero_grad(set_to_none=True)
        out = self.gan.discriminator(fake, fake_labels)
        loss = self.bce(out, torch.ones_like(out))
        loss.backward()
        self.opt_g.step()
        self.gan.discriminator.zero_grad(set_to_none=True)
        self.step_order.append("G")
        self.history.n_g_steps += 1
        return {"loss_g": float(loss.detach())}

    def train_epoch(self, images: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Uma época: lotes embaralhados, D e G alternados. images em [0, 1], uint8 ou float."""
        self.gan.train()
        n = len(images)
        perm = torch.randperm(n, generator=self.gen)
        acc: dict[str, list[float]] = {"loss_d": [], "loss_g": [], "d_real": [], "d_fake": []}
        for i in range(0, n - self.cfg.batch_size + 1, self.cfg.batch_size):
            idx = perm[i : i + self.cfg.batch_size]
            real = images[idx].to(self.device).float()
            real = real / 255.0 if images.dtype == torch.uint8 else real
            real = real * 2 - 1
            lab = labels[idx].to(self.device)
            stats_d, fake, fake_lab = self.discriminator_step(real, lab)
            stats_g = self.generator_step(fake, fake_lab)
            for k, v in {**stats_d, **stats_g}.items():
                acc[k].append(v)
        return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}

    def save_grid(self, epoch: int, z_fixed: torch.Tensor, labels_fixed: torch.Tensor) -> Path:
        """Grade de amostras com z fixo: uma linha por classe."""
        assert self.out_dir is not None
        g = self.gan.generator
        was = g.training
        g.eval()
        with torch.no_grad():
            imgs = (g(z_fixed, labels_fixed) + 1) / 2
        g.train(was)
        path = self.out_dir / f"samples_epoch_{epoch:03d}.png"
        save_image(imgs.clamp(0, 1).cpu(), path, nrow=len(labels_fixed) // self.gan.n_classes)
        return path

    def fit(
        self, images: torch.Tensor, labels: torch.Tensor, epochs: int | None = None
    ) -> GanHistory:
        """Treina por `epochs`, grava grades, FID e checkpoints em out_dir."""
        epochs = epochs or self.cfg.epochs
        per_class = 8
        z_fixed = self.gan.noise(per_class * self.gan.n_classes, self.device, self.gen)
        labels_fixed = torch.arange(self.gan.n_classes).repeat_interleave(per_class).to(self.device)
        best = float("inf")
        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            stats = self.train_epoch(images, labels)
            for k in ("loss_d", "loss_g", "d_real", "d_fake"):
                getattr(self.history, k).append(stats[k])
            last = epoch == epochs
            if self.out_dir and (epoch % self.cfg.sample_every == 0 or epoch == 1 or last):
                self.save_grid(epoch, z_fixed, labels_fixed)
            if self.fid_fn and (epoch % self.cfg.fid_every == 0 or last):
                fid = {"epoch": epoch, **self.fid_fn(self.gan)}
                self.history.fid.append(fid)
                if self.out_dir and fid["fid"] < best:
                    best = fid["fid"]
                    torch.save(self.gan.generator.state_dict(), self.out_dir / "generator_best.pt")
                log.info(
                    "gan %s época %d: loss_d %.3f loss_g %.3f D(x) %.2f D(G(z)) %.2f FID %.1f",
                    self.run_cfg.name,
                    epoch,
                    stats["loss_d"],
                    stats["loss_g"],
                    stats["d_real"],
                    stats["d_fake"],
                    fid["fid"],
                )
        self.history.seconds = time.perf_counter() - start
        if self.out_dir:
            torch.save(self.gan.generator.state_dict(), self.out_dir / "generator_final.pt")
            (self.out_dir / "history.json").write_text(
                json.dumps(self.history.to_dict(), indent=2), encoding="utf-8"
            )
            self.plot_curves(self.out_dir / "curves.png")
        return self.history

    def plot_curves(self, path: Path) -> Path:
        """Curvas de perda, de D(x) e D(G(z)), e do FID."""
        h = self.history
        ep = np.arange(1, len(h.loss_d) + 1)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        ax[0].plot(ep, h.loss_d, label="discriminador")
        ax[0].plot(ep, h.loss_g, label="gerador")
        ax[0].set(title="Perda por época", xlabel="época")
        ax[0].legend()
        ax[1].plot(ep, h.d_real, label="D(x) real")
        ax[1].plot(ep, h.d_fake, label="D(G(z)) falsa")
        ax[1].axhline(0.5, color="gray", lw=0.8, ls="--")
        ax[1].set(title="Saída média do discriminador", xlabel="época", ylim=(0, 1))
        ax[1].legend()
        if h.fid:
            ax[2].plot([f["epoch"] for f in h.fid], [f["fid"] for f in h.fid], marker="o")
        ax[2].set(title="FID (menor é melhor)", xlabel="época")
        fig.suptitle(f"GAN condicional: {self.run_cfg.name}")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
