"""CnnClassifier: EfficientNet-B0 com o corpo congelado e uma cabeça nova."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CnnClassifier(nn.Module):
    """Normaliza a entrada, passa pelo corpo congelado e classifica com uma cabeça nova.

    Entrada: tensor RGB em [0, 1]. A média e o desvio do ImageNet ficam dentro do modelo,
    então treino, avaliação e inferência usam a mesma transformação.
    """

    def __init__(
        self, num_classes: int, weights: str | None = "IMAGENET1K_V1", dropout: float = 0.2
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"precisa de pelo menos 2 classes, veio {num_classes}")
        enum = EfficientNet_B0_Weights[weights] if weights else None
        self.net = efficientnet_b0(weights=enum)
        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes)
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self._freeze_body()

    @property
    def head(self) -> nn.Linear:
        """A camada linear nova (a única treinável)."""
        return self.net.classifier[1]

    def _freeze_body(self) -> None:
        for name, param in self.net.named_parameters():
            param.requires_grad = name.startswith("classifier.")
        self.net.features.eval()

    def train(self, mode: bool = True) -> CnnClassifier:
        """Liga o modo de treino, mas mantém o corpo em eval: a BatchNorm congelada não se mexe."""
        super().train(mode)
        self.net.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devolve os logits (lote, num_classes). O corpo roda sem gradiente."""
        x = (x - self.mean) / self.std
        with torch.no_grad():
            feats = self.net.avgpool(self.net.features(x)).flatten(1)
        return self.net.classifier(feats)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Os pesos que o treino muda (só a cabeça), copiados para a CPU."""
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in trainable}

    def param_counts(self) -> dict[str, int]:
        """Parâmetros totais, treináveis e congelados, e o tamanho da cabeça."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "head_in_features": self.head.in_features,
            "head_out_features": self.head.out_features,
        }
