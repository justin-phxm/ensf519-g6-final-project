from __future__ import annotations

import torch
from torch import nn


class MaskedFeatureMLP(nn.Module):
    """Classify masked features while exposing which values are unavailable."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim * 2
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = x.masked_fill(mask.bool(), 0.0)
        return self.network(torch.cat([x_masked, mask.to(x.dtype)], dim=1)).squeeze(1)


class BaselineMLP(nn.Module):
    """MLP baseline without a mask-indicator channel or masking augmentation."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = x.masked_fill(mask.bool(), 0.0)
        return self.network(x_masked).squeeze(1)


class DenoisingAutoencoderClassifier(nn.Module):
    """Reconstruct masked inputs, then classify the completed feature vector."""

    def __init__(self, input_dim: int, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x.masked_fill(mask.bool(), 0.0)
        encoded = self.encoder(torch.cat([x_masked, mask.to(x.dtype)], dim=1))
        reconstruction = self.decoder(encoded)
        completed = torch.where(mask.bool(), reconstruction, x_masked)
        logits = self.classifier(completed).squeeze(1)
        return logits, reconstruction
