"""Deep SVDD (Ruff et al., 2018): learn a feature map that pulls normal data into a minimum-
volume hypersphere; distance from the learned center is the anomaly score.

Follows the paper's two safeguards against the trivial "collapse to a constant" solution:
bias-free linear layers (so the network cannot simply learn the constant function ``f(x) = c``
by zeroing its weights and using a bias term) and L2 weight decay on the optimizer (which
otherwise permits the *nearly*-trivial solution of shrinking all weights toward zero). The
encoder is pretrained as a bias-free autoencoder first and its encoder weights reused to
initialize the SVDD network, as recommended in the original paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ids_anomaly.reduction.autoencoder import get_device


class _BiasFreeEncoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        dims = [n_features, *hidden_dims, latent_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(in_dim, out_dim, bias=False), nn.ReLU()]
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BiasFreeAutoencoder(nn.Module):
    """Pretraining scaffold: bias-free encoder (kept for weight transfer) + biased decoder."""

    def __init__(self, n_features: int, latent_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        self.encoder = _BiasFreeEncoder(n_features, latent_dim, hidden_dims)
        dims = list(reversed([n_features, *hidden_dims, latent_dim]))
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers.pop()
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


@dataclass
class DeepSVDDConfig:
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (64, 16)
    pretrain_epochs: int = 20
    train_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 256
    random_state: int = 42


@dataclass
class DeepSVDDResult:
    encoder: _BiasFreeEncoder
    center: torch.Tensor
    history: list[dict[str, float]]
    train_seconds: float


def _pretrain(X: torch.Tensor, config: DeepSVDDConfig, device: torch.device) -> _BiasFreeEncoder:
    ae = _BiasFreeAutoencoder(X.shape[1], config.latent_dim, config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=config.lr)
    loader = DataLoader(TensorDataset(X), batch_size=config.batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    for _ in range(config.pretrain_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, x_hat = ae(batch)
            loss = loss_fn(x_hat, batch)
            loss.backward()
            optimizer.step()
    return ae.encoder


def _init_center(encoder: _BiasFreeEncoder, X: torch.Tensor, device: torch.device, eps: float = 0.1) -> torch.Tensor:
    encoder.eval()
    with torch.no_grad():
        z = encoder(X.to(device))
    center = z.mean(dim=0)
    # clamp near-zero dims away from the origin: an exact-zero center is the one point
    # a bias-free network can trivially reach for *any* input, which would make every
    # sample look equally "normal" regardless of the weights learned.
    center = torch.where(center.abs() < eps, torch.sign(center) * eps + eps, center)
    return center


def train_deep_svdd(X: np.ndarray, config: DeepSVDDConfig) -> DeepSVDDResult:
    torch.manual_seed(config.random_state)
    device = get_device()
    X_t = torch.tensor(X, dtype=torch.float32)

    start = time.perf_counter()
    encoder = _pretrain(X_t, config, device).to(device)
    center = _init_center(encoder, X_t, device)

    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loader = DataLoader(TensorDataset(X_t), batch_size=config.batch_size, shuffle=True)
    history: list[dict[str, float]] = []

    encoder.train()
    for epoch in range(config.train_epochs):
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = encoder(batch)
            loss = ((z - center) ** 2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)
        history.append({"epoch": epoch, "svdd_loss": running / X.shape[0]})

    return DeepSVDDResult(
        encoder=encoder,
        center=center.detach(),
        history=history,
        train_seconds=time.perf_counter() - start,
    )


def anomaly_score(result: DeepSVDDResult, X: np.ndarray) -> np.ndarray:
    """Squared distance to the learned hypersphere center -- higher means more anomalous."""
    device = next(result.encoder.parameters()).device
    result.encoder.eval()
    with torch.no_grad():
        z = result.encoder(torch.tensor(X, dtype=torch.float32, device=device))
        dist = ((z - result.center) ** 2).sum(dim=1)
    return dist.cpu().numpy()
