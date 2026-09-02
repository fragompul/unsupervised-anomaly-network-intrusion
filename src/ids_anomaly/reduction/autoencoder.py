"""A compact MLP autoencoder used both as a learned embedding and as a reconstruction-error
anomaly score. Sized for CPU-only training: a handful of narrow linear layers, not conv/attention
stacks the laptop has no business running at this data scale (122 tabular features).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Use CUDA if available, else CPU. No MPS branch: this project targets Windows/CPU-CI."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 3, hidden_dims: tuple[int, ...] = (64, 16)):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        dims = [n_features, *hidden_dims, latent_dim]
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        encoder_layers.pop()  # no activation on the latent bottleneck
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        rev_dims = list(reversed(dims))
        for in_dim, out_dim in zip(rev_dims[:-1], rev_dims[1:], strict=True):
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        decoder_layers.pop()  # linear output head, features are standardized (can be negative)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


@dataclass
class TrainConfig:
    latent_dim: int = 3
    hidden_dims: tuple[int, ...] = (64, 16)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 8
    val_fraction: float = 0.1
    random_state: int = 42


@dataclass
class TrainResult:
    model: Autoencoder
    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    train_seconds: float


def train_autoencoder(
    X: np.ndarray,
    config: TrainConfig,
    checkpoint_path: Path | None = None,
) -> TrainResult:
    """Train with early stopping on a held-out validation slice and checkpoint the best weights.

    Checkpointing lets a multi-hour CPU run be killed and resumed without losing progress: if
    ``checkpoint_path`` exists it is used only to *report* the best state for this call, and the
    file is overwritten every time validation loss improves so an interrupted run always leaves
    the best-so-far weights on disk rather than the latest (possibly overfit) ones.
    """
    torch.manual_seed(config.random_state)
    device = get_device()

    rng = np.random.default_rng(config.random_state)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(n * config.val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(X_train), batch_size=config.batch_size, shuffle=True, drop_last=False
    )

    model = Autoencoder(X.shape[1], config.latent_dim, config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    start = time.perf_counter()

    for epoch in range(config.max_epochs):
        model.train()
        running_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, x_hat = model(batch)
            loss = loss_fn(x_hat, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        train_loss = running_loss / len(train_idx)

        model.eval()
        with torch.no_grad():
            _, val_hat = model(X_val)
            val_loss = loss_fn(val_hat, X_val).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"state_dict": best_state, "config": config, "epoch": epoch},
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_seconds=time.perf_counter() - start,
    )


def encode(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z, _ = model(torch.tensor(X, dtype=torch.float32, device=device))
    return z.cpu().numpy()


def reconstruction_error(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """Per-sample MSE reconstruction error, used both as a scalar anomaly score and diagnostic."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        _, x_hat = model(torch.tensor(X, dtype=torch.float32, device=device))
        err = ((x_hat.cpu().numpy() - X) ** 2).mean(axis=1)
    return err
