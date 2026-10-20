"""Neural detectors behind the common interface.

Autoencoder and Deep SVDD are the reconstruction / one-class baselines; the VAE, the RealNVP
normalizing flow, the NeuTraL-style contrastive model and the masked tabular transformer are the
"current" families a 2026 IDS paper is expected to compare against. All are small on purpose
(CPU-only, bounded by ``max_train_rows`` / ``max_seconds``), and all train **without labels**
with early stopping on held-out *training* loss.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from ids_anomaly.anomaly.deep_svdd import DeepSVDDConfig, train_deep_svdd
from ids_anomaly.anomaly.deep_svdd import anomaly_score as svdd_score
from ids_anomaly.detectors.base import Detector, batched_apply, fit_torch


def _mlp(dims: list[int], final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ------------------------------------------------------------------------------ autoencoder


class _AE(nn.Module):
    def __init__(self, d: int, latent: int):
        super().__init__()
        self.enc = _mlp([d, 128, 64, latent])
        self.dec = _mlp([latent, 64, 128, d])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class AutoencoderDetector(Detector):
    name = "ae"
    family = "reconstruction"
    assumes_clean_training = True

    def __init__(self, seed: int = 42, epochs: int = 60, max_seconds: float = 90.0):
        super().__init__(seed)
        self.epochs, self.max_seconds = epochs, max_seconds

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        d = X.shape[1]
        torch.manual_seed(self.seed)
        self.model = _AE(d, latent=max(4, d // 8))
        self.history = fit_torch(
            self.model, lambda b: F.mse_loss(self.model(b), b), X,
            seed=self.seed, epochs=self.epochs, max_seconds=self.max_seconds,
        )  # fmt: skip
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return batched_apply(lambda b: ((self.model(b) - b) ** 2).mean(dim=1), X)

    def feature_errors(self, X: np.ndarray) -> np.ndarray:
        """Per-feature squared reconstruction error, shape ``(n, d)``: exact attribution."""
        out: list[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, len(X), 8192):
                b = torch.from_numpy(np.ascontiguousarray(X[s : s + 8192], dtype=np.float32))
                out.append(((self.model(b) - b) ** 2).numpy())
        return np.concatenate(out)


# ------------------------------------------------------------------------------- deep SVDD


class DeepSVDDDetector(Detector):
    name = "deep_svdd"
    family = "deep-one-class"
    assumes_clean_training = True

    def __init__(self, seed: int = 42, max_train: int = 60_000):
        super().__init__(seed)
        self.max_train = max_train

    def fit(self, X: np.ndarray) -> DeepSVDDDetector:
        if len(X) > self.max_train:
            X = X[np.random.default_rng(self.seed).choice(len(X), size=self.max_train, replace=False)]
        cfg = DeepSVDDConfig(pretrain_epochs=10, train_epochs=30, random_state=self.seed)
        self.result = train_deep_svdd(X, cfg)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return svdd_score(self.result, X)


# ------------------------------------------------------------------------------------- VAE


class _VAE(nn.Module):
    def __init__(self, d: int, latent: int):
        super().__init__()
        self.enc = _mlp([d, 128, 64], final_activation=True)
        self.mu = nn.Linear(64, latent)
        self.logvar = nn.Linear(64, latent)
        self.dec = _mlp([latent, 64, 128, d])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 6)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.dec(z), mu, logvar


class VAEDetector(Detector):
    """Beta-VAE; score = negative ELBO evaluated at the posterior mean (recon + beta*KL)."""

    name = "vae"
    family = "reconstruction"
    assumes_clean_training = True

    def __init__(self, seed: int = 42, beta: float = 0.5, epochs: int = 60, max_seconds: float = 90.0):
        super().__init__(seed)
        self.beta, self.epochs, self.max_seconds = beta, epochs, max_seconds

    def _neg_elbo(self, x: torch.Tensor) -> torch.Tensor:
        recon, mu, logvar = self.model(x)
        rec = ((recon - x) ** 2).sum(dim=1)
        kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1)
        return rec + self.beta * kl

    def fit(self, X: np.ndarray) -> VAEDetector:
        d = X.shape[1]
        torch.manual_seed(self.seed)
        self.model = _VAE(d, latent=max(4, d // 8))
        self.history = fit_torch(
            self.model, lambda b: self._neg_elbo(b).mean() / d, X,
            seed=self.seed, epochs=self.epochs, max_seconds=self.max_seconds,
        )  # fmt: skip
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        return batched_apply(self._neg_elbo, X)


# ----------------------------------------------------------------- normalizing flow (RealNVP)


class _Coupling(nn.Module):
    mask: torch.Tensor

    def __init__(self, d: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = _mlp([d, hidden, hidden, 2 * d])
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)  # start as the identity map
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = x * self.mask
        s, t = self.net(masked).chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        y = masked + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=1)


class _RealNVP(nn.Module):
    def __init__(self, d: int, n_layers: int, hidden: int, seed: int):
        super().__init__()
        rng = np.random.default_rng(seed)
        layers = []
        for i in range(n_layers):
            if i % 2 == 0:
                perm = rng.permutation(d)
            mask = torch.zeros(d)
            mask[torch.from_numpy(perm[: d // 2] if i % 2 == 0 else perm[d // 2 :].copy())] = 1.0
            layers.append(_Coupling(d, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        logdet = torch.zeros(len(x), device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        base = -0.5 * (x**2 + math.log(2 * math.pi)).sum(dim=1)
        return base + logdet


class FlowDetector(Detector):
    """RealNVP density model; score = negative log-likelihood. Training inputs get small
    Gaussian noise (dequantization) because one-hot columns otherwise let the flow spike its
    density on a measure-zero set."""

    name = "flow"
    family = "density"

    def __init__(self, seed: int = 42, n_layers: int = 6, hidden: int = 128, noise: float = 0.05,
                 epochs: int = 40, max_seconds: float = 90.0):  # fmt: skip
        super().__init__(seed)
        self.n_layers, self.hidden, self.noise = n_layers, hidden, noise
        self.epochs, self.max_seconds = epochs, max_seconds

    def fit(self, X: np.ndarray) -> FlowDetector:
        d = X.shape[1]
        torch.manual_seed(self.seed)
        self.model = _RealNVP(d, self.n_layers, self.hidden, self.seed)

        def loss(b: torch.Tensor) -> torch.Tensor:
            noisy = b + self.noise * torch.randn_like(b) if self.model.training else b
            return -self.model.log_prob(noisy).mean() / d

        self.history = fit_torch(
            self.model, loss, X, seed=self.seed, epochs=self.epochs, max_seconds=self.max_seconds, lr=5e-4
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        return batched_apply(lambda b: -self.model.log_prob(b), X)


# ------------------------------------------------------------ NeuTraL-style contrastive model


class _NeuTraL(nn.Module):
    def __init__(self, d: int, k: int, hidden: int, out: int):
        super().__init__()
        self.k = k
        self.transforms = nn.ModuleList([_mlp([d, hidden, d]) for _ in range(k)])
        for t in self.transforms:  # small init: T_k(x) ~ x at the start
            assert isinstance(t, nn.Sequential)
            last = t[-1]
            assert isinstance(last, nn.Linear)
            nn.init.normal_(last.weight, std=0.01)
            nn.init.zeros_(last.bias)
        self.encoder = _mlp([d, hidden, hidden, out])

    def loss_per_sample(self, x: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
        views = [x] + [x + t(x) for t in self.transforms]  # residual transforms
        n = len(x)
        h = F.normalize(self.encoder(torch.cat(views, dim=0)), dim=1).view(self.k + 1, n, -1)
        ref = h[0]
        pos = torch.exp((h[1:] * ref.unsqueeze(0)).sum(-1) / tau)  # (k, n)
        sims = torch.einsum("anf,bnf->abn", h[1:], h[1:]) / tau  # (k, k, n)
        eye = torch.eye(self.k, device=x.device).unsqueeze(-1)
        neg = (torch.exp(sims) * (1 - eye)).sum(dim=1)  # (k, n)
        return -torch.log(pos / (pos + neg)).mean(dim=0)


class NeuTraLDetector(Detector):
    """Neural transformation learning (Qiu et al., 2021), tabular variant.

    K learned residual transformations of a sample must be mutually distinguishable yet each
    stay close to the original in a shared embedding; normal data can satisfy this, anomalies
    cannot. Score = the contrastive loss of the sample itself.
    """

    name = "neutral"
    family = "self-supervised"
    assumes_clean_training = True

    def __init__(self, seed: int = 42, k: int = 6, epochs: int = 40, max_seconds: float = 120.0):
        super().__init__(seed)
        self.k, self.epochs, self.max_seconds = k, epochs, max_seconds

    def fit(self, X: np.ndarray) -> NeuTraLDetector:
        d = X.shape[1]
        torch.manual_seed(self.seed)
        self.model = _NeuTraL(d, self.k, hidden=128, out=32)
        self.history = fit_torch(
            self.model, lambda b: self.model.loss_per_sample(b).mean(), X,
            seed=self.seed, epochs=self.epochs, max_seconds=self.max_seconds, batch_size=512, max_train_rows=60_000,
        )  # fmt: skip
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        return batched_apply(self.model.loss_per_sample, X, batch_size=4096)


# ------------------------------------------------------------------ masked tabular transformer


class _MaskedTabTransformer(nn.Module):
    """FT-Transformer-style encoder trained to reconstruct masked feature values."""

    def __init__(self, d: int, d_model: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.d = d
        self.weight = nn.Parameter(torch.randn(d, d_model) * 0.1)
        self.bias = nn.Parameter(torch.zeros(d, d_model))
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.1)
        block = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head_w = nn.Parameter(torch.randn(d, d_model) * 0.1)
        self.head_b = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask`` is 1 where a feature is hidden. Returns reconstructions for every feature."""
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        hidden_tok = self.mask_token + self.bias
        tokens = torch.where(mask.unsqueeze(-1) > 0, hidden_tok.expand_as(tokens), tokens)
        h = self.encoder(tokens)
        return (h * self.head_w).sum(-1) + self.head_b


class MaskedTransformerDetector(Detector):
    """Masked-feature reconstruction with a transformer. Score = mean squared error of each
    feature when it is hidden (features are hidden in ``groups`` disjoint interleaved groups at
    test time, so every feature is predicted from the others exactly once)."""

    name = "mtt"
    family = "self-supervised"
    assumes_clean_training = True

    def __init__(self, seed: int = 42, mask_ratio: float = 0.3, groups: int = 4,
                 epochs: int = 12, max_seconds: float = 150.0):  # fmt: skip
        super().__init__(seed)
        self.mask_ratio, self.groups = mask_ratio, groups
        self.epochs, self.max_seconds = epochs, max_seconds

    def fit(self, X: np.ndarray) -> MaskedTransformerDetector:
        d = X.shape[1]
        torch.manual_seed(self.seed)
        self.model = _MaskedTabTransformer(d)
        self._group_of = torch.from_numpy(np.random.default_rng(self.seed).permutation(d) % self.groups)

        def loss(b: torch.Tensor) -> torch.Tensor:
            if self.model.training:
                mask = (torch.rand(b.shape) < self.mask_ratio).float()
                mask[torch.arange(len(b)), torch.randint(0, d, (len(b),))] = 1.0  # >= 1 masked
                err = ((self.model(b, mask) - b) ** 2 * mask).sum() / mask.sum()
                return err
            return self._masked_error(b).mean()

        self.history = fit_torch(
            self.model, loss, X, seed=self.seed, epochs=self.epochs, max_seconds=self.max_seconds,
            batch_size=512, max_train_rows=60_000, lr=2e-3,
        )  # fmt: skip
        return self

    def _masked_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature squared error, each feature predicted while hidden: shape ``(n, d)``."""
        err = torch.zeros_like(x)
        for g in range(self.groups):
            mask = (self._group_of == g).float().expand_as(x)
            err = err + (self.model(x, mask) - x) ** 2 * mask
        return err

    def _masked_error(self, x: torch.Tensor) -> torch.Tensor:
        return self._masked_errors(x).mean(dim=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        return batched_apply(self._masked_error, X, batch_size=2048)

    def feature_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, len(X), 2048):
                b = torch.from_numpy(np.ascontiguousarray(X[s : s + 2048], dtype=np.float32))
                out.append(self._masked_errors(b).numpy())
        return np.concatenate(out)
