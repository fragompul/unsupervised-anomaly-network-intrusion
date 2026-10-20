"""Common detector interface and the label-free training loop shared by every neural model.

Contract: ``fit(X)`` never sees labels; ``score(X)`` returns one float per row where **higher
means more anomalous**. That single convention is what lets the benchmark, the conformal layer
and the web app treat a Mahalanobis-style distance, an isolation depth and a flow's negative
log-likelihood interchangeably.
"""

from __future__ import annotations

import copy
import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import torch
from torch import nn

# Leave cores free: the benchmark runner parallelizes across cells, not inside torch.
TORCH_THREADS = int(os.environ.get("NOCTUA_TORCH_THREADS", "4"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Detector(ABC):
    """Unsupervised anomaly detector: ``fit`` on unlabeled rows, ``score`` -> higher = more anomalous."""

    name: ClassVar[str]
    family: ClassVar[str]  # tree | kernel | density | distance | reconstruction | deep-one-class | ...
    #: Family-level assumption used in the paper: does the method presume a clean training set?
    assumes_clean_training: ClassVar[bool] = False

    def __init__(self, seed: int = 42):
        self.seed = seed

    @abstractmethod
    def fit(self, X: np.ndarray) -> Detector: ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    def fit_score_seconds(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Fit then score, returning ``(scores, fit_seconds, score_seconds)`` for cost profiling."""
        t0 = time.perf_counter()
        self.fit(X_train)
        t1 = time.perf_counter()
        scores = self.score(X_test)
        t2 = time.perf_counter()
        return scores, t1 - t0, t2 - t1


def batched_apply(
    fn: Callable[[torch.Tensor], torch.Tensor], X: np.ndarray, batch_size: int = 8192
) -> np.ndarray:
    """Run ``fn`` on ``X`` in no-grad batches; concatenates the 1-D per-row outputs."""
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(X[start : start + batch_size], dtype=np.float32))
            out.append(fn(chunk).detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def fit_torch(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    X: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.1,
    patience: int = 5,
    max_train_rows: int | None = 100_000,
    max_seconds: float | None = None,
) -> list[dict[str, float]]:
    """Mini-batch Adam with early stopping on **held-out training loss** (no labels involved).

    ``loss_fn`` maps a batch to a scalar loss. The best-validation weights are restored. Rows
    beyond ``max_train_rows`` are randomly dropped so wall-clock stays bounded on CIC-IDS2017;
    ``max_seconds`` is a hard budget checked once per epoch.
    """
    seed_everything(seed)
    torch.set_num_threads(TORCH_THREADS)
    rng = np.random.default_rng(seed)
    if max_train_rows is not None and len(X) > max_train_rows:
        X = X[rng.choice(len(X), size=max_train_rows, replace=False)]
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_fraction))
    X_val = torch.from_numpy(np.ascontiguousarray(X[perm[:n_val]], dtype=np.float32))
    X_tr = torch.from_numpy(np.ascontiguousarray(X[perm[n_val:]], dtype=np.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best, best_state, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    started = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X_tr), generator=gen)
        total = 0.0
        for start in range(0, len(X_tr), batch_size):
            batch = X_tr[order[start : start + batch_size]]
            opt.zero_grad()
            loss = loss_fn(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(batch)
        model.eval()
        with torch.no_grad():
            val = float(np.mean([float(loss_fn(X_val[i : i + 8192]).item()) for i in range(0, len(X_val), 8192)]))
        history.append({"epoch": float(epoch), "train_loss": total / len(X_tr), "val_loss": val})
        if val < best - 1e-6:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
        if max_seconds is not None and time.perf_counter() - started > max_seconds:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history
