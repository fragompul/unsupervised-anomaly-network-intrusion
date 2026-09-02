"""One-Class SVM: kernel-based boundary around the dense (normal) region of feature space.

Trained on a bounded subsample -- its O(n^2)-O(n^3) kernel computation makes the full ~113k-row
normal-only training set impractical on CPU, so ``max_train_samples`` caps it, sized generously
enough (default 8000) that the fitted boundary is still representative.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


def fit_one_class_svm(
    X: np.ndarray,
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    max_train_samples: int = 8000,
    random_state: int = 42,
) -> OneClassSVM:
    if X.shape[0] > max_train_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X.shape[0], size=max_train_samples, replace=False)
        X = X[idx]
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model


def anomaly_score(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (flip sklearn's decision_function, where negative = outlier)."""
    return -model.decision_function(X)
