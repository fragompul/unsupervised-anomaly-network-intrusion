"""Classical and library-backed detectors behind the common interface.

Fixed, documented defaults across datasets (no per-dataset tuning, so the benchmark uses zero
labels anywhere): tuning on labels would make "unsupervised" a claim about the model rather than
about the pipeline.
"""

from __future__ import annotations

import numpy as np
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from ids_anomaly.detectors.base import Detector


def _subsample(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(X) <= n:
        return X
    return X[np.random.default_rng(seed).choice(len(X), size=n, replace=False)]


class IsolationForestDetector(Detector):
    name = "iforest"
    family = "tree"

    def __init__(self, seed: int = 42, n_estimators: int = 200, max_samples: int | str = 256):
        super().__init__(seed)
        self.model = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=4
        )

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class OneClassSVMDetector(Detector):
    name = "ocsvm"
    family = "kernel"
    assumes_clean_training = True

    def __init__(self, seed: int = 42, nu: float = 0.05, max_train: int = 8000):
        super().__init__(seed)
        self.nu, self.max_train = nu, max_train

    def fit(self, X: np.ndarray) -> OneClassSVMDetector:
        self.model = OneClassSVM(nu=self.nu, kernel="rbf", gamma="scale")
        self.model.fit(_subsample(X, self.max_train, self.seed))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)


class PCAReconstructionDetector(Detector):
    name = "pca"
    family = "reconstruction"

    def __init__(self, seed: int = 42, variance: float = 0.95):
        super().__init__(seed)
        self.variance = variance

    def fit(self, X: np.ndarray) -> PCAReconstructionDetector:
        self.model = PCA(n_components=self.variance, svd_solver="full", random_state=self.seed).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        recon = self.model.inverse_transform(self.model.transform(X))
        return ((X - recon) ** 2).mean(axis=1)


class GMMDetector(Detector):
    name = "gmm"
    family = "density"

    def __init__(self, seed: int = 42, n_components: int = 8, max_train: int = 60_000):
        super().__init__(seed)
        self.n_components, self.max_train = n_components, max_train

    def fit(self, X: np.ndarray) -> GMMDetector:
        self.model = GaussianMixture(
            n_components=self.n_components, covariance_type="full", reg_covar=1e-3,
            random_state=self.seed, max_iter=100,
        ).fit(_subsample(X, self.max_train, self.seed))  # fmt: skip
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class KNNDetector(Detector):
    """Mean distance to the k nearest *training* points (exact search on a bounded reference set)."""

    name = "knn"
    family = "distance"

    def __init__(self, seed: int = 42, k: int = 10, max_reference: int = 30_000):
        super().__init__(seed)
        self.k, self.max_reference = k, max_reference

    def fit(self, X: np.ndarray) -> KNNDetector:
        self.nn = NearestNeighbors(n_neighbors=self.k, n_jobs=4).fit(_subsample(X, self.max_reference, self.seed))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        dist, _ = self.nn.kneighbors(X)
        return dist.mean(axis=1)


class LOFDetector(Detector):
    name = "lof"
    family = "distance"

    def __init__(self, seed: int = 42, n_neighbors: int = 20, max_reference: int = 30_000):
        super().__init__(seed)
        self.n_neighbors, self.max_reference = n_neighbors, max_reference

    def fit(self, X: np.ndarray) -> LOFDetector:
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True, n_jobs=4)
        self.model.fit(_subsample(X, self.max_reference, self.seed))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


class _PyODDetector(Detector):
    """Adapter for PyOD's fit/decision_function API (higher decision_function = more anomalous)."""

    def _make(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: np.ndarray) -> _PyODDetector:
        self.model = self._make()
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.decision_function(X), dtype=np.float64)


class ECODDetector(_PyODDetector):
    """ECOD (Li et al., 2022): parameter-free, empirical-CDF tail probabilities per feature."""

    name = "ecod"
    family = "density"

    def _make(self):
        return ECOD()


class COPODDetector(_PyODDetector):
    """COPOD (Li et al., 2020): copula-based tail probabilities, parameter-free."""

    name = "copod"
    family = "density"

    def _make(self):
        return COPOD()


class HBOSDetector(_PyODDetector):
    """HBOS (Goldstein & Dengel, 2012): histogram-based, linear time."""

    name = "hbos"
    family = "density"

    def _make(self):
        return HBOS(n_bins=20)
