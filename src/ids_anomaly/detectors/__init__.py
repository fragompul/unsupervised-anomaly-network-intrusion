"""Detector zoo: one interface, fifteen unsupervised anomaly detectors."""

from __future__ import annotations

from ids_anomaly.detectors.base import Detector
from ids_anomaly.detectors.classical import (
    COPODDetector,
    ECODDetector,
    GMMDetector,
    HBOSDetector,
    IsolationForestDetector,
    KNNDetector,
    LOFDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
)
from ids_anomaly.detectors.deep import (
    AutoencoderDetector,
    DeepSVDDDetector,
    FlowDetector,
    MaskedTransformerDetector,
    NeuTraLDetector,
    VAEDetector,
)

REGISTRY: dict[str, type[Detector]] = {
    cls.name: cls
    for cls in (
        IsolationForestDetector, OneClassSVMDetector, PCAReconstructionDetector, GMMDetector,
        KNNDetector, LOFDetector, ECODDetector, COPODDetector, HBOSDetector,
        AutoencoderDetector, DeepSVDDDetector, VAEDetector, FlowDetector, NeuTraLDetector,
        MaskedTransformerDetector,
    )
}  # fmt: skip

#: Display names for tables and the web app.
TITLES: dict[str, str] = {
    "iforest": "Isolation Forest", "ocsvm": "One-Class SVM", "pca": "PCA reconstruction",
    "gmm": "Gaussian mixture", "knn": "kNN distance", "lof": "Local Outlier Factor",
    "ecod": "ECOD", "copod": "COPOD", "hbos": "HBOS", "ae": "Autoencoder",
    "deep_svdd": "Deep SVDD", "vae": "β-VAE", "flow": "RealNVP flow",
    "neutral": "NeuTraL AD", "mtt": "Masked Tab-Transformer",
}  # fmt: skip


def build(name: str, seed: int = 42, **kwargs) -> Detector:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](seed=seed, **kwargs)


__all__ = ["REGISTRY", "TITLES", "Detector", "build"]
