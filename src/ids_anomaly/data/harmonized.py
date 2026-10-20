"""A shared, hand-aligned feature space for UNSW-NB15 and CIC-IDS2017.

Cross-dataset evaluation (train on one capture, detect on another) needs features that *mean the
same thing* in both. The two datasets use different flow meters (Argus/Bro vs CICFlowMeter) and
different units, so this module maps both onto 18 physically comparable quantities: duration,
packet and byte counts per direction, rates, mean packet sizes, inter-arrival times, jitter and
protocol. Units are normalized (seconds, milliseconds, bytes) and a few ratio features are
derived identically from the totals, so no meter-specific definition leaks in.

This is deliberately a *lossy* view (NSL-KDD is excluded: its content/host-based features have no
counterpart in the other two). What it buys is the ability to ask the transfer question honestly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from ids_anomaly.data import cicids, unsw
from ids_anomaly.data.tabular import build_tabular_preprocessor

HARMONIZED_NUMERIC: list[str] = [
    "dur_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "pkts_per_s", "bytes_per_s",
    "fwd_mean_len", "bwd_mean_len", "fwd_iat_ms", "bwd_iat_ms", "fwd_jitter_ms", "bwd_jitter_ms",
    "fwd_pkt_frac", "fwd_byte_frac",
]  # fmt: skip
HARMONIZED_CATEGORICAL: list[str] = ["proto_group"]
HARMONIZED_COLUMNS = HARMONIZED_NUMERIC + HARMONIZED_CATEGORICAL

_US_PER_MS = 1_000.0
_US_PER_S = 1_000_000.0


@dataclass
class RawSplit:
    """Harmonized, *unscaled* features plus evaluation-only labels."""

    features: pd.DataFrame
    is_attack: np.ndarray
    attack_category: np.ndarray
    group: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.features)


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    total_pkts = frame["fwd_pkts"] + frame["bwd_pkts"]
    total_bytes = frame["fwd_bytes"] + frame["bwd_bytes"]
    dur = frame["dur_s"].clip(lower=1e-6)
    frame["pkts_per_s"] = total_pkts / dur
    frame["bytes_per_s"] = total_bytes / dur
    frame["fwd_pkt_frac"] = frame["fwd_pkts"] / total_pkts.clip(lower=1)
    frame["fwd_byte_frac"] = frame["fwd_bytes"] / total_bytes.clip(lower=1)
    return frame


def _proto_group(series: pd.Series) -> pd.Series:
    lowered = series.astype(str).str.lower()
    return lowered.where(lowered.isin(["tcp", "udp"]), "other")


def harmonize_unsw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "dur_s": df["dur"].astype("float64"),
            "fwd_pkts": df["spkts"].astype("float64"),
            "bwd_pkts": df["dpkts"].astype("float64"),
            "fwd_bytes": df["sbytes"].astype("float64"),
            "bwd_bytes": df["dbytes"].astype("float64"),
            "fwd_mean_len": df["smean"].astype("float64"),
            "bwd_mean_len": df["dmean"].astype("float64"),
            "fwd_iat_ms": df["sinpkt"].astype("float64"),
            "bwd_iat_ms": df["dinpkt"].astype("float64"),
            "fwd_jitter_ms": df["sjit"].astype("float64"),
            "bwd_jitter_ms": df["djit"].astype("float64"),
            "proto_group": _proto_group(df["proto"]),
        }
    )
    return _derived(out)[HARMONIZED_COLUMNS]


def harmonize_cicids(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "dur_s": df["Flow Duration"] / _US_PER_S,
            "fwd_pkts": df["Total Fwd Packets"],
            "bwd_pkts": df["Total Backward Packets"],
            "fwd_bytes": df["Total Length of Fwd Packets"],
            "bwd_bytes": df["Total Length of Bwd Packets"],
            "fwd_mean_len": df["Fwd Packet Length Mean"],
            "bwd_mean_len": df["Bwd Packet Length Mean"],
            "fwd_iat_ms": df["Fwd IAT Mean"] / _US_PER_MS,
            "bwd_iat_ms": df["Bwd IAT Mean"] / _US_PER_MS,
            "fwd_jitter_ms": df["Fwd IAT Std"] / _US_PER_MS,
            "bwd_jitter_ms": df["Bwd IAT Std"] / _US_PER_MS,
            "proto_group": _proto_group(df["protocol"]),
        }
    )
    return _derived(out)[HARMONIZED_COLUMNS]


def load_harmonized(name: str, raw_dir: Path) -> tuple[RawSplit, RawSplit]:
    """Harmonized (train, test) splits for ``"unsw_nb15"`` or ``"cicids2017"``."""
    if name == "unsw_nb15":
        train_df, test_df = unsw._read(raw_dir)
        return (
            RawSplit(
                harmonize_unsw(train_df),
                train_df["label"].to_numpy() == 1,
                train_df["attack_cat"].astype(str).str.lower().to_numpy(),
            ),
            RawSplit(
                harmonize_unsw(test_df),
                test_df["label"].to_numpy() == 1,
                test_df["attack_cat"].astype(str).str.lower().to_numpy(),
            ),
        )
    if name == "cicids2017":
        train_df, test_df, _ = cicids.load_cicids_frames(raw_dir)
        return tuple(  # type: ignore[return-value]
            RawSplit(
                harmonize_cicids(frame),
                (frame["category"] != "normal").to_numpy(),
                frame["category"].astype(str).to_numpy(),
                frame["day"].astype(str).to_numpy(),
            )
            for frame in (train_df, test_df)
        )
    raise ValueError(f"no harmonized view for dataset {name!r}")


def fit_scaler(reference: RawSplit) -> ColumnTransformer:
    """Fit the harmonized preprocessor on ``reference`` (source-train, or target-pool for the
    target-side-standardization variant of the transfer experiment)."""
    pre = build_tabular_preprocessor(HARMONIZED_NUMERIC, HARMONIZED_CATEGORICAL, min_frequency=None)
    pre.fit(reference.features)
    return pre


def transform(split: RawSplit, pre: ColumnTransformer) -> np.ndarray:
    return np.asarray(pre.transform(split.features), dtype=np.float32)
