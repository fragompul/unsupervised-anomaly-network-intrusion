"""CIC-IDS2017 loader (Sharafaldin et al., 2018) with a temporal protocol.

CIC-IDS2017 is five working days of captured traffic with 78 CICFlowMeter flow statistics and
timestamps. That timestamp is what makes it the right dataset for drift: Monday is *attack-free*
(a naturally clean normal-only training window), and each later day introduces new attack
families and shifts the benign mix.

Default protocol (``split="temporal"``): train on Mon+Tue (clean benign + FTP/SSH-Patator),
test on Wed-Fri (DoS, Heartbleed, web attacks, infiltration, botnet, port scan, DDoS), so almost
every test attack family is *novel* to the training window. ``split="iid"`` is a random split
over the same pool, as the contrast.

Identifiers (flow id, IPs, ports) and the timestamp are not features: IP/port identities are a
well-known shortcut in this dataset, and the point is behaviour, not endpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ids_anomaly.data.preprocess import Dataset
from ids_anomaly.data.tabular import build_tabular_preprocessor, stratified_cap

logger = logging.getLogger(__name__)

NON_FEATURES = [
    "flow_id", "source_ip", "source_port", "destination_ip", "destination_port",
    "Timestamp", "attack_label",
]  # fmt: skip
CATEGORICAL = ["protocol"]
DAY_NAMES = {3: "Mon", 4: "Tue", 5: "Wed", 6: "Thu", 7: "Fri"}
TRAIN_DAYS = ("Mon", "Tue")

_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("benign", "normal"),
    ("ddos", "ddos"),
    ("dos", "dos"),
    ("portscan", "portscan"),
    ("patator", "bruteforce"),
    ("bot", "bot"),
    ("web attack", "webattack"),
    ("infiltration", "infiltration"),
    ("heartbleed", "heartbleed"),
)


def attack_category(label: str) -> str:
    lowered = label.lower()
    for needle, category in _CATEGORY_RULES:
        if needle in lowered:
            return category
    return "other_attack"


def parse_timestamps(raw: pd.Series) -> pd.Series:
    """Parse CIC-IDS2017's mixed timestamp formats into proper datetimes.

    The release mixes ``dd/mm/yyyy HH:MM:SS`` (Monday) with ``d/m/yyyy H:MM`` (other days, no
    seconds), and writes hours on a 12-hour clock *without* AM/PM: the capture ran roughly
    08:00-17:00, so a written hour below 8 is an afternoon hour and is shifted by 12.
    """
    parts = raw.astype("string").str.extract(
        r"^(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2})(?::(\d{2}))?$"
    )
    if parts.iloc[:, :5].isna().any().any():
        raise ValueError("unrecognized CIC-IDS2017 timestamp format")
    day, month, year, hour, minute = (parts[i].astype(int) for i in range(5))
    second = parts[5].fillna("0").astype(int)
    hour = hour.where(hour >= 8, hour + 12)
    return pd.to_datetime(
        {"year": year, "month": month, "day": day, "hour": hour, "minute": minute, "second": second}
    )


def _load_frame(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(raw_dir / "cicids2017" / "flows.parquet")
    ts = parse_timestamps(df["Timestamp"])
    out = pd.DataFrame({"timestamp": ts.to_numpy("datetime64[s]")})
    out["day"] = ts.dt.day.map(DAY_NAMES).astype("string").to_numpy()
    out["category"] = df["attack_label"].astype("string").map(attack_category).to_numpy()
    out["label"] = df["attack_label"].astype("string").to_numpy()
    numeric = [c for c in df.columns if c not in NON_FEATURES + CATEGORICAL]
    out[numeric] = df[numeric].astype("float64")
    out["protocol"] = df["protocol"].astype("string").to_numpy()
    return out


def load_cicids_frames(
    raw_dir: Path,
    split: Literal["temporal", "iid"] = "temporal",
    train_cap: int = 150_000,
    test_cap: int = 250_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Raw (un-scaled) train/test frames after the split and class-aware subsampling."""
    rng = np.random.default_rng(seed)
    df = _load_frame(raw_dir)
    numeric = [c for c in df.columns if c not in ("timestamp", "day", "category", "label", "protocol")]

    is_train: np.ndarray
    if split == "temporal":
        is_train = df["day"].isin(TRAIN_DAYS).to_numpy()
    elif split == "iid":
        is_train = rng.random(len(df)) < 0.4
    else:
        raise ValueError(f"unknown split {split!r}")

    parts: list[pd.DataFrame] = []
    for mask, cap in ((is_train, train_cap), (~is_train, test_cap)):
        part = df[mask].reset_index(drop=True)
        keep = stratified_cap(part["category"].to_numpy(), cap, rng)
        parts.append(part.iloc[keep].reset_index(drop=True))
    logger.info("CIC-IDS2017 %s split: %d train / %d test flows", split, len(parts[0]), len(parts[1]))
    return parts[0], parts[1], numeric


def load_cicids(
    raw_dir: Path,
    split: Literal["temporal", "iid"] = "temporal",
    train_cap: int = 150_000,
    test_cap: int = 250_000,
    seed: int = 42,
) -> tuple[Dataset, Dataset, object]:
    train_df, test_df, numeric = load_cicids_frames(raw_dir, split, train_cap, test_cap, seed)

    features = numeric + CATEGORICAL
    pre = build_tabular_preprocessor(numeric, CATEGORICAL)
    X_train = pre.fit_transform(train_df[features])
    X_test = pre.transform(test_df[features])
    names = [str(n) for n in pre.get_feature_names_out()]

    def _ds(frame: pd.DataFrame, X: np.ndarray) -> Dataset:
        return Dataset(
            X=X.astype(np.float32),
            is_attack=(frame["category"] != "normal").to_numpy(),
            attack_category=frame["category"].astype(str).to_numpy(),
            label=frame["label"].astype(str).to_numpy(),
            feature_names=names,
            timestamp=frame["timestamp"].to_numpy(),
            group=frame["day"].astype(str).to_numpy(),
        )

    return _ds(train_df, X_train), _ds(test_df, X_test), pre


def load_cicids_by_day(
    raw_dir: Path, monday_cap: int = 100_000, day_cap: int = 60_000, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """Monday (attack-free) as the training window and each later day as its own frame.

    Used by the drift experiment: a model fit on Monday's benign traffic is evaluated, with a
    threshold fixed on Monday, on Tue/Wed/Thu/Fri. Every frame is class-aware subsampled.
    """
    rng = np.random.default_rng(seed)
    df = _load_frame(raw_dir)
    numeric = [c for c in df.columns if c not in ("timestamp", "day", "category", "label", "protocol")]

    def _cap(part: pd.DataFrame, cap: int) -> pd.DataFrame:
        part = part.reset_index(drop=True)
        return part.iloc[stratified_cap(part["category"].to_numpy(), cap, rng)].reset_index(drop=True)

    monday = _cap(df[df["day"] == "Mon"], monday_cap)
    days = {d: _cap(df[df["day"] == d], day_cap) for d in ("Tue", "Wed", "Thu", "Fri")}
    return monday, days, numeric
