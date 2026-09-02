from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ids_anomaly.data.schema import FEATURE_COLUMNS

_PROTOCOLS = ["tcp", "udp", "icmp"]
_SERVICES = ["http", "ftp_data", "private", "domain_u"]
_FLAGS = ["SF", "S0", "REJ"]


def _synthetic_row(rng: np.random.Generator, label: str) -> str:
    values = []
    for col in FEATURE_COLUMNS:
        if col == "protocol_type":
            values.append(rng.choice(_PROTOCOLS))
        elif col == "service":
            values.append(rng.choice(_SERVICES))
        elif col == "flag":
            values.append(rng.choice(_FLAGS))
        elif col.endswith("_rate"):
            values.append(f"{rng.uniform(0, 1):.2f}")
        else:
            values.append(str(int(rng.integers(0, 500))))
    values.append(label)
    values.append(str(int(rng.integers(1, 21))))  # difficulty
    return ",".join(values)


def _write_synthetic_split(path: Path, n_rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["normal", "neptune", "satan", "guess_passwd", "buffer_overflow"],
        size=n_rows,
        p=[0.5, 0.25, 0.1, 0.1, 0.05],
    )
    lines = [_synthetic_row(rng, label) for label in labels]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    """A tiny NSL-KDD-shaped raw directory, so preprocessing tests don't depend on the network."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_synthetic_split(raw_dir / "KDDTrain+.txt", n_rows=200, seed=1)
    _write_synthetic_split(raw_dir / "KDDTest+.txt", n_rows=80, seed=2)
    return raw_dir
