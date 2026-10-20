"""Tests for the UNSW-NB15 / CIC-IDS2017 layer, run on tiny synthetic stand-ins (no network)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids_anomaly.data import cicids, fetch, harmonized, unsw
from ids_anomaly.data.registry import DATASETS, load_dataset
from ids_anomaly.data.tabular import signed_log1p, stratified_cap

UNSW_CATS = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS"]


def _unsw_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = rng.choice(UNSW_CATS, size=n, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    df = pd.DataFrame(
        {
            "dur": rng.exponential(1.0, n),
            "proto": rng.choice(["tcp", "udp", "arp", "ospf"], n, p=[0.6, 0.3, 0.05, 0.05]),
            "service": rng.choice(["-", "http", "dns"], n),
            "state": rng.choice(["FIN", "INT", "CON"], n),
            "spkts": rng.integers(1, 100, n),
            "dpkts": rng.integers(0, 100, n),
            "sbytes": rng.integers(40, 10_000, n),
            "dbytes": rng.integers(0, 10_000, n),
            "smean": rng.uniform(40, 1500, n),
            "dmean": rng.uniform(0, 1500, n),
            "sinpkt": rng.exponential(10, n),
            "dinpkt": rng.exponential(10, n),
            "sjit": rng.exponential(50, n),
            "djit": rng.exponential(50, n),
            "stcpb": rng.integers(0, 2**31, n),
            "dtcpb": rng.integers(0, 2**31, n),
            "attack_cat": cat,
            "label": (cat != "Normal").astype(int),
        }
    )
    return df


@pytest.fixture
def unsw_raw_dir(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "unsw").mkdir()
    # the mirror's names are swapped: "train.parquet" is the *smaller* official testing set
    _unsw_frame(60, 1).to_parquet(tmp_path / "unsw" / "train.parquet")
    _unsw_frame(150, 2).to_parquet(tmp_path / "unsw" / "test.parquet")
    monkeypatch.setattr(unsw, "OFFICIAL_TRAIN_ROWS", 150)
    monkeypatch.setattr(unsw, "OFFICIAL_TEST_ROWS", 60)
    return tmp_path


def _cic_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = rng.choice(["3/7/2017", "4/7/2017", "5/7/2017", "6/7/2017", "7/7/2017"], n)
    labels = np.where(
        np.isin(days, ["3/7/2017", "4/7/2017"]),
        rng.choice(["BENIGN", "FTP-Patator"], n, p=[0.97, 0.03]),
        rng.choice(["BENIGN", "DoS Hulk", "PortScan", "Heartbleed"], n, p=[0.7, 0.15, 0.13, 0.02]),
    )
    df = pd.DataFrame(
        {
            "flow_id": np.arange(n),
            "source_ip": "10.0.0.1",
            "source_port": rng.integers(1024, 65535, n),
            "destination_ip": "10.0.0.2",
            "destination_port": rng.integers(1, 1024, n),
            "protocol": rng.choice(["tcp", "udp", "other"], n, p=[0.7, 0.28, 0.02]),
            "Timestamp": [f"{d} {rng.integers(8, 12)}:{rng.integers(10, 59)}" for d in days],
            "Flow Duration": rng.integers(1, 5_000_000, n),
            "Total Fwd Packets": rng.integers(1, 50, n),
            "Total Backward Packets": rng.integers(0, 50, n),
            "Total Length of Fwd Packets": rng.integers(0, 5000, n),
            "Total Length of Bwd Packets": rng.integers(0, 5000, n),
            "Fwd Packet Length Mean": rng.uniform(0, 800, n),
            "Bwd Packet Length Mean": rng.uniform(0, 800, n),
            "Fwd IAT Mean": rng.uniform(0, 1e6, n),
            "Bwd IAT Mean": rng.uniform(0, 1e6, n),
            "Fwd IAT Std": rng.uniform(0, 1e6, n),
            "Bwd IAT Std": rng.uniform(0, 1e6, n),
            "Flow Bytes/s": np.where(rng.random(n) < 0.05, np.inf, rng.uniform(0, 1e5, n)),
            "Init_Win_bytes_forward": rng.choice([-1, 229, 8192], n),
            "attack_label": labels,
        }
    )
    return df


@pytest.fixture
def cic_raw_dir(tmp_path: Path) -> Path:
    (tmp_path / "cicids2017").mkdir()
    _cic_frame(3000, 3).to_parquet(tmp_path / "cicids2017" / "flows.parquet")
    return tmp_path


def test_signed_log1p_handles_negative_and_nonfinite():
    out = signed_log1p(np.array([-3.0, 0.0, 3.0, np.inf, -np.inf, np.nan]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(-out[2])
    assert out[3] == out[4] == out[5] == 0.0


def test_stratified_cap_keeps_rare_classes_whole():
    rng = np.random.default_rng(0)
    cats = np.array(["normal"] * 10_000 + ["rare"] * 30)
    idx = stratified_cap(cats, cap=1000, rng=rng)
    kept = cats[idx]
    assert (kept == "rare").sum() == 30
    assert 900 <= (kept == "normal").sum() <= 2100
    assert np.all(np.diff(idx) > 0)


def test_stratified_cap_noop_below_cap():
    cats = np.array(["a", "b", "a"])
    assert stratified_cap(cats, cap=10, rng=np.random.default_rng(0)).tolist() == [0, 1, 2]


def test_fetch_rejects_corrupt_download(tmp_path: Path, monkeypatch):
    payload = b"not the real file"
    remote = fetch.RemoteFile("https://example.invalid/x", "x/f.bin", len(payload), "0" * 64)

    class _Resp:
        def __init__(self):
            self._b = payload

        def read(self, n=-1):
            out, self._b = self._b, b""
            return out

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda url: _Resp())
    with pytest.raises(ValueError, match="Integrity check failed"):
        fetch.fetch(remote, tmp_path)
    assert not (tmp_path / "x" / "f.bin").exists()
    assert not (tmp_path / "x" / "f.bin.part").exists()


def test_fetch_accepts_matching_hash_and_skips_when_present(tmp_path: Path, monkeypatch):
    payload = b"hello noctua"
    remote = fetch.RemoteFile(
        "https://example.invalid/y", "y/f.bin", len(payload), hashlib.sha256(payload).hexdigest()
    )
    calls = []

    class _Resp:
        def __init__(self):
            self._b = payload

        def read(self, n=-1):
            out, self._b = self._b, b""
            return out

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(url):
        calls.append(url)
        return _Resp()

    monkeypatch.setattr(fetch.urllib.request, "urlopen", _open)
    path = fetch.fetch(remote, tmp_path)
    assert path.read_bytes() == payload
    fetch.fetch(remote, tmp_path)
    assert len(calls) == 1  # verified copy is not re-downloaded


def test_unsw_loader_identifies_official_split_by_size(unsw_raw_dir: Path):
    train, test, _ = unsw.load_unsw(unsw_raw_dir)
    assert len(train) == 150 and len(test) == 60
    assert train.X.shape[1] == test.X.shape[1]
    assert np.isfinite(train.X).all() and np.isfinite(test.X).all()
    assert np.abs(test.X).max() <= 10.0 + 1e-6
    assert set(train.attack_category) <= {"normal", "generic", "exploits", "fuzzers", "dos"}
    assert (train.is_attack == (train.attack_category != "normal")).all()
    assert not any(n.endswith(("stcpb", "dtcpb")) for n in train.feature_names)


def test_unsw_loader_rejects_wrong_row_counts(unsw_raw_dir: Path, monkeypatch):
    monkeypatch.setattr(unsw, "OFFICIAL_TRAIN_ROWS", 999)
    with pytest.raises(ValueError, match="Unexpected UNSW-NB15 row counts"):
        unsw.load_unsw(unsw_raw_dir)


def test_parse_timestamps_handles_mixed_formats_and_pm_shift():
    ts = cicids.parse_timestamps(
        pd.Series(["03/07/2017 08:55:58", "5/7/2017 4:23", "7/7/2017 10:04"])
    )
    assert ts.iloc[0] == pd.Timestamp("2017-07-03 08:55:58")
    assert ts.iloc[1] == pd.Timestamp("2017-07-05 16:23:00")  # 4:23 without AM/PM -> afternoon
    assert ts.iloc[2] == pd.Timestamp("2017-07-07 10:04:00")


def test_parse_timestamps_rejects_garbage():
    with pytest.raises(ValueError, match="unrecognized"):
        cicids.parse_timestamps(pd.Series(["yesterday"]))


def test_cic_category_mapping():
    assert cicids.attack_category("BENIGN") == "normal"
    assert cicids.attack_category("DoS Hulk") == "dos"
    assert cicids.attack_category("DDoS") == "ddos"
    assert cicids.attack_category("Web Attack – XSS") == "webattack"
    assert cicids.attack_category("SSH-Patator") == "bruteforce"
    assert cicids.attack_category("Heartbleed") == "heartbleed"


def test_cic_temporal_split_trains_on_early_days_only(cic_raw_dir: Path):
    train, test, _ = cicids.load_cicids(cic_raw_dir, train_cap=800, test_cap=1200)
    assert set(train.group) <= {"Mon", "Tue"}
    assert set(test.group) <= {"Wed", "Thu", "Fri"}
    assert np.isfinite(train.X).all() and np.isfinite(test.X).all()
    assert train.timestamp is not None and test.timestamp is not None
    assert train.timestamp.max() < test.timestamp.min()
    assert "heartbleed" in set(test.attack_category)  # rare class survives the cap


def test_cic_iid_split_mixes_days(cic_raw_dir: Path):
    train, test, _ = cicids.load_cicids(cic_raw_dir, split="iid", train_cap=800, test_cap=1200)
    assert {"Mon", "Wed"} & set(train.group) and {"Mon", "Wed"} & set(test.group)


def test_harmonized_views_share_columns_and_units(unsw_raw_dir: Path, cic_raw_dir: Path):
    u_train, u_test = harmonized.load_harmonized("unsw_nb15", unsw_raw_dir)
    c_train, c_test = harmonized.load_harmonized("cicids2017", cic_raw_dir)
    assert list(u_train.features.columns) == list(c_train.features.columns) == harmonized.HARMONIZED_COLUMNS
    # microseconds -> seconds: CIC durations were generated in [1, 5e6] us
    assert c_train.features["dur_s"].max() <= 5.0 + 1e-9
    pre = harmonized.fit_scaler(u_train)
    X = harmonized.transform(c_test, pre)  # a foreign dataset goes through the same scaler
    assert X.shape[0] == len(c_test) and np.isfinite(X).all()
    assert X.shape[1] == harmonized.transform(u_test, pre).shape[1]


def test_registry_lists_all_datasets_and_rejects_unknown(tmp_path: Path):
    assert set(DATASETS) == {"nsl_kdd", "unsw_nb15", "cicids2017"}
    with pytest.raises(KeyError):
        load_dataset("nope", tmp_path)


def test_registry_loads_without_download(unsw_raw_dir: Path):
    bundle = load_dataset("unsw_nb15", unsw_raw_dir, download=False)
    assert bundle.key == "unsw_nb15" and len(bundle.train) == 150
