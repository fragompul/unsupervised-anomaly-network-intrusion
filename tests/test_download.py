from __future__ import annotations

from pathlib import Path

from ids_anomaly.data import download as download_module
from ids_anomaly.data.download import download_raw


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_raw_writes_all_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(download_module.urllib.request, "urlopen", lambda url: _FakeResponse(b"row1\nrow2\n"))

    paths = download_raw(tmp_path)

    assert len(paths) == len(download_module.FILES)
    for path in paths:
        assert path.exists()
        assert path.read_bytes() == b"row1\nrow2\n"


def test_download_raw_skips_existing_files_without_network_call(tmp_path: Path, monkeypatch):
    (tmp_path).mkdir(exist_ok=True)
    for filename in download_module.FILES:
        (tmp_path / filename).write_bytes(b"already here")

    def _fail(*_args, **_kwargs):
        raise AssertionError("should not hit the network when files already exist")

    monkeypatch.setattr(download_module.urllib.request, "urlopen", _fail)

    paths = download_raw(tmp_path)
    for path in paths:
        assert path.read_bytes() == b"already here"


def test_download_raw_force_redownloads(tmp_path: Path, monkeypatch):
    for filename in download_module.FILES:
        (tmp_path / filename).write_bytes(b"stale")

    monkeypatch.setattr(download_module.urllib.request, "urlopen", lambda url: _FakeResponse(b"fresh"))

    paths = download_raw(tmp_path, force=True)
    for path in paths:
        assert path.read_bytes() == b"fresh"
