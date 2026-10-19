"""Verified downloads for the two modern datasets (UNSW-NB15, CIC-IDS2017).

Both are pulled from public, ungated Hugging Face mirrors. Every file is pinned by size and
SHA-256, so a silently changed or truncated mirror fails loudly instead of quietly changing the
benchmark. Downloads stream to a temporary file and are renamed atomically only after
verification.

Mirror quirk worth knowing: the ``wwydmanski/UNSW-NB15`` mirror names its files the opposite
way round from the official split (its ``train.parquet`` is the official 82,332-row *testing*
set). :mod:`ids_anomaly.data.unsw` maps them by their contents' size, not by their names.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_HF = "https://huggingface.co/datasets"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class RemoteFile:
    url: str
    relpath: str
    size: int
    sha256: str


UNSW_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        f"{_HF}/wwydmanski/UNSW-NB15/resolve/main/train.parquet",
        "unsw/train.parquet",
        4_538_929,
        "a0270aeb2219aaa686551cdf6d4f94c4478b69f819225176149606cd1492d5e1",
    ),
    RemoteFile(
        f"{_HF}/wwydmanski/UNSW-NB15/resolve/main/test.parquet",
        "unsw/test.parquet",
        9_618_034,
        "f6989e65032e75770f37a5fa64d1d556effd6ac6240c99b0ab4df73b490c1918",
    ),
)

CICIDS_FILES: tuple[RemoteFile, ...] = (
    RemoteFile(
        f"{_HF}/rdpahalavan/CIC-IDS2017/resolve/main/Network-Flows/CICIDS_Flow.parquet",
        "cicids2017/flows.parquet",
        369_814_810,
        "e64f6adfbcb7f96b018cf36c731be64ef4de3353ccd7340b190590734a0dda35",
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid(path: Path, remote: RemoteFile) -> bool:
    return path.exists() and path.stat().st_size == remote.size and sha256_of(path) == remote.sha256


def fetch(remote: RemoteFile, raw_dir: Path, force: bool = False) -> Path:
    """Download one pinned file into ``raw_dir`` unless a verified copy is already there."""
    dest = raw_dir / remote.relpath
    if not force and _is_valid(dest, remote):
        logger.info("%s already present and verified", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", remote.url, dest)
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(remote.url) as response, tmp.open("wb") as out:  # noqa: S310 - pinned https URLs
        while chunk := response.read(_CHUNK):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != remote.size or digest.hexdigest() != remote.sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Integrity check failed for {remote.relpath}: got {size} bytes / "
            f"{digest.hexdigest()[:12]}..., expected {remote.size} bytes / {remote.sha256[:12]}..."
        )
    tmp.replace(dest)
    return dest


def fetch_unsw(raw_dir: Path, force: bool = False) -> list[Path]:
    return [fetch(f, raw_dir, force) for f in UNSW_FILES]


def fetch_cicids(raw_dir: Path, force: bool = False) -> list[Path]:
    return [fetch(f, raw_dir, force) for f in CICIDS_FILES]
