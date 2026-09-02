"""Fetch the NSL-KDD train/test splits into ``data/raw/``.

NSL-KDD is a de-duplicated, difficulty-rebalanced revision of the original
KDD Cup 1999 intrusion detection set (Tavallaee et al., 2009), distributed
without any access-request gate. We pull it from a long-standing public
mirror of the official ISCX/UNB release rather than the raw KDD99 dump,
since the original has severe train/test redundancy that inflates scores.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"

FILES: dict[str, str] = {
    "KDDTrain+.txt": f"{_BASE_URL}/KDDTrain%2B.txt",
    "KDDTest+.txt": f"{_BASE_URL}/KDDTest%2B.txt",
}


def download_raw(dest_dir: Path, force: bool = False) -> list[Path]:
    """Download the NSL-KDD train/test CSV-like files if not already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in FILES.items():
        dest = dest_dir / filename
        if dest.exists() and not force:
            logger.info("%s already present, skipping download", dest)
            paths.append(dest)
            continue
        logger.info("Downloading %s -> %s", url, dest)
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https allowlist above
            dest.write_bytes(response.read())
        paths.append(dest)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parents[3]
    download_raw(root / "data" / "raw")
