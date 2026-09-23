"""Fetch the three MaleCNS v1.0 feathers (CC BY 4.0) into FLY_DATA_DIR/raw.

The connectome is downloaded, never redistributed. `local_from` hardlinks
copies that already exist on this box, so the worn SSD doesn't write them twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}
EXPECTED_BYTES = {
    "annotations": 14_483_314,
    "neurotransmitters": 43_282_834,
    "weights": 508_025_642,
}
LOCAL_ALIASES = {
    "annotations": ("ann.feather",),
    "neurotransmitters": ("nt.feather",),
    "weights": ("weights.feather",),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def _download(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    part.replace(dest)


def _local_source(local_from: Path, key: str, fname: str) -> Path | None:
    for name in (fname, *LOCAL_ALIASES.get(key, ())):
        candidate = local_from / name
        if candidate.exists():
            return candidate
    return None


def fetch(
    raw: Path,
    *,
    base_url: str = BASE_URL,
    local_from: Path | None = None,
    files: dict[str, str] = FILES,
    expected_bytes: dict[str, int] | None = EXPECTED_BYTES,
) -> dict[str, dict]:
    raw.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for key, fname in files.items():
        dest = raw / fname
        if not dest.exists():
            src = _local_source(local_from, key, fname) if local_from else None
            if src is not None:
                _link_or_copy(src, dest)
            else:
                _download(base_url + fname, dest)
        size = dest.stat().st_size
        if expected_bytes is not None and key in expected_bytes and size != expected_bytes[key]:
            raise ValueError(f"{fname}: {size} bytes, expected {expected_bytes[key]}")
        manifest[key] = {"file": fname, "bytes": size, "sha256": sha256_file(dest)}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
