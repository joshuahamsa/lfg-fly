import json

import pytest

from lfg_fly.connectome import fetch as F

FILES = {"annotations": "a.feather", "neurotransmitters": "n.feather", "weights": "w.feather"}


def _src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in FILES.values():
        (src / name).write_bytes(name.encode() * 3)
    return src


def test_download_writes_files_and_manifest(tmp_path):
    src = _src(tmp_path)
    raw = tmp_path / "raw"
    manifest = F.fetch(raw, base_url=src.as_uri() + "/", files=FILES, expected_bytes=None)
    assert sorted(manifest) == ["annotations", "neurotransmitters", "weights"]
    assert (raw / "a.feather").read_bytes() == b"a.feather" * 3
    on_disk = json.loads((raw / "manifest.json").read_text())
    assert on_disk["weights"]["sha256"] == F.sha256_file(raw / "w.feather")


def test_local_from_uses_aliases_without_download(tmp_path):
    local = tmp_path / "review"
    local.mkdir()
    (local / "ann.feather").write_bytes(b"A")
    (local / "nt.feather").write_bytes(b"N")
    (local / "weights.feather").write_bytes(b"W")
    raw = tmp_path / "raw"
    F.fetch(raw, base_url="file:///nonexistent/", local_from=local, files=FILES,
             expected_bytes=None)
    assert (raw / "a.feather").read_bytes() == b"A"
    assert (raw / "w.feather").read_bytes() == b"W"


def test_size_mismatch_raises(tmp_path):
    src = _src(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        F.fetch(tmp_path / "raw", base_url=src.as_uri() + "/", files=FILES,
                expected_bytes={"annotations": 1})


def test_existing_file_is_not_refetched(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in FILES.values():
        (raw / name).write_bytes(b"kept")
    F.fetch(raw, base_url="file:///nonexistent/", files=FILES, expected_bytes=None)
    assert (raw / "a.feather").read_bytes() == b"kept"
