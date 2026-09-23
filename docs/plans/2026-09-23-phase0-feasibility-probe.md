# Phase 0: Feasibility Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the fly's connectome loader, candidate brains, senses and a planted-taste probe, then run the probe grid on the GPU and record a pass/fail verdict against the ≥ 0.60 gate.

**Architecture:**
- **Package:** `lfg_fly`, a Python package in `~/lfg-fly` with its own venv (torch cu121 on the RTX 2060 SUPER).
- **Connectome:** loaded once into an integer-valued CSR, so GPU runs are bitwise reproducible.
- **Brains:** four candidates (`lif`, `lif-avg`, `lif-volley`, `rate`).
- **Input codes:** five (`nose`, `eyes`, `eyes+nose`, `all-sensory-brain`, `all-sensory+vnc`).
- **Probe:** a grid runner screens every setting on cheap constraint and smoothness checks (Stage A), runs the full planted-taste probe on the survivors (Stage B), and runs rewired and sign-shuffled twins of the best one (Stage C). Results go to `data/probe/`, the report to `docs/PHASE0.md`.

**Tech Stack:** Python 3.10, torch 2.5.1+cu121, numpy 2.2.6, pandas 2.3.3, pyarrow 25.0.1, pillow, pyyaml, pytest, ruff.

**Spec:** `docs/specs/2026-09-22-fly-design.md`: §2 (brain, senses) and §3.0 (Phase 0). Read both before starting any task.

## Global Constraints

- **The CPU belongs to the XRPL validator.**
  - `lfg_fly/__init__.py` imports `lfg_fly.env` first, which sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1.
  - Every command in this plan runs under `nice -n 10 ionice -c3`.
  - Heavy simulation and training run on `cuda`. Tests use tiny graphs on CPU.
- **Pins:** `torch==2.5.1` from `https://download.pytorch.org/whl/cu121` locally and `/whl/cpu` in CI; `numpy==2.2.6`; `pandas==2.3.3`; `pyarrow==25.0.1`; Python `>=3.10,<3.11`.
- **No data in the repo.** Feathers, graphs, caches and renders live in `FLY_DATA_DIR` (default `~/fly-data`). The repo gets only code, tests, `data/probe/*.json*` results and docs. `.gitignore` already blocks `*.feather`, `*.npz`, `*.pt` and `.env*`.
- **No AI attribution** in commits or PR bodies (no `Co-Authored-By` trailers, no "Generated with" footers).
- **Nothing** in Phase 0 talks to a wallet or signs anything. LFG is read only through the public `GET /api/rarity` and `GET /api/layer`, plus `raw.githubusercontent.com` for `trait_config.yaml`.
- **Sign map keys** (lower-case, exact): `acetylcholine`, `gaba`, `glutamate`, `histamine`, `dopamine`, `serotonin`, `octopamine`, `unclear`. A missing NT row counts as `unclear`; an unknown non-null string fails the build.
- **Real-data facts** (v1.0, verified 2026-09-22), used for sanity asserts in Task 12:

  | Quantity | v1.0 value |
  |---|---|
  | Traced neurons | 165,122 |
  | Edges at ≥3 synapses | 10,511,038 |
  | Readout neurons (1,314 DN + 708 `vnc_motor` + 107 `cb_motor`) | 2,129 |
  | Photoreceptors | 4,107 |
  | ORNs | 2,635 in 53 glomeruli |
  | Brain sensory neurons (`ol_sensory` 4,114 + `cb_sensory` 4,868) | 8,982 |
  | `vnc_sensory` neurons | 6,365 |
  | `sensory_any` (brain + VNC sensory; no tonic bias) | 15,347 |
  | Traced bodies with no NT row | 502 |
- **Pinned LFG commit** for `trait_config.yaml`: `0415a63afdcea11156b57169d34e38bb8d377adc`.
- **Gate:** the best setting that passes all constraints reaches **held-out ≥ 0.60** on the planted taste.
- **Clarifications of the spec:**
  - Exact bitwise reproducibility holds for the spiking brains (integer spike × integer weight sums). The `rate` brain multiplies real-valued activity, so on the GPU it is reproducible to a tolerance (`rtol=1e-4`), not bitwise.
  - The runaway rule ("no step has > 10% of neurons firing") is spiking language. For `rate`, "firing" in that rule means **saturated** (`h > 9`, within 10% of the clip at 10). A tonic bias keeps every rate unit slightly active by design; saturation is what runaway looks like there. The readout-activity rule uses `h > 0.05`.

## File Structure

```
pyproject.toml                    # package + pins + tool config
scripts/setup_venv.sh             # uv venv + torch cu121 + editable install
.github/workflows/ci.yml          # CPU-only: ruff, pytest (torch cpu), gitleaks seed test
lfg_fly/__init__.py               # imports env first
lfg_fly/env.py                    # CPU thread caps
lfg_fly/paths.py                  # FLY_DATA_DIR layout
lfg_fly/cli.py                    # `fly <command>`
lfg_fly/connectome/__init__.py
lfg_fly/connectome/fetch.py       # download / hardlink + sha256 manifest
lfg_fly/connectome/build.py       # Graph: traced filter, threshold, sign map, integer CSR
lfg_fly/connectome/neurons.py     # Populations: readout, photoreceptors, ORNs, sensory pools
lfg_fly/connectome/columns.py     # partner-derived photoreceptor columns + lattice
lfg_fly/brain/__init__.py
lfg_fly/brain/sim.py              # BrainParams, Simulator (torch), simulate_reference (numpy)
lfg_fly/brain/senses.py           # trait codes, Retina, InputBuilder
lfg_fly/teacher/__init__.py
lfg_fly/teacher/catalog.py        # LFG catalog + layer cache (public API)
lfg_fly/teacher/render.py         # z-order from pinned trait_config, LayerBank, composite
lfg_fly/teacher/sample.py         # looks, near/far pairs, families
lfg_fly/teacher/probe.py          # planted taste, BT + grouped CV, bootstrap, checks
lfg_fly/teacher/rewire.py         # degree-preserving rewire with repair, sign shuffle
lfg_fly/teacher/grid.py           # settings, Stage A/B/C, verdict, resumable JSONL
lfg_fly/teacher/report.py         # docs/PHASE0.md generator
tests/conftest.py                 # FLY_DATA_DIR → tmp, tiny-graph fixtures
tests/test_*.py                   # one file per module
data/probe/                       # committed results (Task 12)
docs/PHASE0.md                    # committed report (Task 12)
```

---

### Task 1: Scaffold, CPU discipline, CI

**Files:**
- Create: `pyproject.toml`, `scripts/setup_venv.sh`, `.github/workflows/ci.yml`, `lfg_fly/__init__.py`, `lfg_fly/env.py`, `lfg_fly/paths.py`, `lfg_fly/cli.py`, `lfg_fly/connectome/__init__.py`, `lfg_fly/brain/__init__.py`, `lfg_fly/teacher/__init__.py`
- Test: `tests/conftest.py`, `tests/test_env_paths.py`, `tests/test_gitleaks_seed.py`

**Interfaces:**
- Produces:
  - `lfg_fly.env.THREAD_VARS: tuple[str, ...]`
  - `lfg_fly.env.configure_libraries() -> None`
  - `lfg_fly.paths.data_dir() -> Path`
  - `lfg_fly.paths.raw_dir() -> Path`
  - `lfg_fly.paths.graph_dir() -> Path`
  - `lfg_fly.paths.cache_dir() -> Path`
  - `lfg_fly.paths.network_dir(network: str) -> Path`
  - `lfg_fly.paths.repo_root() -> Path`
  - `lfg_fly.cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Create the venv script and project file**

`scripts/setup_venv.sh`:
```bash
#!/usr/bin/env bash
# Build the fly's venv: torch for the RTX 2060 SUPER (driver 535 -> CUDA <= 12.2 -> cu121 wheels).
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lfg-fly"
version = "0.0.1"
description = "A MaleCNS fruit-fly connectome that dresses its own LFG NFT"
requires-python = ">=3.10,<3.11"
license = { text = "MIT" }
dependencies = [
  "torch==2.5.1",
  "numpy==2.2.6",
  "pandas==2.3.3",
  "pyarrow==25.0.1",
  "pillow>=10",
  "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]

[project.scripts]
fly = "lfg_fly.cli:main"

[tool.setuptools.packages.find]
include = ["lfg_fly*"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Build the venv**

Run: `chmod +x scripts/setup_venv.sh && nice -n 10 ionice -c3 scripts/setup_venv.sh`
Expected: the last line prints `torch 2.5.1+cu121 cuda True`.

- [ ] **Step 3: Write the failing tests**

`tests/conftest.py`:
```python
import lfg_fly  # noqa: F401  (must be first: applies the CPU thread caps)
import pytest


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """No test may touch ~/fly-data: every test gets its own FLY_DATA_DIR."""
    monkeypatch.setenv("FLY_DATA_DIR", str(tmp_path / "fly-data"))
    return tmp_path / "fly-data"
```

`tests/test_env_paths.py`:
```python
import os

import pytest

from lfg_fly import env, paths


def test_thread_caps_are_set_on_import():
    for var in env.THREAD_VARS:
        assert os.environ[var] == "1"


def test_data_dir_follows_env(_data_dir):
    assert paths.data_dir() == _data_dir
    assert paths.raw_dir() == _data_dir / "raw"
    assert paths.graph_dir() == _data_dir / "graph"
    assert paths.cache_dir() == _data_dir / "cache"


def test_network_dir_is_scoped(_data_dir):
    assert paths.network_dir("mainnet") == _data_dir / "mainnet"
    assert paths.network_dir("testnet") == _data_dir / "testnet"
    with pytest.raises(ValueError):
        paths.network_dir("devnet")


def test_repo_root_contains_pyproject():
    assert (paths.repo_root() / "pyproject.toml").exists()


def test_cli_help_exits_zero(capsys):
    from lfg_fly import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "fly" in capsys.readouterr().out
```

`tests/test_gitleaks_seed.py`:
```python
"""The repo is public and gitleaks' defaults miss XRPL seeds: prove our rule fires."""
import shutil
import subprocess
from pathlib import Path

import pytest
from lfg_fly.paths import repo_root

gitleaks = shutil.which("gitleaks")
pytestmark = pytest.mark.skipif(gitleaks is None, reason="gitleaks not installed (CI installs it)")


def _scan(tmp_path: Path, text: str) -> int:
    (tmp_path / "leak.txt").write_text(text)
    config = repo_root() / ".gitleaks.toml"
    cmd = [gitleaks, "dir", str(tmp_path), "--config", str(config), "--no-banner", "--exit-code", "1"]
    return subprocess.run(cmd, capture_output=True).returncode


def test_rule_catches_generated_seeds(tmp_path):
    xrpl = pytest.importorskip("xrpl")
    from xrpl.core.keypairs import generate_seed

    seeds = [generate_seed(algorithm=a) for a in (xrpl.CryptoAlgorithm.SECP256K1, xrpl.CryptoAlgorithm.ED25519)]
    assert _scan(tmp_path, "\n".join(seeds)) == 1


def test_rule_ignores_addresses_and_hashes(tmp_path):
    text = "rrrrrrrrrrrrrrrrrNAMEtxvNvQ\n4109C6F2045FC7EFF4CDE8F9905D19C28820D86304080FF886B299F0206E42B5\n"
    assert _scan(tmp_path, text) == 0
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_env_paths.py`
Expected: FAIL (`ModuleNotFoundError: lfg_fly.env` or missing attributes).

- [ ] **Step 5: Implement**

`lfg_fly/env.py`:
```python
"""CPU discipline for a box whose cores belong to an XRPL validator.

Imported first by lfg_fly/__init__.py, so the thread caps are in the
environment before numpy, torch or pyarrow is imported anywhere in the package.
"""

from __future__ import annotations

import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def apply_thread_caps() -> None:
    for var in THREAD_VARS:
        os.environ[var] = "1"


def configure_libraries() -> None:
    """Cap the libraries' own pools. Call once, before heavy work."""
    import pyarrow
    import torch

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # torch allows this once per process; a second call is harmless to skip


apply_thread_caps()
```

`lfg_fly/__init__.py`:
```python
"""The fly: a MaleCNS fruit-fly connectome that dresses its own LFG NFT."""

from lfg_fly import env as _env  # noqa: F401  (thread caps before numpy/torch/pyarrow)

__version__ = "0.0.1"
```

`lfg_fly/paths.py`:
```python
"""Where the fly keeps data: FLY_DATA_DIR (default ~/fly-data), never the repo."""

from __future__ import annotations

import os
from pathlib import Path

NETWORKS = ("mainnet", "testnet")


def data_dir() -> Path:
    return Path(os.environ.get("FLY_DATA_DIR", str(Path.home() / "fly-data"))).expanduser()


def raw_dir() -> Path:
    return data_dir() / "raw"


def graph_dir() -> Path:
    return data_dir() / "graph"


def cache_dir() -> Path:
    return data_dir() / "cache"


def network_dir(network: str) -> Path:
    if network not in NETWORKS:
        raise ValueError(f"unknown network {network!r}")
    return data_dir() / network


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
```

`lfg_fly/cli.py`:
```python
"""`fly` command line. Subcommands are added by later tasks."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fly", description="The fly: an LFG-dressing connectome")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 0
    return int(args.func(args) or 0)
```

Also create empty `lfg_fly/connectome/__init__.py`, `lfg_fly/brain/__init__.py` and `lfg_fly/teacher/__init__.py`.

`.github/workflows/ci.yml`:
```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install (CPU torch)
        run: |
          pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
          pip install -e '.[dev]' xrpl-py
      - name: Install gitleaks
        run: |
          curl -sSL https://github.com/gitleaks/gitleaks/releases/download/v8.30.0/gitleaks_8.30.0_linux_x64.tar.gz | tar -xz gitleaks
          sudo mv gitleaks /usr/local/bin/
      - run: ruff check .
      - run: pytest -q
      - name: gitleaks (repo history)
        run: gitleaks git . --config .gitleaks.toml --no-banner
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_env_paths.py tests/test_gitleaks_seed.py && .venv/bin/ruff check .`
Expected: the env/paths tests PASS; the gitleaks tests are SKIPPED locally (gitleaks isn't on PATH); ruff reports no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml scripts/setup_venv.sh .github/workflows/ci.yml lfg_fly tests
git commit -m "feat: package scaffold, CPU thread caps, data paths, CI with gitleaks seed test"
```

---

### Task 2: Connectome fetch

**Files:**
- Create: `lfg_fly/connectome/fetch.py`
- Modify: `lfg_fly/cli.py` (add `fetch`)
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `lfg_fly.paths.raw_dir()`
- Produces:
  - `FILES: dict[str, str]` (keys `annotations`, `neurotransmitters`, `weights`)
  - `sha256_file(path: Path) -> str`
  - `fetch(raw: Path, *, base_url: str = BASE_URL, local_from: Path | None = None, files: dict[str, str] = FILES, expected_bytes: dict[str, int] | None = EXPECTED_BYTES) -> dict[str, dict]`
  - `raw/manifest.json`

- [ ] **Step 1: Write the failing test**

`tests/test_fetch.py`:
```python
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
    F.fetch(raw, base_url="file:///nonexistent/", local_from=local, files=FILES, expected_bytes=None)
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_fetch.py`
Expected: FAIL (`ImportError: cannot import name 'fetch'`).

- [ ] **Step 3: Implement**

`lfg_fly/connectome/fetch.py`:
```python
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
EXPECTED_BYTES = {"annotations": 14_483_314, "neurotransmitters": 43_282_834, "weights": 508_025_642}
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
```

Add to `lfg_fly/cli.py` (inside `build_parser`, after `sub = parser.add_subparsers(dest="command")`; keep that variable name `sub` for every later task):
```python
    p = sub.add_parser("fetch", help="download (or hardlink) the MaleCNS feathers")
    p.add_argument("--from", dest="local_from", type=Path, default=None)
    p.set_defaults(func=_cmd_fetch)
```
and at module level:
```python
from pathlib import Path


def _cmd_fetch(args: argparse.Namespace) -> int:
    from lfg_fly import paths
    from lfg_fly.connectome.fetch import fetch

    manifest = fetch(paths.raw_dir(), local_from=args.local_from)
    for key, row in manifest.items():
        print(f"{key}: {row['bytes']:,} bytes sha256 {row['sha256'][:16]}…")
    return 0
```
(Change the Task 1 line `parser.add_subparsers(dest="command")` to `sub = parser.add_subparsers(dest="command")`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_fetch.py tests/test_env_paths.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/connectome/fetch.py lfg_fly/cli.py tests/test_fetch.py
git commit -m "feat(connectome): fetch MaleCNS feathers with sha256 manifest and local hardlinks"
```

---

### Task 3: Graph build and populations

**Files:**
- Create: `lfg_fly/connectome/build.py`, `lfg_fly/connectome/neurons.py`
- Modify: `lfg_fly/cli.py` (add `build`)
- Test: `tests/test_build.py`, `tests/test_neurons.py`

**Interfaces:**
- Consumes: `lfg_fly.connectome.fetch.FILES`, `lfg_fly.paths.raw_dir()`, `graph_dir()`
- Produces (`lfg_fly.connectome.build`):
  - `SIGN: dict[str, int]`
  - `class Graph`, a frozen dataclass with fields:
    - `body_id: int64[N]`, `superclass: str[N]`, `cell_type: str[N]`, `root_side: str[N]`
    - `sign: int8[N]`, `crow: int64[N+1]`, `col: int64[nnz]`, `count: int32[nnz]`, `insum: float64[N]`
    - `min_syn: int`, `nt_missing: int`

    It also has the properties `n` and `ival -> float32[nnz]` (sign[col] × count) and the method `graph_hash() -> str`.
  - `signs_from_nt(bodies: np.ndarray, nt: pd.DataFrame) -> tuple[np.ndarray, int]`
  - `csr_from_edges(pre, post, count, n, device="cpu") -> tuple[crow, col, count, insum]`
  - `build_graph(ann, nt, edges, *, min_syn=3, device="cpu") -> Graph`
  - `with_edges(g: Graph, pre, post, count, device="cpu") -> Graph` (same neurons, new edges; used by rewire)
  - `with_sign(g: Graph, sign) -> Graph`
  - `edge_list(g: Graph) -> tuple[pre, post, count]` (in CSR order)
  - `load_annotations(path) -> DataFrame`
  - `load_nt(path) -> DataFrame`
  - `load_edges(path, min_syn) -> DataFrame`
  - `save_graph(g, path)` / `load_graph(path) -> Graph`
- Produces (`lfg_fly.connectome.neurons`):
  - `PHOTORECEPTOR_TYPES`, `CHANNEL_BY_TYPE`, `READOUT_SUPERCLASSES`
  - `class Populations`, a frozen dataclass with fields `readout`, `photoreceptor`, `photoreceptor_type`, `orn`, `orn_glomerulus`, `sensory_brain`, `sensory_vnc`, `sensory_any: bool[N]`
  - `populations(g: Graph) -> Populations`

- [ ] **Step 1: Write the failing tests**

`tests/test_build.py`:
```python
import numpy as np
import pandas as pd
import pytest

from lfg_fly.connectome import build as B


def _tables():
    ann = pd.DataFrame({
        "bodyId": [10, 20, 30, 40, 50],
        "status": ["Traced", "Traced", "Traced", "Traced", "Orphan"],
        "superclass": ["ol_sensory", "ol_intrinsic", "descending_neuron", "vnc_motor", "x"],
        "type": ["R1-R6", "L1", "DNa01", "MN1", "x"],
        "rootSide": ["R", None, None, None, None],
    })
    nt = pd.DataFrame({"body": [10, 20, 30], "consensus_nt": ["histamine", "acetylcholine", "gaba"]})
    edges = pd.DataFrame({
        "body_pre": [10, 20, 20, 30, 10, 50],
        "body_post": [20, 30, 40, 40, 30, 20],
        "weight": [5, 3, 7, 2, 4, 9],
    })
    return ann, nt, edges


def test_build_filters_thresholds_and_signs():
    ann, nt, edges = _tables()
    g = B.build_graph(ann, nt, edges, min_syn=3)
    assert g.n == 4 and list(g.body_id) == [10, 20, 30, 40]
    assert g.nt_missing == 1  # body 40 has no NT row -> unclear (+1)
    assert list(g.sign) == [-1, 1, -1, 1]
    # kept edges (>=3, both traced): 10->20 (5), 20->30 (3), 20->40 (7), 10->30 (4)
    dense = np.zeros((4, 4), dtype=np.float32)
    for post in range(4):
        for k in range(g.crow[post], g.crow[post + 1]):
            dense[post, g.col[k]] = g.ival[k]
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[1, 0] = -5
    expected[2, 1] = 3
    expected[3, 1] = 7
    expected[2, 0] = -4
    assert np.array_equal(dense, expected)
    assert g.insum.tolist() == [1.0, 5.0, 7.0, 7.0]  # unsigned synapse sums, 0 -> 1


def test_unknown_nt_fails_but_missing_is_unclear():
    ann, nt, edges = _tables()
    bad = pd.concat([nt, pd.DataFrame({"body": [40], "consensus_nt": ["tyramine"]})])
    with pytest.raises(ValueError, match="tyramine"):
        B.build_graph(ann, bad, edges)


def test_save_load_roundtrip_and_hash(tmp_path):
    ann, nt, edges = _tables()
    g = B.build_graph(ann, nt, edges)
    B.save_graph(g, tmp_path / "g.npz")
    h = B.load_graph(tmp_path / "g.npz")
    assert h.graph_hash() == g.graph_hash()
    assert list(h.cell_type) == list(g.cell_type)
    assert h.min_syn == 3 and h.nt_missing == 1


def test_with_sign_and_with_edges_keep_neurons():
    ann, nt, edges = _tables()
    g = B.build_graph(ann, nt, edges)
    flipped = B.with_sign(g, -g.sign)
    assert list(flipped.ival) == [-x for x in g.ival]
    rew = B.with_edges(g, np.array([0, 1]), np.array([1, 2]), np.array([5, 3]))
    assert rew.n == g.n and int(rew.crow[-1]) == 2
```

`tests/test_neurons.py`:
```python
import pandas as pd

from lfg_fly.connectome import build as B
from lfg_fly.connectome import neurons as N


def test_populations_select_by_type_and_superclass():
    ann = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5, 6, 7],
        "status": ["Traced"] * 7,
        "superclass": ["ol_sensory", "ol_sensory", "cb_sensory", "vnc_sensory",
                       "descending_neuron", "cb_motor", "cb_intrinsic"],
        "type": ["R1-R6", "R8y", "ORN_DA1", "SNxx", "DNa01", "MNx", "KC"],
        "rootSide": ["L", "R", None, None, None, None, None],
    })
    nt = pd.DataFrame({"body": [1, 2, 3], "consensus_nt": ["histamine", "histamine", "acetylcholine"]})
    edges = pd.DataFrame({"body_pre": [3], "body_post": [5], "weight": [3]})
    g = B.build_graph(ann, nt, edges)
    p = N.populations(g)
    assert p.photoreceptor.tolist() == [0, 1]
    assert p.photoreceptor_type.tolist() == ["R1-R6", "R8y"]
    assert p.orn.tolist() == [2] and p.orn_glomerulus.tolist() == ["DA1"]
    assert p.readout.tolist() == [4, 5]
    assert p.sensory_brain.tolist() == [0, 1, 2]
    assert p.sensory_vnc.tolist() == [3]
    assert p.sensory_any.tolist() == [True, True, True, True, False, False, False]
    assert N.CHANNEL_BY_TYPE["R8y"] == "G" and N.CHANNEL_BY_TYPE["R1-R6"] == "lum"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_build.py tests/test_neurons.py`
Expected: FAIL (import errors).

- [ ] **Step 3: Implement**

`lfg_fly/connectome/build.py`:
```python
"""The connectome as an integer-valued CSR (rows = post, cols = pre).

Weights are stored as synapse COUNTS plus a per-neuron sign. The simulator
multiplies spikes by sign*count (exact integers in fp32, because every input sum
is < 2^24) and only then scales each row by g_syn / insum. That keeps spiking
runs bitwise reproducible on the GPU, whatever order cuSPARSE reduces in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,  # photoreceptors: without the tonic bias they can only inhibit
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
    "unclear": 1,
}
ANN_COLUMNS = ["bodyId", "status", "superclass", "type", "rootSide", "somaSide",
               "assignedOlHex1", "assignedOlHex2"]
_STR = "<U48"


@dataclass(frozen=True)
class Graph:
    body_id: np.ndarray
    superclass: np.ndarray
    cell_type: np.ndarray
    root_side: np.ndarray
    sign: np.ndarray
    crow: np.ndarray
    col: np.ndarray
    count: np.ndarray
    insum: np.ndarray
    min_syn: int
    nt_missing: int

    @property
    def n(self) -> int:
        return len(self.body_id)

    @property
    def ival(self) -> np.ndarray:
        return self.sign[self.col].astype(np.float32) * self.count.astype(np.float32)

    def graph_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.crow, self.col, self.count, self.sign):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(self.min_syn).encode())
        return h.hexdigest()


def load_annotations(path: Path) -> pd.DataFrame:
    return pd.read_feather(path, columns=ANN_COLUMNS)


def load_nt(path: Path) -> pd.DataFrame:
    return pd.read_feather(path, columns=["body", "consensus_nt"])


def load_edges(path: Path, min_syn: int) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.feather as pf

    table = pf.read_table(path, columns=["body_pre", "body_post", "weight"])
    if min_syn > 1:
        table = table.filter(pc.greater_equal(table["weight"], min_syn))
    return table.to_pandas()


def signs_from_nt(bodies: np.ndarray, nt: pd.DataFrame) -> tuple[np.ndarray, int]:
    series = pd.Series(nt["consensus_nt"].to_numpy(), index=nt["body"].to_numpy())
    series = series[~series.index.duplicated(keep="first")]
    values = series.reindex(bodies)
    missing = int(values.isna().sum())
    unknown = sorted(set(values.dropna().unique()) - set(SIGN))
    if unknown:
        raise ValueError(f"unknown consensus_nt values: {unknown}")
    return values.fillna("unclear").map(SIGN).to_numpy(dtype=np.int8), missing


def csr_from_edges(pre, post, count, n: int, device: str = "cpu"):
    import torch

    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)
    count = np.asarray(count)
    keys = torch.as_tensor(post, device=device) * n + torch.as_tensor(pre, device=device)
    order = torch.argsort(keys).cpu().numpy()
    pre, post, count = pre[order], post[order], count[order]
    crow = np.zeros(n + 1, dtype=np.int64)
    crow[1:] = np.cumsum(np.bincount(post, minlength=n))
    insum = np.bincount(post, weights=count, minlength=n).astype(np.float64)
    insum[insum == 0] = 1.0
    if insum.max() >= 2**24:
        raise ValueError("an input sum reaches 2^24: fp32 integer sums would stop being exact")
    return crow, pre, count.astype(np.int32), insum


def _strings(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna("").astype(str).to_numpy(dtype=_STR)


def build_graph(ann: pd.DataFrame, nt: pd.DataFrame, edges: pd.DataFrame, *,
                min_syn: int = 3, device: str = "cpu") -> Graph:
    traced = ann[ann["status"] == "Traced"].reset_index(drop=True)
    bodies = traced["bodyId"].to_numpy(dtype=np.int64)
    index = pd.Index(bodies)
    kept = edges[edges["weight"] >= min_syn]
    pre = index.get_indexer(kept["body_pre"].to_numpy())
    post = index.get_indexer(kept["body_post"].to_numpy())
    ok = (pre >= 0) & (post >= 0)
    sign, missing = signs_from_nt(bodies, nt)
    crow, col, count, insum = csr_from_edges(
        pre[ok], post[ok], kept["weight"].to_numpy()[ok], len(bodies), device
    )
    return Graph(
        body_id=bodies,
        superclass=_strings(traced, "superclass"),
        cell_type=_strings(traced, "type"),
        root_side=_strings(traced, "rootSide"),
        sign=sign,
        crow=crow,
        col=col,
        count=count,
        insum=insum,
        min_syn=min_syn,
        nt_missing=missing,
    )


def with_edges(g: Graph, pre, post, count, device: str = "cpu") -> Graph:
    crow, col, cnt, insum = csr_from_edges(pre, post, count, g.n, device)
    return replace(g, crow=crow, col=col, count=cnt, insum=insum)


def with_sign(g: Graph, sign: np.ndarray) -> Graph:
    return replace(g, sign=np.asarray(sign, dtype=np.int8))


def edge_list(g: Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pre, post, count) of every edge, in CSR order."""
    post = np.repeat(np.arange(g.n, dtype=np.int64), np.diff(g.crow))
    return g.col.copy(), post, g.count.copy()


_ARRAYS = ("body_id", "superclass", "cell_type", "root_side", "sign", "crow", "col", "count", "insum")


def save_graph(g: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"min_syn": g.min_syn, "nt_missing": g.nt_missing, "hash": g.graph_hash()})
    np.savez(path, meta=np.array(meta), **{name: getattr(g, name) for name in _ARRAYS})


def load_graph(path: Path) -> Graph:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        g = Graph(min_syn=int(meta["min_syn"]), nt_missing=int(meta["nt_missing"]),
                  **{name: z[name] for name in _ARRAYS})
    if g.graph_hash() != meta["hash"]:
        raise ValueError(f"{path}: graph hash mismatch")
    return g
```

`lfg_fly/connectome/neurons.py`:
```python
"""Named neuron populations of a built Graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfg_fly.connectome.build import Graph

PHOTORECEPTOR_TYPES = ("R1-R6", "R7p", "R7y", "R7d", "R7_unclear",
                       "R8p", "R8y", "R8d", "R8_unclear", "R7R8_unclear")
# R1-R6 broad luminance; R8y green; R8p blue; R7* short-wavelength (blue stands in for UV).
CHANNEL_BY_TYPE = {
    "R1-R6": "lum", "R7p": "B", "R7y": "B", "R7d": "B", "R7_unclear": "B",
    "R8p": "B", "R8y": "G", "R8d": "lum", "R8_unclear": "lum", "R7R8_unclear": "lum",
}
READOUT_SUPERCLASSES = ("descending_neuron", "vnc_motor", "cb_motor")
BRAIN_SENSORY = ("ol_sensory", "cb_sensory")
VNC_SENSORY = ("vnc_sensory",)


@dataclass(frozen=True)
class Populations:
    readout: np.ndarray
    photoreceptor: np.ndarray
    photoreceptor_type: np.ndarray
    orn: np.ndarray
    orn_glomerulus: np.ndarray
    sensory_brain: np.ndarray
    sensory_vnc: np.ndarray
    sensory_any: np.ndarray


def populations(g: Graph) -> Populations:
    sc, t = g.superclass, g.cell_type
    pr = np.flatnonzero(np.isin(t, PHOTORECEPTOR_TYPES))
    orn = np.flatnonzero(np.char.startswith(t, "ORN_"))
    return Populations(
        readout=np.flatnonzero(np.isin(sc, READOUT_SUPERCLASSES)),
        photoreceptor=pr,
        photoreceptor_type=t[pr],
        orn=orn,
        orn_glomerulus=np.char.replace(t[orn], "ORN_", "", count=1),
        sensory_brain=np.flatnonzero(np.isin(sc, BRAIN_SENSORY)),
        sensory_vnc=np.flatnonzero(np.isin(sc, VNC_SENSORY)),
        # exactly the superclasses an input code can drive; everything else gets the tonic bias
        # (a substring match would also catch sensory_ascending/_descending/*_tbc: 565 neurons on v1.0)
        sensory_any=np.isin(sc, BRAIN_SENSORY + VNC_SENSORY),
    )
```

Add to `lfg_fly/cli.py` `build_parser`:
```python
    p = sub.add_parser("build", help="build the thresholded integer-CSR graph")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_build)
```
and:
```python
def _cmd_build(args: argparse.Namespace) -> int:
    from lfg_fly import env, paths
    from lfg_fly.connectome import build as B
    from lfg_fly.connectome.fetch import FILES

    env.configure_libraries()
    raw = paths.raw_dir()
    ann = B.load_annotations(raw / FILES["annotations"])
    nt = B.load_nt(raw / FILES["neurotransmitters"])
    edges = B.load_edges(raw / FILES["weights"], args.min_syn)
    g = B.build_graph(ann, nt, edges, min_syn=args.min_syn, device=args.device)
    out = paths.graph_dir() / f"graph-syn{args.min_syn}.npz"
    B.save_graph(g, out)
    print(f"{g.n:,} neurons, {len(g.col):,} edges (>= {g.min_syn}), NT missing {g.nt_missing}, "
          f"hash {g.graph_hash()[:16]} -> {out}")
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_build.py tests/test_neurons.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/connectome/build.py lfg_fly/connectome/neurons.py lfg_fly/cli.py tests/test_build.py tests/test_neurons.py
git commit -m "feat(connectome): integer-CSR graph with sign map, NT-missing count, populations"
```

---

### Task 4: Photoreceptor columns

**Files:**
- Create: `lfg_fly/connectome/columns.py`
- Modify: `lfg_fly/cli.py` (add `columns`)
- Test: `tests/test_columns.py`

**Interfaces:**
- Consumes: `Graph`, `Populations`, `load_annotations`
- Produces:
  - `class Columns`, a frozen dataclass with fields `neuron: int64[P]`, `eye: <U1[P]`, `q: int32[P]`, `r: int32[P]`, `share: float32[P]`, `dropped: int`, `total: int`
  - `MIN_ASSIGNED = 0.90`
  - `assign_columns(g, pops, ann, edges_any) -> Columns`
  - `load_pr_edges(path, pr_bodies) -> DataFrame`
  - `lattice_xy(cols) -> tuple[float64[P], float64[P]]` (each in [0,1], normalized per eye)
  - `save_columns(cols, path)` / `load_columns(path) -> Columns`

- [ ] **Step 1: Write the failing test**

`tests/test_columns.py`:
```python
import numpy as np
import pandas as pd
import pytest

from lfg_fly.connectome import build as B
from lfg_fly.connectome import columns as C
from lfg_fly.connectome import neurons as N


def _world():
    ann = pd.DataFrame({
        "bodyId": [1, 2, 3, 11, 12, 13],
        "status": ["Traced"] * 6,
        "superclass": ["ol_sensory"] * 3 + ["ol_intrinsic"] * 3,
        "type": ["R1-R6", "R8y", "R7p", "L1", "L2", "L3"],
        "rootSide": ["L", "R", "R", None, None, None],
        "somaSide": [None, None, None, "L", "R", "R"],
        "assignedOlHex1": [np.nan, np.nan, np.nan, 3.0, 5.0, 6.0],
        "assignedOlHex2": [np.nan, np.nan, np.nan, 4.0, 1.0, 1.0],
    })
    nt = pd.DataFrame({"body": [1, 2, 3], "consensus_nt": ["histamine"] * 3})
    # PR1 -> L1 (col L 3,4) x9, -> L2 (col R 5,1) x1 ; PR2 -> L2 x4, L3 x4 (tie -> sort order) ; PR3 -> none
    edges_any = pd.DataFrame({
        "body_pre": [1, 1, 2, 2],
        "body_post": [11, 12, 12, 13],
        "weight": [9, 1, 4, 4],
    })
    g = B.build_graph(ann, nt, edges_any, min_syn=1)
    return g, N.populations(g), ann, edges_any


def test_assigns_majority_column_and_drops_orphans(monkeypatch):
    g, pops, ann, edges = _world()
    monkeypatch.setattr(C, "MIN_ASSIGNED", 0.5)
    cols = C.assign_columns(g, pops, ann, edges)
    assert cols.total == 3 and cols.dropped == 1
    assert cols.neuron.tolist() == [0, 1]
    assert cols.eye.tolist() == ["L", "R"]
    assert (cols.q.tolist(), cols.r.tolist()) == ([3, 5], [4, 1])  # PR2 tie: (R,5,1) sorts first
    assert cols.share[0] == pytest.approx(0.9)


def test_too_few_assigned_fails():
    g, pops, ann, edges = _world()
    with pytest.raises(ValueError, match="assigned"):
        C.assign_columns(g, pops, ann, edges)  # 2/3 < 0.90


def test_lattice_is_normalized_per_eye(monkeypatch):
    cols = C.Columns(neuron=np.array([0, 1, 2, 3]), eye=np.array(["L", "L", "R", "R"]),
                     q=np.array([0, 2, 10, 10]), r=np.array([0, 0, 0, 4]),
                     share=np.ones(4, np.float32), dropped=0, total=4)
    x, y = C.lattice_xy(cols)
    assert x.tolist()[:2] == [0.0, 1.0]
    assert y.tolist()[2:] == [0.0, 1.0]
    assert x.min() >= 0 and x.max() <= 1 and y.min() >= 0 and y.max() <= 1


def test_roundtrip(tmp_path, monkeypatch):
    g, pops, ann, edges = _world()
    monkeypatch.setattr(C, "MIN_ASSIGNED", 0.5)
    cols = C.assign_columns(g, pops, ann, edges)
    C.save_columns(cols, tmp_path / "c.npz")
    back = C.load_columns(tmp_path / "c.npz")
    assert back.neuron.tolist() == cols.neuron.tolist() and back.dropped == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_columns.py`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`lfg_fly/connectome/columns.py`:
```python
"""Give every photoreceptor an ommatidial column (spec §2, Eyes).

No photoreceptor carries assignedOlHex in v1.0; only columnar ol_intrinsic cells
do (L1-L3, L5, C2, C3, Mi1/4/9, Tm1/2/4/9/20, T1). A photoreceptor takes the
column (somaSide, hex1, hex2) receiving most of its output synapses among
hex-labelled cells, at ANY synapse weight. Its eye is that column's somaSide
(fallback: its own rootSide).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lfg_fly.connectome.build import Graph
from lfg_fly.connectome.neurons import Populations

MIN_ASSIGNED = 0.90


@dataclass(frozen=True)
class Columns:
    neuron: np.ndarray
    eye: np.ndarray
    q: np.ndarray
    r: np.ndarray
    share: np.ndarray
    dropped: int
    total: int


def load_pr_edges(path: Path, pr_bodies: np.ndarray) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as pf

    table = pf.read_table(path, columns=["body_pre", "body_post", "weight"])
    table = table.filter(pc.is_in(table["body_pre"], value_set=pa.array(pr_bodies, type=pa.int64())))
    return table.to_pandas()


def assign_columns(g: Graph, pops: Populations, ann: pd.DataFrame, edges_any: pd.DataFrame) -> Columns:
    hexed = ann[(ann["status"] == "Traced") & ann["assignedOlHex1"].notna() & ann["assignedOlHex2"].notna()]
    column_of = pd.DataFrame({
        "body_post": hexed["bodyId"].to_numpy(dtype=np.int64),
        "side": hexed["somaSide"].fillna("").astype(str).to_numpy(),
        "q": hexed["assignedOlHex1"].astype(int).to_numpy(),
        "r": hexed["assignedOlHex2"].astype(int).to_numpy(),
    })
    pr_bodies = g.body_id[pops.photoreceptor]
    e = edges_any[edges_any["body_pre"].isin(pr_bodies)].merge(column_of, on="body_post", how="inner")
    agg = e.groupby(["body_pre", "side", "q", "r"], as_index=False)["weight"].sum()
    agg["share"] = agg["weight"] / agg.groupby("body_pre")["weight"].transform("sum")
    best = agg.sort_values(["body_pre", "weight", "side", "q", "r"],
                           ascending=[True, False, True, True, True]).drop_duplicates("body_pre")
    idx = pd.Index(g.body_id).get_indexer(best["body_pre"].to_numpy())
    side = best["side"].to_numpy().astype("<U1")
    eye = np.where(np.isin(side, ["L", "R"]), side, g.root_side[idx].astype("<U1"))
    ok = np.isin(eye, ["L", "R"])
    total = len(pops.photoreceptor)
    assigned = int(ok.sum())
    if assigned < MIN_ASSIGNED * total:
        raise ValueError(f"only {assigned}/{total} photoreceptors assigned a column (< {MIN_ASSIGNED:.0%})")
    order = np.argsort(idx[ok])
    return Columns(
        neuron=idx[ok][order].astype(np.int64),
        eye=eye[ok][order],
        q=best["q"].to_numpy()[ok][order].astype(np.int32),
        r=best["r"].to_numpy()[ok][order].astype(np.int32),
        share=best["share"].to_numpy()[ok][order].astype(np.float32),
        dropped=total - assigned,
        total=total,
    )


def lattice_xy(cols: Columns) -> tuple[np.ndarray, np.ndarray]:
    """Axial hex (q, r) -> (q + r/2, r*sqrt(3)/2), min-max normalized within each eye."""
    x = cols.q + cols.r / 2.0
    y = cols.r * (np.sqrt(3.0) / 2.0)
    xn, yn = np.empty_like(x, dtype=np.float64), np.empty_like(y, dtype=np.float64)
    for side in ("L", "R"):
        m = cols.eye == side
        if not m.any():
            continue
        for src, dst in ((x, xn), (y, yn)):
            lo, hi = src[m].min(), src[m].max()
            dst[m] = 0.5 if hi == lo else (src[m] - lo) / (hi - lo)
    return xn, yn


def save_columns(cols: Columns, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, neuron=cols.neuron, eye=cols.eye, q=cols.q, r=cols.r, share=cols.share,
             dropped=np.array(cols.dropped), total=np.array(cols.total))


def load_columns(path: Path) -> Columns:
    with np.load(path, allow_pickle=False) as z:
        return Columns(neuron=z["neuron"], eye=z["eye"], q=z["q"], r=z["r"], share=z["share"],
                       dropped=int(z["dropped"]), total=int(z["total"]))
```

Add to `lfg_fly/cli.py`:
```python
    p = sub.add_parser("columns", help="assign photoreceptor columns from synaptic partners")
    p.add_argument("--min-syn", type=int, default=3)
    p.set_defaults(func=_cmd_columns)
```
```python
def _cmd_columns(args: argparse.Namespace) -> int:
    from lfg_fly import env, paths
    from lfg_fly.connectome import build as B
    from lfg_fly.connectome import columns as C
    from lfg_fly.connectome.fetch import FILES
    from lfg_fly.connectome.neurons import populations

    env.configure_libraries()
    g = B.load_graph(paths.graph_dir() / f"graph-syn{args.min_syn}.npz")
    pops = populations(g)
    ann = B.load_annotations(paths.raw_dir() / FILES["annotations"])
    edges = C.load_pr_edges(paths.raw_dir() / FILES["weights"], g.body_id[pops.photoreceptor])
    cols = C.assign_columns(g, pops, ann, edges)
    C.save_columns(cols, paths.graph_dir() / f"columns-syn{args.min_syn}.npz")
    print(f"assigned {cols.total - cols.dropped}/{cols.total} photoreceptors; "
          f"median top-column share {float(np.median(cols.share)):.2f}")
    return 0
```
(add `import numpy as np` inside `_cmd_columns`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_columns.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/connectome/columns.py lfg_fly/cli.py tests/test_columns.py
git commit -m "feat(connectome): partner-derived photoreceptor columns and per-eye lattice"
```

---

### Task 5: Simulator

**Files:**
- Create: `lfg_fly/brain/sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: `Graph` (`crow`, `col`, `ival`, `insum`, `n`), `Populations.sensory_any`, `Populations.readout`
- Produces:
  - `BRAIN_KINDS = ("lif", "lif-avg", "lif-volley", "rate")`
  - `ACTIVE_RATE = 0.05`, `SATURATED_RATE = 9.0`
  - `class BrainParams`, a frozen dataclass: `kind: str`, `g_syn: float`, `bias: float = 0.0`, `steps: int = 60`, `tau_ms: float = 20.0`, `refractory: int = 2`, `trials: int = 1`, `noise: float = 0.0`, `alpha: float = 0.7`, `burn_in: int = 200`, `seed: int = 0`. It validates in `__post_init__`, and `BrainParams.as_dict()` returns a plain dict.
  - `class SimResult`, a dataclass with fields:
    - `features: Tensor[B, R]`
    - `active_frac: Tensor[B]`, `readout_active_frac: Tensor[B]`
    - `max_step_frac: float`
    - `neurons_fired: Tensor[B]`, `total_spikes: Tensor[B]`, `readout_spikes: Tensor[B]`
  - `class Simulator(g, sensory_any: bool[N], readout: int64[R], device: str)` with:
    - `.n`, `.device`
    - `.rest(p, rest_drive: Tensor[N,1], key: str) -> dict[str, Tensor]`
    - `.run(drive: Tensor[N,B], p, rest_drive: Tensor[N,1], rest_key: str) -> SimResult`
  - `simulate_reference(g, sensory_any, readout, drive: np.ndarray[N,B], p, rest_drive: np.ndarray[N,1]) -> np.ndarray[B,R]` (numpy, noise-free kinds only)

- [ ] **Step 1: Write the failing tests**

`tests/test_sim.py`:
```python
import numpy as np
import pandas as pd
import pytest
import torch

from lfg_fly.brain import sim as S
from lfg_fly.connectome import build as B


def _graph(pre, post, weight, superclass, nts):
    n = len(superclass)
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": superclass, "type": ["t"] * n, "rootSide": [None] * n})
    nt = pd.DataFrame({"body": np.arange(1, n + 1), "consensus_nt": nts})
    edges = pd.DataFrame({"body_pre": np.asarray(pre) + 1, "body_post": np.asarray(post) + 1,
                          "weight": weight})
    return B.build_graph(ann, nt, edges, min_syn=1)


def _random_graph(n=40, m=300, seed=0):
    rng = np.random.default_rng(seed)
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = pre != post
    pairs = pd.DataFrame({"pre": pre[keep], "post": post[keep]}).drop_duplicates()
    sc = ["cb_sensory"] * 5 + ["cb_intrinsic"] * (n - 10) + ["descending_neuron"] * 5
    nts = ["acetylcholine" if i % 4 else "gaba" for i in range(n)]
    return _graph(pairs.pre.to_numpy(), pairs.post.to_numpy(), rng.integers(1, 9, len(pairs)), sc, nts)


def _masks(g):
    sensory = np.char.find(g.superclass, "sensory") >= 0
    readout = np.flatnonzero(g.superclass == "descending_neuron")
    return sensory, readout


@pytest.mark.parametrize("kind,steps", [("lif", 30), ("lif-volley", 6)])
def test_lif_matches_numpy_reference_exactly(kind, steps):
    g = _random_graph()
    sensory, readout = _masks(g)
    p = S.BrainParams(kind=kind, g_syn=3.0, bias=0.06, steps=steps, burn_in=20)
    rng = np.random.default_rng(1)
    drive = np.zeros((g.n, 7), np.float32)
    drive[:5] = rng.uniform(0, 1.5, (5, 7))
    rest = np.zeros((g.n, 1), np.float32)
    ref = S.simulate_reference(g, sensory, readout, drive, p, rest)
    sim = S.Simulator(g, sensory, readout, device="cpu")
    got = sim.run(torch.as_tensor(drive), p, torch.as_tensor(rest), "zero").features.numpy()
    assert np.array_equal(got, ref)


def test_rate_matches_numpy_reference_to_tolerance():
    g = _random_graph()
    sensory, readout = _masks(g)
    p = S.BrainParams(kind="rate", g_syn=1.0, bias=0.1, steps=30, burn_in=20)
    drive = np.random.default_rng(2).uniform(0, 1, (g.n, 4)).astype(np.float32)
    drive[5:] = 0
    rest = np.zeros((g.n, 1), np.float32)
    ref = S.simulate_reference(g, sensory, readout, drive, p, rest)
    got = S.Simulator(g, sensory, readout, "cpu").run(
        torch.as_tensor(drive), p, torch.as_tensor(rest), "zero").features.numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)


def test_runs_are_deterministic():
    g = _random_graph()
    sensory, readout = _masks(g)
    sim = S.Simulator(g, sensory, readout, "cpu")
    p = S.BrainParams(kind="lif-avg", g_syn=3.0, bias=0.05, trials=4, noise=0.05, seed=9)
    d = torch.rand(g.n, 3)
    rest = torch.zeros(g.n, 1)
    a = sim.run(d, p, rest, "z").features
    b = sim.run(d, p, rest, "z").features
    assert torch.equal(a, b)


def test_refractory_limits_rate():
    # one self-driven neuron with huge input fires every (refractory+1) steps
    g = _graph([0], [1], [1], ["cb_sensory", "descending_neuron"], ["acetylcholine"] * 2)
    sensory, readout = _masks(g)
    p = S.BrainParams(kind="lif", g_syn=0.0, steps=30, burn_in=0, refractory=2)
    drive = torch.zeros(2, 1)
    drive[1] = 5.0  # the readout neuron is driven directly
    res = S.Simulator(g, sensory, readout, "cpu").run(drive, p, torch.zeros(2, 1), "z")
    assert res.features.item() == 10  # 30 steps / (2 refractory + 1)


def test_tonic_bias_lets_inhibitory_eyes_reach_downstream():
    # photoreceptor P (sensory, histamine) -| L (intrinsic) -> D (readout)
    g = _graph([0, 1], [1, 2], [5, 5], ["ol_sensory", "ol_intrinsic", "descending_neuron"],
               ["histamine", "acetylcholine", "acetylcholine"])
    sensory, readout = _masks(g)
    sim = S.Simulator(g, sensory, readout, "cpu")
    rest = torch.zeros(3, 1)
    dark, light = torch.zeros(3, 1), torch.zeros(3, 1)
    light[0] = 1.5
    for bias, should_differ in ((0.0, False), (0.12, True)):
        p = S.BrainParams(kind="lif", g_syn=2.0, bias=bias, steps=40, burn_in=50)
        a = sim.run(dark, p, rest, f"b{bias}").features
        b = sim.run(light, p, rest, f"b{bias}").features
        assert (not torch.equal(a, b)) == should_differ


def test_stats_are_fractions():
    g = _random_graph()
    sensory, readout = _masks(g)
    res = S.Simulator(g, sensory, readout, "cpu").run(
        torch.rand(g.n, 5), S.BrainParams(kind="lif", g_syn=3.0, bias=0.05), torch.zeros(g.n, 1), "z")
    assert res.active_frac.shape == (5,) and float(res.active_frac.max()) <= 1.0
    assert 0.0 <= res.max_step_frac <= 1.0
    assert res.readout_spikes.shape == (5,)


def test_params_validate():
    with pytest.raises(ValueError):
        S.BrainParams(kind="banana", g_syn=1.0)
    with pytest.raises(ValueError):
        S.BrainParams(kind="lif-avg", g_syn=1.0, trials=1, noise=0.05)
    with pytest.raises(ValueError):
        S.BrainParams(kind="lif", g_syn=1.0, trials=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_matches_reference_exactly():
    g = _random_graph(seed=5)
    sensory, readout = _masks(g)
    p = S.BrainParams(kind="lif", g_syn=3.0, bias=0.06, steps=40, burn_in=20)
    drive = np.random.default_rng(3).uniform(0, 1.5, (g.n, 9)).astype(np.float32)
    rest = np.zeros((g.n, 1), np.float32)
    ref = S.simulate_reference(g, sensory, readout, drive, p, rest)
    got = S.Simulator(g, sensory, readout, "cuda").run(
        torch.as_tensor(drive, device="cuda"), p, torch.as_tensor(rest, device="cuda"), "z"
    ).features.cpu().numpy()
    assert np.array_equal(got, ref)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_sim.py`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`lfg_fly/brain/sim.py`:
```python
"""Candidate brains on the connectome (spec §2, Candidate brains).

All kinds share: dt = 1 ms, the integer CSR (syn = (W_int @ x) * (1/insum)),
a tonic bias on every NON-sensory neuron, and a cached resting state reached by
a noise-free burn-in under `rest_drive` (the mid-grey retina, no odour).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from lfg_fly.connectome.build import Graph

BRAIN_KINDS = ("lif", "lif-avg", "lif-volley", "rate")
ACTIVE_RATE = 0.05  # a rate unit counts as "active" above this
SATURATED_RATE = 9.0  # the rate analogue of "firing" in the runaway rule (clip is 10)


@dataclass(frozen=True)
class BrainParams:
    kind: str
    g_syn: float
    bias: float = 0.0
    steps: int = 60
    tau_ms: float = 20.0
    refractory: int = 2
    trials: int = 1
    noise: float = 0.0
    alpha: float = 0.7
    burn_in: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in BRAIN_KINDS:
            raise ValueError(f"unknown brain kind {self.kind!r}")
        if self.kind == "lif-avg":
            if self.trials < 2 or self.noise <= 0:
                raise ValueError("lif-avg needs trials >= 2 and noise > 0")
        elif self.trials != 1 or self.noise != 0:
            raise ValueError(f"{self.kind} takes trials=1 and noise=0")

    @property
    def spiking(self) -> bool:
        return self.kind != "rate"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimResult:
    features: torch.Tensor
    active_frac: torch.Tensor
    readout_active_frac: torch.Tensor
    max_step_frac: float
    neurons_fired: torch.Tensor
    total_spikes: torch.Tensor
    readout_spikes: torch.Tensor


class Simulator:
    def __init__(self, g: Graph, sensory_any: np.ndarray, readout: np.ndarray, device: str = "cuda"):
        self.n = g.n
        self.device = device
        self.W = torch.sparse_csr_tensor(
            torch.as_tensor(g.crow, dtype=torch.int64),
            torch.as_tensor(g.col, dtype=torch.int64),
            torch.as_tensor(g.ival, dtype=torch.float32),
            size=(g.n, g.n),
        ).to(device)
        self.scale = torch.as_tensor((1.0 / g.insum).astype(np.float32), device=device).unsqueeze(1)
        self.nonsensory = torch.as_tensor(~np.asarray(sensory_any), device=device).float().unsqueeze(1)
        self.readout = torch.as_tensor(np.asarray(readout, dtype=np.int64), device=device)
        self._rest: dict[tuple, dict[str, torch.Tensor]] = {}

    def _syn(self, x: torch.Tensor) -> torch.Tensor:
        return (self.W @ x) * self.scale

    def _lif(self, v, ref, s, drive, p: BrainParams, steps: int, gen, record: bool):
        decay = math.exp(-1.0 / p.tau_ms)
        count = torch.zeros_like(v)
        max_frac = 0.0
        zero = torch.zeros((), device=v.device)
        for _ in range(steps):
            x = drive
            if gen is not None:
                x = x + p.noise * torch.randn(v.shape, generator=gen, device=v.device)
            v = v * decay + p.g_syn * self._syn(s) + x
            v = torch.where(ref > 0, zero, v)
            s = (v >= 1.0).float()
            v = torch.where(s > 0, zero, v)
            ref = torch.clamp(ref - 1.0, min=0.0) + p.refractory * s
            count += s
            if record:
                max_frac = max(max_frac, float(s.mean(0).max()))
        return v, ref, s, count, max_frac

    def _rate(self, h, drive, p: BrainParams, steps: int, record: bool):
        max_frac = 0.0
        for _ in range(steps):
            h = (1 - p.alpha) * h + p.alpha * torch.clamp(torch.relu(p.g_syn * self._syn(h) + drive), 0.0, 10.0)
            if record:
                max_frac = max(max_frac, float((h > SATURATED_RATE).float().mean(0).max()))
        return h, max_frac

    def rest(self, p: BrainParams, rest_drive: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
        cache_key = (p.kind, p.g_syn, p.bias, p.tau_ms, p.refractory, p.alpha, p.burn_in, key)
        if cache_key not in self._rest:
            drive = rest_drive + p.bias * self.nonsensory
            if p.spiking:
                z = torch.zeros(self.n, 1, device=self.device)
                v, ref, s, _, _ = self._lif(z, z.clone(), z.clone(), drive, p, p.burn_in, None, False)
                self._rest[cache_key] = {"v": v, "ref": ref, "s": s}
            else:
                h, _ = self._rate(torch.zeros(self.n, 1, device=self.device), drive, p, p.burn_in, False)
                self._rest[cache_key] = {"h": h}
        return self._rest[cache_key]

    @torch.no_grad()
    def run(self, drive: torch.Tensor, p: BrainParams, rest_drive: torch.Tensor, rest_key: str) -> SimResult:
        state = self.rest(p, rest_drive, rest_key)
        b = drive.shape[1]
        full = drive + p.bias * self.nonsensory
        if p.spiking:
            r = p.trials
            full = full.repeat(1, r)  # [N, R*B] as R blocks of B
            v, ref, s = (state[k].expand(self.n, r * b).clone() for k in ("v", "ref", "s"))
            gen = None
            if p.noise > 0:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(p.seed)
            _, _, _, count, max_frac = self._lif(v, ref, s, full, p, p.steps, gen, True)
            count = count.view(self.n, r, b).mean(1)
            ro = count[self.readout]
            active = (count > 0).float()
            return SimResult(
                features=ro.T.contiguous(),
                active_frac=active.mean(0),
                readout_active_frac=(ro > 0).float().mean(0),
                max_step_frac=max_frac,
                neurons_fired=active.sum(0).long(),
                total_spikes=count.sum(0),
                readout_spikes=ro.sum(0),
            )
        h = state["h"].expand(self.n, b).clone()
        h, max_frac = self._rate(h, full, p, p.steps, True)
        ro = h[self.readout]
        active = (h > ACTIVE_RATE).float()
        return SimResult(
            features=ro.T.contiguous(),
            active_frac=active.mean(0),
            readout_active_frac=(ro > ACTIVE_RATE).float().mean(0),
            max_step_frac=max_frac,
            neurons_fired=active.sum(0).long(),
            total_spikes=h.sum(0),
            readout_spikes=ro.sum(0),
        )


def simulate_reference(g: Graph, sensory_any: np.ndarray, readout: np.ndarray, drive: np.ndarray,
                       p: BrainParams, rest_drive: np.ndarray) -> np.ndarray:
    """Dense numpy twin of Simulator.run for noise-free kinds (tests only; tiny graphs)."""
    if p.noise:
        raise ValueError("the reference covers noise-free kinds only")
    n = g.n
    W = np.zeros((n, n), np.float32)
    post = np.repeat(np.arange(n), np.diff(g.crow))
    W[post, g.col] = g.ival
    scale = (1.0 / g.insum).astype(np.float32)[:, None]
    bias = (p.bias * (~np.asarray(sensory_any))).astype(np.float32)[:, None]
    syn = lambda x: (W @ x).astype(np.float32) * scale  # noqa: E731

    if p.spiking:
        decay = np.float32(math.exp(-1.0 / p.tau_ms))

        def lif(v, ref, s, d, steps):
            count = np.zeros_like(v)
            for _ in range(steps):
                v = v * decay + np.float32(p.g_syn) * syn(s) + d
                v = np.where(ref > 0, np.float32(0), v).astype(np.float32)
                s = (v >= 1.0).astype(np.float32)
                v = np.where(s > 0, np.float32(0), v).astype(np.float32)
                ref = np.maximum(ref - 1.0, 0.0).astype(np.float32) + np.float32(p.refractory) * s
                count += s
            return v, ref, s, count

        z = np.zeros((n, 1), np.float32)
        v, ref, s, _ = lif(z, z.copy(), z.copy(), rest_drive + bias, p.burn_in)
        b = drive.shape[1]
        v, ref, s = (np.repeat(x, b, axis=1) for x in (v, ref, s))
        _, _, _, count = lif(v, ref, s, drive + bias, p.steps)
        return count[readout].T

    def rate(h, d, steps):
        for _ in range(steps):
            h = ((1 - p.alpha) * h + p.alpha * np.clip(np.maximum(p.g_syn * syn(h) + d, 0), 0, 10)).astype(np.float32)
        return h

    h = rate(np.zeros((n, 1), np.float32), rest_drive + bias, p.burn_in)
    h = rate(np.repeat(h, drive.shape[1], axis=1), drive + bias, p.steps)
    return h[readout].T
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_sim.py`
Expected: PASS, including `test_gpu_matches_reference_exactly` on this box (it's skipped in CI).
If `test_lif_matches_numpy_reference_exactly` shows tiny float differences, fix the implementation, not the test: the float32 operation order in `_lif` and the reference must match (`v*decay`, `+ g*syn`, `+ x`). Exact equality is the whole point.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/brain/sim.py tests/test_sim.py
git commit -m "feat(brain): LIF/lif-avg/lif-volley/rate simulator on integer CSR, tonic bias, resting state"
```

---

### Task 6: Senses and input codes

**Files:**
- Create: `lfg_fly/brain/senses.py`
- Test: `tests/test_senses.py`

**Interfaces:**
- Consumes: `Populations`, `Columns`, `lattice_xy`, `CHANNEL_BY_TYPE`, `Graph.cell_type`
- Produces:
  - `SLOTS: tuple[str, ...]` (the 9 LFG slots in LFG order)
  - `NON_BODY_SLOTS`, `NONE = "None"`, `C_REF = 0.7`
  - `CODE_NAMES = ("nose", "eyes", "eyes+nose", "all-sensory-brain", "all-sensory+vnc")`
  - `trait_seed(tag, slot, value) -> int`
  - `class Code`, a frozen dataclass: `neurons: int64[K]`, `weights: float32[K]`
  - `odor_code(slot, value, orn, orn_glomerulus, k=4) -> Code`
  - `pool_code(slot, value, pool, k=14, tag="allsens/v1") -> Code`
  - `class Retina`, a frozen dataclass: `neurons`, `px`, `py`, `channel` (0 = luminance, 1 = G, 2 = B)
  - `build_retina(cols, g, size=64) -> Retina`
  - `retina_values(images: Tensor[B,3,S,S], retina) -> Tensor[P,B]`
  - `class InputBuilder(name, n, pops, retina, g_in, device)` with:
    - `.components -> tuple[str, ...]`
    - `.uses_eyes`
    - `.g_eye`
    - `.rest_drive() -> Tensor[N,1]`
    - `.rest_key() -> str`
    - `.drive(looks, images=None, conc=None) -> Tensor[N,B]`
    - `.only(component) -> InputBuilder`

  A look is `tuple[str, ...]` aligned with `SLOTS`. `conc` is `np.ndarray[B, 9]` or `None` (meaning `C_REF` everywhere).

- [ ] **Step 1: Write the failing tests**

`tests/test_senses.py`:
```python
import hashlib

import numpy as np
import pytest
import torch

from lfg_fly.brain import senses as Z
from lfg_fly.connectome.columns import Columns
from lfg_fly.connectome.neurons import Populations

ORN = np.arange(100, 160, dtype=np.int64)
GLOM = np.array([f"G{i % 6}" for i in range(60)])


def test_odor_code_golden():
    code = Z.odor_code("Head", "Pirate Hat", ORN, GLOM)
    assert code.neurons.tolist() == [
        103, 109, 115, 121, 127, 133, 139, 145, 151, 157,
        100, 106, 112, 118, 124, 130, 136, 142, 148, 154,
        104, 110, 116, 122, 128, 134, 140, 146, 152, 158,
        102, 108, 114, 120, 126, 132, 138, 144, 150, 156,
    ]
    assert hashlib.sha256(code.weights.tobytes()).hexdigest()[:16] == "53991c51ab6963c6"
    assert code.weights[0] == pytest.approx(0.271866, abs=1e-6)


def test_pool_code_golden():
    code = Z.pool_code("Head", "Pirate Hat", np.arange(1000, 1200, dtype=np.int64))
    assert code.neurons.tolist() == [1194, 1085, 1157, 1168, 1079, 1111, 1070,
                                     1011, 1043, 1072, 1092, 1037, 1029, 1059]
    assert hashlib.sha256(code.weights.tobytes()).hexdigest()[:16] == "50d128800a761497"


def test_codes_separate_values_and_slots():
    a = Z.odor_code("Head", "Crown", ORN, GLOM)
    b = Z.odor_code("Head", "Pirate Hat", ORN, GLOM)
    c = Z.odor_code("Eyes", "Crown", ORN, GLOM)
    assert set(a.neurons.tolist()) != set(b.neurons.tolist()) or not np.allclose(a.weights, b.weights)
    assert Z.trait_seed("odor/v2", "Head", "Crown") != Z.trait_seed("odor/v2", "Eyes", "Crown")
    assert c.neurons.dtype == np.int64


def _pops(n=20):
    return Populations(
        readout=np.array([18, 19]), photoreceptor=np.array([0, 1, 2]),
        photoreceptor_type=np.array(["R1-R6", "R8y", "R7p"]),
        orn=np.arange(3, 9), orn_glomerulus=np.array(["A", "A", "B", "B", "C", "C"]),
        sensory_brain=np.arange(0, 12), sensory_vnc=np.arange(12, 15),
        sensory_any=np.array([True] * 15 + [False] * (n - 15)),
    )


def _retina():
    cols = Columns(neuron=np.array([0, 1, 2]), eye=np.array(["L", "L", "L"]),
                   q=np.array([0, 4, 0]), r=np.array([0, 0, 4]), share=np.ones(3, np.float32),
                   dropped=0, total=3)

    class G:
        cell_type = np.array(["R1-R6", "R8y", "R7p"] + ["x"] * 17)

    return Z.build_retina(cols, G(), size=8)


def test_retina_samples_channels():
    retina = _retina()
    img = torch.zeros(1, 3, 8, 8)
    img[0, 1] = 1.0  # pure green image
    vals = Z.retina_values(img, retina)
    assert vals.shape == (3, 1)
    assert vals[0, 0] == pytest.approx(0.587, abs=1e-5)  # luminance of pure green
    assert vals[1, 0] == pytest.approx(1.0)               # R8y -> G
    assert vals[2, 0] == pytest.approx(0.0)               # R7p -> B


def test_builder_drive_components():
    pops, retina = _pops(), _retina()
    look = ("bg", "None", "b1", "c1", "m1", "e1", "y1", "h1", "a1")
    nose = Z.InputBuilder("nose", 20, pops, retina, g_in=2.0, device="cpu")
    d = nose.drive([look])
    assert d.shape == (20, 1)
    assert torch.allclose(d[[0, 1, 2], 0], torch.full((3,), 0.5))  # grey retina baseline (g_eye=1)
    assert float(d[3:9].sum()) > 0 and float(d[9:].sum()) == 0
    eyes = Z.InputBuilder("eyes", 20, pops, retina, g_in=2.0, device="cpu")
    img = torch.ones(1, 3, 8, 8)
    de = eyes.drive([look], images=img)
    assert float(de[3:9].sum()) == 0 and torch.allclose(de[[0, 1, 2], 0], torch.full((3,), 2.0))
    assert eyes.rest_drive()[0, 0] == pytest.approx(1.0)  # grey 0.5 * g_eye 2.0
    both = Z.InputBuilder("eyes+nose", 20, pops, retina, g_in=2.0, device="cpu")
    assert both.components == ("eyes", "nose")
    assert both.only("nose").name == "nose" and both.only("eyes").name == "eyes"


def test_concentration_scales_identity_drive():
    pops, retina = _pops(), _retina()
    b = Z.InputBuilder("all-sensory+vnc", 20, pops, retina, g_in=1.0, device="cpu")
    look = ("bg", "None", "b1", "c1", "m1", "e1", "y1", "h1", "a1")
    lo = b.drive([look], conc=np.full((1, 9), 0.4))
    hi = b.drive([look], conc=np.full((1, 9), 0.8))
    ident = slice(3, 20)
    assert torch.allclose(hi[ident] - 0.0, 2 * (lo[ident] - 0.0))
    with pytest.raises(ValueError):
        Z.InputBuilder("smell-o-vision", 20, pops, retina, g_in=1.0, device="cpu")
```

Note that `test_concentration_scales_identity_drive` slices out rows 0–2 (the grey retina baseline) before comparing, so only the identity drive has to scale with concentration.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_senses.py`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`lfg_fly/brain/senses.py`:
```python
"""The fly's senses (spec §2, Senses). Fixed; never trained."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from lfg_fly.connectome.columns import Columns, lattice_xy
from lfg_fly.connectome.neurons import CHANNEL_BY_TYPE, Populations

SLOTS = ("Background", "Back", "Body", "Clothing", "Mouth", "Eyebrows", "Eyes", "Head", "Accessory")
NON_BODY_SLOTS = tuple(s for s in SLOTS if s != "Body")
NONE = "None"
C_REF = 0.7
CODE_NAMES = ("nose", "eyes", "eyes+nose", "all-sensory-brain", "all-sensory+vnc")
_CHANNEL_INDEX = {"lum": 0, "G": 1, "B": 2}


def trait_seed(tag: str, slot: str, value: str) -> int:
    digest = hashlib.sha256(f"lfg-fly/{tag}|{slot}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class Code:
    neurons: np.ndarray
    weights: np.ndarray


def odor_code(slot: str, value: str, orn: np.ndarray, orn_glomerulus: np.ndarray, k: int = 4) -> Code:
    """k whole glomeruli (ORNs converge by type), each ORN scaled 1/sqrt(type size)."""
    rng = np.random.default_rng(trait_seed("odor/v2", slot, value))
    gloms = np.unique(orn_glomerulus)
    pick = rng.choice(len(gloms), size=k, replace=False)
    u = rng.uniform(0.5, 1.0, size=k)
    neurons, weights = [], []
    for gi, ui in zip(pick, u):
        members = orn[orn_glomerulus == gloms[gi]]
        neurons.append(members)
        weights.append(np.full(len(members), ui / np.sqrt(len(members)), dtype=np.float32))
    return Code(np.concatenate(neurons).astype(np.int64), np.concatenate(weights))


def pool_code(slot: str, value: str, pool: np.ndarray, k: int = 14, tag: str = "allsens/v1") -> Code:
    rng = np.random.default_rng(trait_seed(tag, slot, value))
    neurons = rng.choice(pool, size=k, replace=False).astype(np.int64)
    weights = rng.uniform(0.5, 1.0, size=k).astype(np.float32)
    return Code(neurons, weights)


@dataclass(frozen=True)
class Retina:
    neurons: np.ndarray
    px: np.ndarray
    py: np.ndarray
    channel: np.ndarray


def build_retina(cols: Columns, g, size: int = 64) -> Retina:
    x, y = lattice_xy(cols)
    channel = np.array([_CHANNEL_INDEX[CHANNEL_BY_TYPE[t]] for t in g.cell_type[cols.neuron]], np.int64)
    px = np.clip(np.rint(x * (size - 1)), 0, size - 1).astype(np.int64)
    py = np.clip(np.rint(y * (size - 1)), 0, size - 1).astype(np.int64)
    return Retina(neurons=cols.neuron.astype(np.int64), px=px, py=py, channel=channel)


_KERNEL = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


def retina_values(images: torch.Tensor, retina: Retina) -> torch.Tensor:
    """images [B,3,S,S] in [0,1] -> photoreceptor values [P,B] (Gaussian 3x3 at each column)."""
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    planes = torch.stack([0.299 * r + 0.587 * g + 0.114 * b, g, b], dim=1)
    kernel = _KERNEL.to(images.device, images.dtype).expand(3, 1, 3, 3)
    blurred = F.conv2d(F.pad(planes, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
    ch = torch.as_tensor(retina.channel, device=images.device)
    py = torch.as_tensor(retina.py, device=images.device)
    px = torch.as_tensor(retina.px, device=images.device)
    return blurred[:, ch, py, px].T


class InputBuilder:
    """Turns looks (and their renders) into an [N,B] drive for one input code."""

    def __init__(self, name: str, n: int, pops: Populations, retina: Retina, g_in: float,
                 device: str = "cuda"):
        if name not in CODE_NAMES:
            raise ValueError(f"unknown input code {name!r}")
        self.name, self.n, self.pops, self.retina = name, n, pops, retina
        self.g_in, self.device = float(g_in), device
        self.uses_eyes = "eyes" in name
        self.g_eye = self.g_in if self.uses_eyes else 1.0
        self._codes: dict[tuple[str, str], Code] = {}

    @property
    def components(self) -> tuple[str, ...]:
        return {"eyes+nose": ("eyes", "nose")}.get(self.name, (self.name,))

    def only(self, component: str) -> InputBuilder:
        if component not in self.components:
            raise ValueError(f"{self.name} has no component {component!r}")
        other = InputBuilder(component, self.n, self.pops, self.retina, self.g_in, self.device)
        other.g_eye = self.g_eye
        return other

    def _code(self, slot: str, value: str) -> Code:
        key = (slot, value)
        if key not in self._codes:
            if self.name in ("nose", "eyes+nose"):
                self._codes[key] = odor_code(slot, value, self.pops.orn, self.pops.orn_glomerulus)
            elif self.name == "all-sensory-brain":
                self._codes[key] = pool_code(slot, value, self.pops.sensory_brain)
            else:
                pool = np.concatenate([self.pops.sensory_brain, self.pops.sensory_vnc])
                self._codes[key] = pool_code(slot, value, pool, tag="allsens+vnc/v1")
        return self._codes[key]

    def rest_key(self) -> str:
        return f"grey@{self.g_eye:g}"

    def rest_drive(self) -> torch.Tensor:
        drive = torch.zeros(self.n, 1, device=self.device)
        drive[torch.as_tensor(self.retina.neurons, device=self.device)] = 0.5 * self.g_eye
        return drive

    def drive(self, looks: list[tuple[str, ...]], images: torch.Tensor | None = None,
              conc: np.ndarray | None = None) -> torch.Tensor:
        b = len(looks)
        drive = torch.zeros(self.n, b, device=self.device)
        rn = torch.as_tensor(self.retina.neurons, device=self.device)
        if self.uses_eyes:
            if images is None:
                raise ValueError(f"{self.name} needs rendered images")
            drive[rn] = self.g_eye * retina_values(images.to(self.device), self.retina)
        else:
            drive[rn] = 0.5 * self.g_eye
        if self.name == "eyes":
            return drive
        if conc is None:
            conc = np.full((b, len(SLOTS)), C_REF, np.float32)
        rows, cols, vals = [], [], []
        for j, look in enumerate(looks):
            for s, (slot, value) in enumerate(zip(SLOTS, look)):
                code = self._code(slot, value)
                rows.append(code.neurons)
                cols.append(np.full(len(code.neurons), j, np.int64))
                vals.append(code.weights * np.float32(self.g_in * conc[j, s]))
        index = (torch.as_tensor(np.concatenate(rows), device=self.device),
                 torch.as_tensor(np.concatenate(cols), device=self.device))
        drive.index_put_(index, torch.as_tensor(np.concatenate(vals), device=self.device), accumulate=True)
        return drive
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_senses.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/brain/senses.py tests/test_senses.py
git commit -m "feat(brain): glomerulus odour codes, all-sensory codes, hex retina, input builder"
```

---

### Task 7: Catalog, z-order, layer bank, compositor

**Files:**
- Create: `lfg_fly/teacher/catalog.py`, `lfg_fly/teacher/render.py`
- Modify: `lfg_fly/cli.py` (add `catalog`)
- Test: `tests/test_catalog.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: `SLOTS`, `NONE` from `lfg_fly.brain.senses`
- Produces (`catalog`):
  - `BODIES`
  - `http_get(url, timeout=30) -> tuple[int, bytes]`
  - `class Catalog`, a frozen dataclass: `body`, `values: dict[str, list[str]]`, `odds: dict[str, dict[str, float]]`, `api: str`. It has `to_json()` and `Catalog.from_json(text)`.
  - `layer_cache_path(cache, body, slot, value) -> Path`
  - `build_catalog(api, cache, body="male", get=http_get) -> Catalog`
- Produces (`render`):
  - `LFG_TRAIT_CONFIG_COMMIT`
  - `class ZOrder` with `.key(slot, value) -> tuple[float, int]`
  - `zorder_from_config(cfg: dict) -> ZOrder`
  - `load_zorder(cache, commit=LFG_TRAIT_CONFIG_COMMIT, get=http_get) -> ZOrder`
  - `class LayerBank(catalog, cache, size=64, device="cpu")` with `.index` and `.stack: Tensor[K,4,S,S]`
  - `composite(looks, bank, zorder) -> Tensor[B,3,S,S]`

- [ ] **Step 1: Write the failing tests**

`tests/test_catalog.py`:
```python
import json

from lfg_fly.teacher import catalog as K


def _fake_get(layers_ok):
    def get(url, timeout=30):
        if "/api/rarity?body=" in url:
            body = url.split("body=")[1]
            slots = {"Head": [{"value": f"{body}-hat", "odds_pct": 50.0, "enabled": True},
                              {"value": "None", "odds_pct": 50.0, "enabled": True}],
                     "Body": [{"value": f"{body}-body", "odds_pct": 100.0, "enabled": True}]}
            return 200, json.dumps({"body": body, "slots": slots}).encode()
        if "/api/layer?" in url:
            ok = any(v in url for v in layers_ok)
            return (200, b"PNGDATA") if ok else (404, b"")
        return 404, b""

    return get


def test_catalog_unions_bodies_filters_by_layer_and_adds_none(tmp_path):
    get = _fake_get(layers_ok=["male-hat", "female-hat", "male-body"])
    cat = K.build_catalog("http://lfg", tmp_path, body="male", get=get)
    assert cat.values["Head"] == ["female-hat", "male-hat", "None"]
    assert cat.values["Body"] == ["male-body"]  # only the hero's own body class
    assert "None" in cat.values["Accessory"] and cat.values["Accessory"] == ["None"]
    assert cat.odds["Head"] == {"male-hat": 50.0, "None": 50.0}
    assert K.layer_cache_path(tmp_path, "male", "Head", "male-hat").read_bytes() == b"PNGDATA"
    assert K.layer_cache_path(tmp_path, "male", "Head", "ape-hat").read_bytes() == b""  # cached 404


def test_catalog_json_roundtrip(tmp_path):
    cat = K.build_catalog("http://lfg", tmp_path, get=_fake_get(["male"]))
    assert K.Catalog.from_json(cat.to_json()) == cat


def test_cached_layers_are_not_refetched(tmp_path):
    calls = []
    base = _fake_get(["male-hat", "male-body"])

    def counting(url, timeout=30):
        calls.append(url)
        return base(url, timeout)

    K.build_catalog("http://lfg", tmp_path, get=counting)
    n = sum("/api/layer?" in u for u in calls)
    K.build_catalog("http://lfg", tmp_path, get=counting)
    assert sum("/api/layer?" in u for u in calls) == n
```

`tests/test_render.py`:
```python
import io

import numpy as np
import torch
from PIL import Image

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import catalog as K
from lfg_fly.teacher import render as R

CFG = {
    "layers": [{"name": s, "z": 10 * (i + 1)} for i, s in enumerate(SLOTS)],
    "z_overrides": [{"trait_type": "Eyes", "value": "Laser", "z": 95}],
}


def _png(rgba):
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), rgba).save(buf, format="PNG")
    return buf.getvalue()


def _catalog(tmp_path, layers):
    values = {s: ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = []
    for (slot, value), rgba in layers.items():
        path = K.layer_cache_path(tmp_path, "male", slot, value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png(rgba))
        values.setdefault(slot, [])
        values[slot] = [value] + [v for v in values[slot] if v != value]
    return K.Catalog(body="male", values=values, odds={}, api="http://lfg")


def test_zorder_uses_layers_and_overrides():
    z = R.zorder_from_config(CFG)
    assert z.key("Background", "x") < z.key("Head", "x")
    assert z.key("Eyes", "Laser") > z.key("Accessory", "x")  # 95 > 90


def test_composite_alpha_over_in_z_order(tmp_path):
    cat = _catalog(tmp_path, {("Background", "Red"): (255, 0, 0, 255),
                              ("Head", "HalfBlue"): (0, 0, 255, 128)})
    bank = R.LayerBank(cat, tmp_path, size=4)
    look = tuple({"Background": "Red", "Head": "HalfBlue"}.get(s, "None") for s in SLOTS)
    img = R.composite([look], bank, R.zorder_from_config(CFG))
    a = 128 / 255
    expected = torch.tensor([1 - a, 0.0, a])
    assert img.shape == (1, 3, 4, 4)
    assert torch.allclose(img[0, :, 0, 0], expected, atol=0.01)


def test_z_override_changes_stacking(tmp_path):
    cat = _catalog(tmp_path, {("Eyes", "Laser"): (0, 255, 0, 255),
                              ("Accessory", "Red"): (255, 0, 0, 255)})
    bank = R.LayerBank(cat, tmp_path, size=4)
    look = tuple({"Eyes": "Laser", "Accessory": "Red"}.get(s, "None") for s in SLOTS)
    img = R.composite([look], bank, R.zorder_from_config(CFG))
    assert np.allclose(img[0, :, 0, 0].numpy(), [0, 1, 0])  # Laser (95) sits above Accessory (90)


def test_load_zorder_caches_pinned_file(tmp_path):
    import yaml

    calls = []

    def get(url, timeout=30):
        calls.append(url)
        return 200, yaml.safe_dump(CFG).encode()

    R.load_zorder(tmp_path, commit="abc123", get=get)
    R.load_zorder(tmp_path, commit="abc123", get=get)
    assert len(calls) == 1 and "/abc123/trait_config.yaml" in calls[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_catalog.py tests/test_render.py`
Expected: FAIL (import errors).

- [ ] **Step 3: Implement**

`lfg_fly/teacher/catalog.py`:
```python
"""What the hero can wear, read from LFG's public API (spec §3.1, Catalog).

Values are the union of /api/rarity?body=<b> over all five bodies (Body only from
the hero's class), kept iff /api/layer?body=<hero> resolves them. Every non-Body
slot also gets "None", which needs no layer. Layer thumbs are cached on disk; a
cached 404 is an empty file.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

from lfg_fly.brain.senses import NONE, SLOTS

BODIES = ("ape", "female", "male", "milady", "skeleton")


def http_get(url: str, timeout: float = 30) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "lfg-fly/0.0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


@dataclass(frozen=True)
class Catalog:
    body: str
    values: dict[str, list[str]]
    odds: dict[str, dict[str, float]]
    api: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Catalog:
        return cls(**json.loads(text))


def layer_cache_path(cache: Path, body: str, slot: str, value: str) -> Path:
    key = hashlib.sha1(f"{body}|{slot}|{value}".encode()).hexdigest()
    return cache / "layers" / body / slot / f"{key}.img"


def _ensure_layer(api: str, cache: Path, body: str, slot: str, value: str, get) -> bool:
    path = layer_cache_path(cache, body, slot, value)
    if path.exists():
        return path.stat().st_size > 0
    query = urlencode({"body": body, "trait": slot, "value": value, "thumb": "1"})
    status, data = get(f"{api}/api/layer?{query}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = status == 200 and len(data) > 0
    path.write_bytes(data if ok else b"")
    return ok


def build_catalog(api: str, cache: Path, body: str = "male", get=http_get) -> Catalog:
    union: dict[str, set[str]] = {s: set() for s in SLOTS}
    odds: dict[str, dict[str, float]] = {}
    for b in BODIES:
        status, data = get(f"{api}/api/rarity?body={b}")
        if status != 200:
            raise RuntimeError(f"/api/rarity?body={b} -> HTTP {status}")
        for slot, rows in json.loads(data)["slots"].items():
            if slot not in union or (slot == "Body" and b != body):
                continue
            for row in rows:
                union[slot].add(str(row["value"]))
                if b == body and row.get("enabled", True):
                    odds.setdefault(slot, {})[str(row["value"])] = float(row["odds_pct"])
    values: dict[str, list[str]] = {}
    for slot in SLOTS:
        kept = [v for v in sorted(union[slot] - {NONE}) if _ensure_layer(api, cache, body, slot, v, get)]
        if slot != "Body":
            kept.append(NONE)
        values[slot] = kept
    return Catalog(body=body, values=values, odds=odds, api=api)
```

`lfg_fly/teacher/render.py`:
```python
"""Composite looks like LFG does: layers stacked by z (trait_config.yaml at a
pinned LFG commit, incl. z_overrides), None slots skipped, first frame of any
animated layer. Runs on the device at retina resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from lfg_fly.brain.senses import NONE, SLOTS
from lfg_fly.teacher.catalog import Catalog, http_get, layer_cache_path

LFG_TRAIT_CONFIG_COMMIT = "0415a63afdcea11156b57169d34e38bb8d377adc"
RAW_URL = "https://raw.githubusercontent.com/Team-Hamsa/LFG/{commit}/trait_config.yaml"


@dataclass(frozen=True)
class ZOrder:
    layer_z: dict[str, float]
    overrides: dict[tuple[str, str], float]
    rank: dict[str, int]

    def key(self, slot: str, value: str) -> tuple[float, int]:
        return (self.overrides.get((slot, value), self.layer_z[slot]), self.rank[slot])


def zorder_from_config(cfg: dict) -> ZOrder:
    layers = cfg["layers"]
    return ZOrder(
        layer_z={row["name"]: float(row["z"]) for row in layers},
        overrides={(o["trait_type"], o["value"]): float(o["z"]) for o in cfg.get("z_overrides") or []},
        rank={row["name"]: i for i, row in enumerate(layers)},
    )


def load_zorder(cache: Path, commit: str = LFG_TRAIT_CONFIG_COMMIT, get=http_get) -> ZOrder:
    path = cache / f"trait_config-{commit}.yaml"
    if not path.exists():
        status, data = get(RAW_URL.format(commit=commit))
        if status != 200:
            raise RuntimeError(f"trait_config.yaml@{commit}: HTTP {status}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return zorder_from_config(yaml.safe_load(path.read_text()))


class LayerBank:
    """Every catalog layer as a premultiplied RGBA tensor [K,4,S,S] on the device."""

    def __init__(self, catalog: Catalog, cache: Path, size: int = 64, device: str = "cpu"):
        entries = [(s, v) for s in SLOTS for v in catalog.values.get(s, []) if v != NONE]
        tiles = []
        for slot, value in entries:
            with Image.open(layer_cache_path(cache, catalog.body, slot, value)) as im:
                im.seek(0)
                rgba = im.convert("RGBA").resize((size, size), Image.LANCZOS)
            a = np.asarray(rgba, dtype=np.float32) / 255.0
            tiles.append(np.concatenate([a[..., :3] * a[..., 3:4], a[..., 3:4]], axis=-1))
        self.size = size
        self.index = {e: i for i, e in enumerate(entries)}
        stack = np.stack(tiles) if tiles else np.zeros((0, size, size, 4), np.float32)
        self.stack = torch.as_tensor(stack).permute(0, 3, 1, 2).contiguous().to(device)


def composite(looks: list[tuple[str, ...]], bank: LayerBank, zorder: ZOrder) -> torch.Tensor:
    s = bank.size
    out = torch.zeros(len(looks), 3, s, s, device=bank.stack.device)
    for b, look in enumerate(looks):
        layers = sorted(((slot, v) for slot, v in zip(SLOTS, look) if v != NONE),
                        key=lambda sv: zorder.key(*sv))
        acc = torch.zeros(3, s, s, device=bank.stack.device)
        for sv in layers:
            tile = bank.stack[bank.index[sv]]
            acc = tile[:3] + acc * (1.0 - tile[3:4])
        out[b] = acc
    return out.clamp_(0.0, 1.0)
```

Add to `lfg_fly/cli.py`:
```python
    p = sub.add_parser("catalog", help="read the hero's wardrobe catalog from LFG's public API")
    p.add_argument("--api", default="http://localhost:8176")
    p.add_argument("--body", default="male")
    p.set_defaults(func=_cmd_catalog)
```
```python
def _cmd_catalog(args: argparse.Namespace) -> int:
    from lfg_fly import paths
    from lfg_fly.teacher.catalog import build_catalog
    from lfg_fly.teacher.render import load_zorder

    cache = paths.network_dir("mainnet") / "catalog"
    cat = build_catalog(args.api.rstrip("/"), cache, body=args.body)
    (cache / f"catalog-{args.body}.json").write_text(cat.to_json())
    load_zorder(cache)
    print({slot: len(v) for slot, v in cat.values.items()})
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_catalog.py tests/test_render.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/teacher/catalog.py lfg_fly/teacher/render.py lfg_fly/cli.py tests/test_catalog.py tests/test_render.py
git commit -m "feat(teacher): LFG catalog from public API, pinned z-order, GPU layer compositor"
```

---

### Task 8: Looks, pairs, planted taste

**Files:**
- Create: `lfg_fly/teacher/sample.py`
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `Catalog`, `SLOTS`, `NON_BODY_SLOTS`
- Produces:
  - `Look = tuple[str, ...]`
  - `class Pair`, a frozen dataclass: `family: int`, `a: Look`, `b: Look`, `near: bool`
  - `realistic_look(cat, rng)`, `uniform_look(cat, rng)`, `hybrid_look(cat, rng)`, `base_look(cat, rng)`
  - `near_variant(look, cat, rng, n_slots=None) -> Look`
  - `make_pairs(cat, n_families, pairs_per_family=3, near_frac=0.6, seed=0) -> list[Pair]`
  - `class PlantedTaste`, a frozen dataclass: `theta: dict[tuple[str,str], float]`, `k: float = 1.2`, with `.utility(look) -> float`
  - `plant_taste(cat, seed) -> PlantedTaste`
  - `class ProbeSet`, a frozen dataclass: `looks`, `a_idx`, `b_idx`, `family`, `near`, `y`, `p`, `test`
  - `make_probe_set(cat, n_pairs=3000, pairs_per_family=3, test_frac=0.2, seed=0) -> ProbeSet`
  - `bayes_ceiling(p) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_sample.py`:
```python
import numpy as np

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog


def _cat():
    values = {s: [f"{s}{i}" for i in range(6)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1"]
    odds = {s: {v: 1.0 for v in vals[:3]} for s, vals in values.items()}
    return Catalog(body="male", values=values, odds=odds, api="x")


def test_realistic_draws_only_odds_values():
    cat, rng = _cat(), np.random.default_rng(0)
    for _ in range(50):
        look = T.realistic_look(cat, rng)
        assert all(v in cat.odds[s] for s, v in zip(SLOTS, look))


def test_near_variant_changes_1_or_2_non_body_slots():
    cat, rng = _cat(), np.random.default_rng(1)
    for _ in range(200):
        base = T.base_look(cat, rng)
        other = T.near_variant(base, cat, rng)
        diff = [s for s, x, y in zip(SLOTS, base, other) if x != y]
        assert 1 <= len(diff) <= 2 and "Body" not in diff


def test_pairs_are_deterministic_and_grouped():
    cat = _cat()
    a = T.make_pairs(cat, n_families=20, seed=3)
    b = T.make_pairs(cat, n_families=20, seed=3)
    assert a == b and len(a) == 60
    assert sorted({p.family for p in a}) == list(range(20))
    near = [p for p in a if p.near]
    assert 0.3 < len(near) / len(a) < 0.9


def test_probe_set_splits_by_family_and_labels_follow_taste():
    ps = T.make_probe_set(_cat(), n_pairs=600, seed=4)
    test_fams, train_fams = set(ps.family[ps.test]), set(ps.family[~ps.test])
    assert not (test_fams & train_fams)
    assert 0.19 <= ps.test.mean() <= 0.21
    assert set(np.unique(ps.y)) <= {0.0, 1.0}
    assert 0.5 < T.bayes_ceiling(ps.p) <= 1.0
    # labels agree with the planted preference more often than not
    assert ((ps.p > 0.5) == (ps.y == 1)).mean() > 0.6
    assert len(ps.looks) == len(set(ps.looks))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_sample.py`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`lfg_fly/teacher/sample.py`:
```python
"""Looks, near/far pairs grouped into families, and the planted taste (spec §3.0, §3.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lfg_fly.brain.senses import NON_BODY_SLOTS, SLOTS
from lfg_fly.teacher.catalog import Catalog

Look = tuple[str, ...]


@dataclass(frozen=True)
class Pair:
    family: int
    a: Look
    b: Look
    near: bool


def _pick(values: list[str], rng) -> str:
    return str(values[int(rng.integers(len(values)))])


def realistic_look(cat: Catalog, rng) -> Look:
    out = []
    for slot in SLOTS:
        odds = {v: w for v, w in cat.odds.get(slot, {}).items() if v in cat.values[slot] and w > 0}
        if odds:
            vals = sorted(odds)
            p = np.array([odds[v] for v in vals], dtype=np.float64)
            out.append(str(vals[int(rng.choice(len(vals), p=p / p.sum()))]))
        else:
            out.append(_pick(cat.values[slot], rng))
    return tuple(out)


def uniform_look(cat: Catalog, rng) -> Look:
    return tuple(_pick(cat.values[s], rng) for s in SLOTS)


def hybrid_look(cat: Catalog, rng) -> Look:
    look = list(realistic_look(cat, rng))
    for slot in rng.choice(NON_BODY_SLOTS, size=3, replace=False):
        look[SLOTS.index(str(slot))] = _pick(cat.values[str(slot)], rng)
    return tuple(look)


def base_look(cat: Catalog, rng) -> Look:
    r = rng.random()
    if r < 0.4:
        return realistic_look(cat, rng)
    return uniform_look(cat, rng) if r < 0.7 else hybrid_look(cat, rng)


def near_variant(look: Look, cat: Catalog, rng, n_slots: int | None = None) -> Look:
    """Change 1-2 non-Body slots (the Builder never changes Body)."""
    candidates = [s for s in NON_BODY_SLOTS if len(cat.values[s]) > 1]
    k = n_slots if n_slots is not None else int(rng.integers(1, 3))
    out = list(look)
    for slot in rng.choice(candidates, size=min(k, len(candidates)), replace=False):
        i = SLOTS.index(str(slot))
        choices = [v for v in cat.values[str(slot)] if v != look[i]]
        out[i] = _pick(choices, rng)
    return tuple(out)


def make_pairs(cat: Catalog, n_families: int, pairs_per_family: int = 3, near_frac: float = 0.6,
               seed: int = 0) -> list[Pair]:
    rng = np.random.default_rng(seed)
    pairs = []
    for fam in range(n_families):
        base = base_look(cat, rng)
        for _ in range(pairs_per_family):
            near = bool(rng.random() < near_frac)
            other = near_variant(base, cat, rng) if near else base_look(cat, rng)
            a, b = (base, other) if rng.random() < 0.5 else (other, base)
            pairs.append(Pair(fam, a, b, near))
    return pairs


@dataclass(frozen=True)
class PlantedTaste:
    theta: dict[tuple[str, str], float]
    k: float = 1.2

    def utility(self, look: Look) -> float:
        return float(sum(self.theta[(s, v)] for s, v in zip(SLOTS, look)))


def plant_taste(cat: Catalog, seed: int) -> PlantedTaste:
    rng = np.random.default_rng(seed)
    keys = [(s, v) for s in SLOTS for v in cat.values[s]]
    return PlantedTaste(theta=dict(zip(keys, rng.normal(0.0, 1.0, len(keys)).tolist())))


@dataclass(frozen=True)
class ProbeSet:
    looks: list[Look]
    a_idx: np.ndarray
    b_idx: np.ndarray
    family: np.ndarray
    near: np.ndarray
    y: np.ndarray
    p: np.ndarray
    test: np.ndarray


def bayes_ceiling(p: np.ndarray) -> float:
    return float(np.mean(np.maximum(p, 1.0 - p)))


def make_probe_set(cat: Catalog, n_pairs: int = 3000, pairs_per_family: int = 3,
                   test_frac: float = 0.2, seed: int = 0) -> ProbeSet:
    pairs = make_pairs(cat, n_pairs // pairs_per_family, pairs_per_family, seed=seed)
    index: dict[Look, int] = {}
    for pr in pairs:
        for look in (pr.a, pr.b):
            index.setdefault(look, len(index))
    taste = plant_taste(cat, seed + 1)
    du = np.array([taste.utility(pr.a) - taste.utility(pr.b) for pr in pairs])
    p = 1.0 / (1.0 + np.exp(-taste.k * du))
    rng = np.random.default_rng(seed + 2)
    y = (rng.random(len(pairs)) < p).astype(np.float32)
    family = np.array([pr.family for pr in pairs])
    fams = rng.permutation(np.unique(family))
    test_fams = set(fams[: math.ceil(test_frac * len(fams))].tolist())
    return ProbeSet(
        looks=list(index),
        a_idx=np.array([index[pr.a] for pr in pairs]),
        b_idx=np.array([index[pr.b] for pr in pairs]),
        family=family,
        near=np.array([pr.near for pr in pairs]),
        y=y,
        p=p,
        test=np.isin(family, list(test_fams)),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_sample.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/teacher/sample.py tests/test_sample.py
git commit -m "feat(teacher): look sampler, near/far pairs in families, planted additive taste"
```

---

### Task 9: Probe evaluation

**Files:**
- Create: `lfg_fly/teacher/probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Consumes: `ProbeSet`, `Catalog`, `SLOTS`, `Look`
- Produces:
  - `LAMBDAS: tuple[float, ...]`
  - `class ProbeResult`, a dataclass: `heldout`, `ci_lo`, `ci_hi`, `train_acc`, `cv_acc`, `lam`, `n_test`, with `.as_dict()`
  - `evaluate(X: np.ndarray[n_looks, F], ps: ProbeSet, device="cpu", seed=0) -> ProbeResult`
  - `fit_bt(D, y, groups, device, lambdas=LAMBDAS, folds=5) -> tuple[Tensor, float, float]`
  - `family_bootstrap(correct: bool[n], family: int[n], n_boot=2000, seed=0) -> tuple[float, float]`
  - `one_hot(looks, cat) -> np.ndarray`
  - `per_slot_decodability(X, looks, cat, train_mask, device="cpu") -> dict[str, dict]`
  - `readout_distance(A: np.ndarray, B: np.ndarray) -> float` (mean L1 per look)
  - `smooth_ok(d_min, d_slot, d_full) -> bool` (`d_min ≤ 0.1·d_slot and d_slot < d_full`)
  - `activity_ok(max_step_frac, readout_active_frac) -> bool` (`≤ 0.10` and `≥ 0.05`)

- [ ] **Step 1: Write the failing tests**

`tests/test_probe.py`:
```python
import numpy as np
import pytest

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import probe as P
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog


def _cat():
    values = {s: [f"{s}{i}" for i in range(8)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1", "B2"]
    return Catalog(body="male", values=values, odds={}, api="x")


def test_one_hot_learns_planted_taste_near_ceiling():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=1500, seed=0)
    res = P.evaluate(P.one_hot(ps.looks, cat), ps, device="cpu")
    assert res.heldout > 0.7
    assert res.ci_lo < res.heldout < res.ci_hi
    assert res.heldout <= T.bayes_ceiling(ps.p[ps.test]) + 0.05


def test_random_features_stay_near_chance():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=1500, seed=1)
    X = np.random.default_rng(0).normal(size=(len(ps.looks), 200)).astype(np.float32)
    res = P.evaluate(X, ps, device="cpu")
    assert abs(res.heldout - 0.5) < 0.08


def test_cv_folds_never_split_a_family():
    groups = np.repeat(np.arange(30), 3)
    folds = P._folds(groups, 5)
    for k in range(5):
        assert not (set(groups[folds == k]) & set(groups[folds != k]))


def test_family_bootstrap_brackets_mean():
    correct = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1] * 10, bool)
    fam = np.repeat(np.arange(50), 2)
    lo, hi = P.family_bootstrap(correct, fam, n_boot=500, seed=0)
    assert lo < correct.mean() < hi


def test_per_slot_decodability_on_one_hot_is_perfect():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=600, seed=2)
    X = P.one_hot(ps.looks, cat)
    train = np.random.default_rng(0).random(len(ps.looks)) < 0.8
    dec = P.per_slot_decodability(X, ps.looks, cat, train)
    assert dec["Head"]["acc"] > 0.95 and dec["Head"]["chance"] < 0.5


def test_checks():
    assert P.smooth_ok(1.0, 20.0, 30.0) and not P.smooth_ok(5.0, 20.0, 30.0)
    assert not P.smooth_ok(1.0, 30.0, 30.0)
    assert P.activity_ok(0.08, 0.06) and not P.activity_ok(0.2, 0.5) and not P.activity_ok(0.05, 0.01)
    assert P.readout_distance(np.ones((3, 4)), np.zeros((3, 4))) == pytest.approx(4.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_probe.py`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`lfg_fly/teacher/probe.py`:
```python
"""Score a feature map against the planted taste (spec §3.0, §3.4).

Bradley-Terry logistic regression on feature differences, L2 chosen by 5-fold
cross-validation grouped by look family, a held-out test split by family, and a
family-resampling bootstrap CI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.sample import ProbeSet

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


@dataclass
class ProbeResult:
    heldout: float
    ci_lo: float
    ci_hi: float
    train_acc: float
    cv_acc: float
    lam: float
    n_test: int

    def as_dict(self) -> dict:
        return asdict(self)


def _folds(groups: np.ndarray, k: int) -> np.ndarray:
    uniq = np.unique(groups)
    order = np.random.default_rng(12345).permutation(len(uniq))
    fold_of = {g: int(i % k) for i, g in zip(order, uniq)}
    return np.array([fold_of[g] for g in groups])


def _fit(D: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    w = torch.zeros(D.shape[1], device=D.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(D @ w, y) + lam * (w**2).sum() / len(y)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def _acc(D: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> float:
    return float(((D @ w > 0).float() == y).float().mean())


def fit_bt(D: torch.Tensor, y: torch.Tensor, groups: np.ndarray, device: str,
           lambdas: tuple[float, ...] = LAMBDAS, folds: int = 5) -> tuple[torch.Tensor, float, float]:
    fold = _folds(groups, folds)
    best = (-1.0, lambdas[-1])
    for lam in lambdas:
        accs = []
        for k in range(folds):
            tr = torch.as_tensor(np.flatnonzero(fold != k), device=device)
            va = torch.as_tensor(np.flatnonzero(fold == k), device=device)
            accs.append(_acc(D[va], y[va], _fit(D[tr], y[tr], lam)))
        score = float(np.mean(accs))
        if score > best[0] or (score == best[0] and lam > best[1]):
            best = (score, lam)
    return _fit(D, y, best[1]), best[1], best[0]


def family_bootstrap(correct: np.ndarray, family: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    fams = np.unique(family)
    per = {f: correct[family == f] for f in fams}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(fams, size=len(fams), replace=True)
        stats.append(np.concatenate([per[f] for f in pick]).mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _standardize(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    mu = X[rows].mean(0)
    sd = X[rows].std(0)
    sd[sd < 1e-3] = 1e-3
    return ((X - mu) / sd).astype(np.float32)


def evaluate(X: np.ndarray, ps: ProbeSet, device: str = "cpu", seed: int = 0) -> ProbeResult:
    train = ~ps.test
    rows = np.unique(np.concatenate([ps.a_idx[train], ps.b_idx[train]]))
    Xs = _standardize(np.asarray(X, dtype=np.float32), rows)
    D = torch.as_tensor(Xs[ps.a_idx] - Xs[ps.b_idx], device=device)
    y = torch.as_tensor(ps.y, device=device)
    tr = torch.as_tensor(np.flatnonzero(train), device=device)
    te = torch.as_tensor(np.flatnonzero(ps.test), device=device)
    w, lam, cv = fit_bt(D[tr], y[tr], ps.family[train], device)
    correct = ((D[te] @ w > 0).float() == y[te]).cpu().numpy().astype(bool)
    lo, hi = family_bootstrap(correct, ps.family[ps.test], seed=seed)
    return ProbeResult(heldout=float(correct.mean()), ci_lo=lo, ci_hi=hi,
                       train_acc=_acc(D[tr], y[tr], w), cv_acc=cv, lam=lam, n_test=int(len(te)))


def one_hot(looks: list[tuple[str, ...]], cat: Catalog) -> np.ndarray:
    offsets, col = {}, 0
    for slot in SLOTS:
        for value in cat.values[slot]:
            offsets[(slot, value)] = col
            col += 1
    X = np.zeros((len(looks), col), np.float32)
    for i, look in enumerate(looks):
        for slot, value in zip(SLOTS, look):
            X[i, offsets[(slot, value)]] = 1.0
    return X


def per_slot_decodability(X: np.ndarray, looks, cat: Catalog, train_mask: np.ndarray,
                          device: str = "cpu") -> dict[str, dict]:
    Xs = torch.as_tensor(_standardize(np.asarray(X, np.float32), np.flatnonzero(train_mask)), device=device)
    tr = torch.as_tensor(np.flatnonzero(train_mask), device=device)
    te = torch.as_tensor(np.flatnonzero(~train_mask), device=device)
    out = {}
    for s, slot in enumerate(SLOTS):
        labels_np = np.array([cat.values[slot].index(look[s]) for look in looks])
        labels = torch.as_tensor(labels_np, device=device)
        k = len(cat.values[slot])
        W = torch.zeros(Xs.shape[1], k, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([W], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xs[tr] @ W, labels[tr]) + 1e-2 * (W**2).sum() / len(tr)
            loss.backward()
            return loss

        opt.step(closure)
        pred = (Xs[te] @ W).argmax(1)
        chance = float(np.bincount(labels_np[train_mask], minlength=k).max() / train_mask.sum())
        out[slot] = {"acc": float((pred == labels[te]).float().mean()), "chance": chance, "values": k}
    return out


def readout_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.abs(np.asarray(A) - np.asarray(B)).sum(1).mean())


def smooth_ok(d_min: float, d_slot: float, d_full: float) -> bool:
    return d_min <= 0.1 * d_slot and d_slot < d_full


def activity_ok(max_step_frac: float, readout_active_frac: float) -> bool:
    return max_step_frac <= 0.10 and readout_active_frac >= 0.05
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_probe.py`
Expected: PASS. If `test_random_features_stay_near_chance` is flaky, the bug is in `evaluate` (it's probably using test pairs for standardization or λ selection); fix the code, never widen the bound.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/teacher/probe.py tests/test_probe.py
git commit -m "feat(teacher): Bradley-Terry probe with family-grouped CV, bootstrap CI, decodability, checks"
```

---

### Task 10: Rewired and sign-shuffled twins

**Files:**
- Create: `lfg_fly/teacher/rewire.py`
- Test: `tests/test_rewire.py`

**Interfaces:**
- Consumes: `Graph`, `build.edge_list`, `build.with_edges`, `build.with_sign`
- Produces:
  - `rewire_edges(pre, post, n, seed=0, max_rounds=200) -> tuple[np.ndarray, int]` (new post, repair swaps)
  - `rewired_graph(g, seed=0, device="cpu") -> tuple[Graph, int]`
  - `sign_shuffled_graph(g, seed=0) -> Graph`

- [ ] **Step 1: Write the failing tests**

`tests/test_rewire.py`:
```python
import numpy as np
import pandas as pd

from lfg_fly.connectome import build as B
from lfg_fly.teacher import rewire as W


def _dense_graph(n=30, seed=0):
    # ~260 of 870 possible edges: dense enough that the raw permutation collides,
    # sparse enough that repair swaps exist
    rng = np.random.default_rng(seed)
    pairs = {(int(a), int(b)) for a, b in rng.integers(0, n, (300, 2)) if a != b}
    pre, post = map(np.array, zip(*sorted(pairs)))
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": ["x"] * n, "type": ["t"] * n, "rootSide": [None] * n})
    nt = pd.DataFrame({"body": np.arange(1, n + 1),
                       "consensus_nt": ["gaba" if i % 3 == 0 else "acetylcholine" for i in range(n)]})
    edges = pd.DataFrame({"body_pre": pre + 1, "body_post": post + 1, "weight": rng.integers(1, 9, len(pre))})
    return B.build_graph(ann, nt, edges, min_syn=1)


def test_rewire_preserves_degrees_with_no_loops_or_duplicates():
    g = _dense_graph()
    pre, post, count = B.edge_list(g)
    new_post, repairs = W.rewire_edges(pre, post, g.n, seed=1)
    assert repairs > 0
    # pre is untouched (out-degrees kept by construction); in-degrees must match exactly
    assert np.array_equal(np.bincount(new_post, minlength=g.n), np.bincount(post, minlength=g.n))
    assert not np.any(pre == new_post)
    keys = pre * g.n + new_post
    assert len(np.unique(keys)) == len(keys)
    assert not np.array_equal(new_post, post)


def test_rewired_graph_keeps_counts_and_signs():
    g = _dense_graph()
    r, repairs = W.rewired_graph(g, seed=2)
    assert r.n == g.n and len(r.col) == len(g.col)
    assert sorted(r.count.tolist()) == sorted(g.count.tolist())
    assert np.array_equal(r.sign, g.sign)
    assert r.graph_hash() != g.graph_hash()


def test_sign_shuffle_permutes_labels():
    g = _dense_graph()
    s = W.sign_shuffled_graph(g, seed=3)
    assert sorted(s.sign.tolist()) == sorted(g.sign.tolist())
    assert not np.array_equal(s.sign, g.sign)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_rewire.py`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`lfg_fly/teacher/rewire.py`:
```python
"""Controls (spec §3.4): a degree-preserving rewire and a sign shuffle.

Permuting the post column keeps every neuron's in- and out-degree. The self-loops
and duplicates it creates are then repaired by swapping post targets with random
other edges, accepting a swap only if it creates no new self-loop or duplicate.
Swaps also preserve degrees, so the result keeps both degree sequences exactly.
"""

from __future__ import annotations

import numpy as np

from lfg_fly.connectome.build import Graph, edge_list, with_edges, with_sign


def _bad(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    keys = pre * n + post
    order = np.argsort(keys, kind="stable")
    dup = np.zeros(len(keys), bool)
    dup[order[1:]] = keys[order][1:] == keys[order][:-1]
    return dup | (pre == post)


def rewire_edges(pre: np.ndarray, post: np.ndarray, n: int, seed: int = 0,
                 max_rounds: int = 200) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)[rng.permutation(len(post))]
    repairs = 0
    for _ in range(max_rounds):
        bad = _bad(pre, post, n)
        if not bad.any():
            return post, repairs
        bi = np.flatnonzero(bad)
        cand = rng.integers(0, len(post), size=len(bi))
        keys = np.sort(pre * n + post)
        new_b = pre[bi] * n + post[cand]
        new_c = pre[cand] * n + post[bi]
        ok = (
            (pre[bi] != post[cand]) & (pre[cand] != post[bi]) & ~bad[cand]
            & ~_in_sorted(keys, new_b) & ~_in_sorted(keys, new_c) & (new_b != new_c)
        )
        # one swap per edge per round; no two accepted swaps may create the same key
        _, first = np.unique(cand, return_index=True)
        keep = np.zeros(len(bi), bool)
        keep[first] = True
        ok &= keep & ~np.isin(cand, bi)
        created = np.concatenate([new_b[ok], new_c[ok]])
        uniq, counts = np.unique(created, return_counts=True)
        clash = np.isin(new_b, uniq[counts > 1]) | np.isin(new_c, uniq[counts > 1])
        ok &= ~clash
        post[bi[ok]], post[cand[ok]] = post[cand[ok]], post[bi[ok]].copy()
        repairs += int(ok.sum())
    raise RuntimeError(f"rewire did not converge in {max_rounds} rounds")


def _in_sorted(sorted_keys: np.ndarray, x: np.ndarray) -> np.ndarray:
    i = np.searchsorted(sorted_keys, x)
    i = np.clip(i, 0, len(sorted_keys) - 1)
    return sorted_keys[i] == x


def rewired_graph(g: Graph, seed: int = 0, device: str = "cpu") -> tuple[Graph, int]:
    pre, post, count = edge_list(g)
    new_post, repairs = rewire_edges(pre, post, g.n, seed=seed)
    return with_edges(g, pre, new_post, count, device), repairs


def sign_shuffled_graph(g: Graph, seed: int = 0) -> Graph:
    return with_sign(g, np.random.default_rng(seed).permutation(g.sign))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_rewire.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/teacher/rewire.py tests/test_rewire.py
git commit -m "feat(teacher): exact degree-preserving rewire with repair swaps; sign shuffle"
```

---

### Task 11: Grid runner, verdict, report, CLI

**Files:**
- Create: `lfg_fly/teacher/grid.py`, `lfg_fly/teacher/report.py`
- Modify: `lfg_fly/cli.py` (add `grid`, `report`)
- Test: `tests/test_grid.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `class Setting`, a frozen dataclass: `brain: BrainParams`, `code: str`, `g_in: float`. It has `.key() -> str` and `.as_dict()`.
  - `default_grid() -> list[Setting]`
  - `class Context`, a dataclass: `graph`, `pops`, `retina`, `catalog`, `bank`, `zorder`, `device`, `probe_set`, `seed`
  - `features_for(ctx, sim, setting, looks, images_cache=None, conc=None) -> tuple[np.ndarray, dict]`
  - `stage_a(ctx, settings, out_path) -> list[dict]` (resumable)
  - `stage_b(ctx, records, out_path, cap=24, probe_all=False) -> list[dict]`
  - `stage_c(ctx, best, out_path) -> dict`
  - `GATE = 0.60`
  - `verdict(stage_a, stage_b, stage_c, one_hot, bayes) -> dict`
  - `read_jsonl(path) -> list[dict]`
  - `report.write_report(results_dir, out_md) -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_grid.py`:
```python
import json

import numpy as np
import pandas as pd
import torch

from lfg_fly.brain.senses import SLOTS, Retina
from lfg_fly.brain.sim import BrainParams
from lfg_fly.connectome import build as B
from lfg_fly.connectome.neurons import populations
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.render import ZOrder


def _toy_context(tmp_path, feedforward: bool):
    """ORNs -> readout. Feed-forward: a clean, learnable map. Otherwise: strong recurrence."""
    rng = np.random.default_rng(0)
    n_orn, n_mid, n_ro = 60, 80, 60
    n = n_orn + n_mid + n_ro
    sc = ["cb_sensory"] * n_orn + ["cb_intrinsic"] * n_mid + ["descending_neuron"] * n_ro
    types = [f"ORN_G{i % 12}" for i in range(n_orn)] + ["x"] * (n_mid + n_ro)
    pre, post = [], []
    for o in range(n_orn):
        for m in rng.choice(n_mid, 6, replace=False):
            pre.append(o)
            post.append(n_orn + m)
    for m in range(n_mid):
        for r in rng.choice(n_ro, 5, replace=False):
            pre.append(n_orn + m)
            post.append(n_orn + n_mid + r)
    if not feedforward:
        for _ in range(900):
            a, b = rng.integers(n_orn, n, 2)
            if a != b:
                pre.append(int(a))
                post.append(int(b))
    edges = pd.DataFrame({"pre": pre, "post": post}).drop_duplicates()
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": sc, "type": types, "rootSide": [None] * n})
    nt = pd.DataFrame({"body": np.arange(1, n + 1), "consensus_nt": ["acetylcholine"] * n})
    g = B.build_graph(ann, nt, pd.DataFrame({"body_pre": edges.pre + 1, "body_post": edges.post + 1,
                                             "weight": 5}), min_syn=1)
    pops = populations(g)
    retina = Retina(neurons=np.array([], np.int64), px=np.array([], np.int64),
                    py=np.array([], np.int64), channel=np.array([], np.int64))
    values = {s: [f"{s}{i}" for i in range(6)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1"]
    cat = Catalog(body="male", values=values, odds={}, api="x")
    ps = T.make_probe_set(cat, n_pairs=900, seed=0)
    z = ZOrder(layer_z={s: float(i) for i, s in enumerate(SLOTS)}, overrides={},
               rank={s: i for i, s in enumerate(SLOTS)})
    return G.Context(graph=g, pops=pops, retina=retina, catalog=cat, bank=None, zorder=z,
                     device="cpu", probe_set=ps, seed=0)


def test_learnable_toy_passes_and_chaotic_toy_fails_smoothness(tmp_path):
    good = _toy_context(tmp_path, feedforward=True)
    # all-sensory codes give the toy ~60 identity channels (its 12 glomeruli would bottleneck "nose")
    s_good = G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.0, steps=8, burn_in=5),
                       "all-sensory-brain", 1.0)
    a = G.stage_a(good, [s_good], tmp_path / "a.jsonl")
    assert a[0]["smooth"]["ok"] is True
    b = G.stage_b(good, a, tmp_path / "b.jsonl", cap=4, probe_all=True)
    assert b[0]["probe"]["heldout"] > 0.6

    bad = _toy_context(tmp_path, feedforward=False)
    s_bad = G.Setting(BrainParams(kind="lif", g_syn=12.0, bias=0.3, steps=40, burn_in=20), "nose", 3.0)
    a_bad = G.stage_a(bad, [s_bad], tmp_path / "a2.jsonl")
    assert a_bad[0]["passed"] is False


def test_stage_a_is_resumable(tmp_path):
    ctx = _toy_context(tmp_path, feedforward=True)
    s = G.Setting(BrainParams(kind="rate", g_syn=1.0, steps=8, burn_in=5), "nose", 1.0)
    out = tmp_path / "a.jsonl"
    G.stage_a(ctx, [s], out)
    first = out.read_text()
    G.stage_a(ctx, [s], out)
    assert out.read_text() == first  # the second run found the key and skipped it


def test_verdict_applies_gate():
    a = [{"key": "k1", "passed": True}]
    b = [{"key": "k1", "passed": True, "probe": {"heldout": 0.61, "ci_lo": 0.57, "ci_hi": 0.65}}]
    v = G.verdict(a, b, {"rewired": {"heldout": 0.55}}, {"heldout": 0.77}, 0.84)
    assert v["pass"] is True and v["best"]["key"] == "k1"
    b[0]["probe"]["heldout"] = 0.58
    assert G.verdict(a, b, {}, {"heldout": 0.77}, 0.84)["pass"] is False


def test_default_grid_covers_every_brain_and_code():
    grid = G.default_grid()
    assert {s.brain.kind for s in grid} == {"lif", "lif-avg", "lif-volley", "rate"}
    assert {s.code for s in grid} == {"nose", "eyes", "eyes+nose", "all-sensory-brain", "all-sensory+vnc"}
    assert len({s.key() for s in grid}) == len(grid)
    json.dumps([s.as_dict() for s in grid])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_grid.py`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`lfg_fly/teacher/grid.py`:
```python
"""The Phase 0 grid (spec §3.0): screen (A), probe (B), twin controls (C), verdict.

- Stage A: every setting simulates 32 base looks plus three perturbations, and
  must pass the activity, sense-change and smoothness checks.
- Stage B: the full planted-taste probe for Stage A passers, capped at the
  `cap` smoothest. When nothing passes, the smoothest settings are probed
  anyway, flagged `passed: False` and ineligible for the gate.
- Stage C: the best eligible setting's rewired and sign-shuffled twins, plus
  per-slot decodability.

Every stage appends JSON lines, keyed by `Setting.key()`, so a killed run
resumes where it stopped.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lfg_fly.brain.senses import CODE_NAMES, NON_BODY_SLOTS, SLOTS, C_REF, InputBuilder
from lfg_fly.brain.sim import BrainParams, Simulator
from lfg_fly.teacher import probe as P
from lfg_fly.teacher.render import composite
from lfg_fly.teacher.sample import base_look, near_variant

GATE = 0.60
BATCH_COLUMNS = 512  # looks x trials per GPU batch (8 GB card, N = 165k)


@dataclass(frozen=True)
class Setting:
    brain: BrainParams
    code: str
    g_in: float

    def as_dict(self) -> dict:
        return {"brain": self.brain.as_dict(), "code": self.code, "g_in": self.g_in}

    def key(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


def default_grid() -> list[Setting]:
    out = []
    lif_like = [
        ("lif", {"steps": 60}),
        ("lif-avg", {"steps": 60, "trials": 8, "noise": 0.05}),
        ("lif-volley", {"steps": 6}),
        ("lif-volley", {"steps": 10}),
    ]
    for code in CODE_NAMES:
        for kind, extra in lif_like:
            for g in (4.0, 6.0, 8.0):
                for bias in (0.0, 0.05, 0.1):
                    for g_in in (0.5, 2.0):
                        out.append(Setting(BrainParams(kind=kind, g_syn=g, bias=bias, **extra), code, g_in))
        for g in (0.5, 1.0, 2.0):
            for bias in (0.0, 0.1):
                for g_in in (0.5, 2.0):
                    out.append(Setting(BrainParams(kind="rate", g_syn=g, bias=bias, steps=60), code, g_in))
    return out


@dataclass
class Context:
    graph: Any
    pops: Any
    retina: Any
    catalog: Any
    bank: Any
    zorder: Any
    device: str
    probe_set: Any
    seed: int = 0


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _images(ctx: Context, looks, cache: dict | None) -> torch.Tensor | None:
    if ctx.bank is None:
        return None
    missing = [lk for lk in looks if cache is None or lk not in cache]
    if missing:
        rendered = composite(missing, ctx.bank, ctx.zorder)
        if cache is not None:
            for lk, img in zip(missing, rendered):
                cache[lk] = img
        else:
            return rendered
    return torch.stack([cache[lk] for lk in looks])


def features_for(ctx: Context, sim: Simulator, setting: Setting, looks, images_cache=None,
                 conc: np.ndarray | None = None, image_scale: float = 1.0,
                 builder: InputBuilder | None = None) -> tuple[np.ndarray, dict]:
    builder = builder or InputBuilder(setting.code, ctx.graph.n, ctx.pops, ctx.retina, setting.g_in, ctx.device)
    batch = max(1, BATCH_COLUMNS // setting.brain.trials)
    feats, act, ro_act, max_frac = [], [], [], 0.0
    for i in range(0, len(looks), batch):
        chunk = looks[i: i + batch]
        imgs = _images(ctx, chunk, images_cache) if builder.uses_eyes else None
        if imgs is not None and image_scale != 1.0:
            imgs = (imgs * image_scale).clamp(0.0, 1.0)
        c = None if conc is None else conc[i: i + batch]
        res = sim.run(builder.drive(chunk, imgs, c), setting.brain, builder.rest_drive(), builder.rest_key())
        feats.append(res.features.float().cpu().numpy())
        act.append(res.active_frac.cpu().numpy())
        ro_act.append(res.readout_active_frac.cpu().numpy())
        max_frac = max(max_frac, res.max_step_frac)
    stats = {"active_frac": float(np.concatenate(act).mean()),
             "readout_active_frac": float(np.concatenate(ro_act).mean()),
             "max_step_frac": float(max_frac)}
    return np.concatenate(feats), stats


def _sense_checks(ctx, sim, setting, rng, images_cache) -> dict:
    base = InputBuilder(setting.code, ctx.graph.n, ctx.pops, ctx.retina, setting.g_in, ctx.device)
    one = [base_look(ctx.catalog, rng) for _ in range(16)]
    two = [base_look(ctx.catalog, rng) for _ in range(16)]
    out = {}
    for comp in base.components:
        b = base.only(comp) if len(base.components) > 1 else base
        f1, _ = features_for(ctx, sim, setting, one, images_cache, builder=b)
        f2, _ = features_for(ctx, sim, setting, two, images_cache, builder=b)
        out[comp] = P.readout_distance(f1, f2) > 0
    return out


def stage_a(ctx: Context, settings: list[Setting], out_path: Path, n: int = 32,
            sim: Simulator | None = None) -> list[dict]:
    done = {r["key"]: r for r in read_jsonl(out_path)}
    sim = sim or Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
    records = []
    for setting in settings:
        key = setting.key()
        if key in done:
            records.append(done[key])
            continue
        t0 = time.time()
        rng = np.random.default_rng(ctx.seed + 7)
        cache: dict = {}
        bases = [base_look(ctx.catalog, rng) for _ in range(n)]
        slot1 = [near_variant(b, ctx.catalog, rng, n_slots=1) for b in bases]
        full = [base_look(ctx.catalog, rng) for _ in range(n)]
        f_base, stats = features_for(ctx, sim, setting, bases, cache)
        f_slot, _ = features_for(ctx, sim, setting, slot1, cache)
        f_full, _ = features_for(ctx, sim, setting, full, cache)
        if setting.code == "eyes":
            f_min, _ = features_for(ctx, sim, setting, bases, cache, image_scale=1.01)
        else:
            conc = np.full((n, len(SLOTS)), C_REF, np.float32)
            conc[:, SLOTS.index(str(rng.choice(NON_BODY_SLOTS)))] += 0.01
            f_min, _ = features_for(ctx, sim, setting, bases, cache, conc=conc)
        d = {"min": P.readout_distance(f_base, f_min), "slot": P.readout_distance(f_base, f_slot),
             "full": P.readout_distance(f_base, f_full)}
        smooth = {**d, "ok": P.smooth_ok(d["min"], d["slot"], d["full"])}
        senses = _sense_checks(ctx, sim, setting, rng, cache)
        activity = P.activity_ok(stats["max_step_frac"], stats["readout_active_frac"])
        rec = {"key": key, "setting": setting.as_dict(), "stats": stats, "smooth": smooth,
               "senses": senses, "activity_ok": activity,
               "passed": bool(activity and smooth["ok"] and all(senses.values())),
               "seconds": round(time.time() - t0, 2)}
        _append(out_path, rec)
        records.append(rec)
    return records


def _smooth_ratio(rec: dict) -> float:
    s = rec["smooth"]
    return s["min"] / s["slot"] if s["slot"] > 0 else math.inf


def _setting_from(rec: dict) -> Setting:
    d = rec["setting"]
    return Setting(BrainParams(**d["brain"]), d["code"], d["g_in"])


def stage_b(ctx: Context, records: list[dict], out_path: Path, cap: int = 24,
            probe_all: bool = False, sim: Simulator | None = None) -> list[dict]:
    passed = [r for r in records if r["passed"]]
    if probe_all:
        pool = list(records)
    elif passed:
        pool = passed
    else:  # nothing eligible: still probe the 6 smoothest, flagged ineligible, for the report
        pool = sorted(records, key=_smooth_ratio)[:6]
    chosen = sorted(pool, key=_smooth_ratio)[:cap]
    dropped = len(pool) - len(chosen)
    done = {r["key"]: r for r in read_jsonl(out_path)}
    sim = sim or Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
    ps, cache, out = ctx.probe_set, {}, []
    for rec in chosen:
        if rec["key"] in done:
            out.append(done[rec["key"]])
            continue
        t0 = time.time()
        X, stats = features_for(ctx, sim, _setting_from(rec), ps.looks, cache)
        result = P.evaluate(X, ps, device=ctx.device, seed=ctx.seed)
        row = {"key": rec["key"], "setting": rec["setting"], "passed": rec["passed"],
               "probe": result.as_dict(), "stats": stats, "dropped_by_cap": dropped,
               "seconds": round(time.time() - t0, 2)}
        _append(out_path, row)
        out.append(row)
    return out


def stage_c(ctx: Context, best: dict, out_path: Path) -> dict:
    from lfg_fly.teacher.rewire import rewired_graph, sign_shuffled_graph

    setting = _setting_from(best)
    ps, out = ctx.probe_set, {"key": best["key"]}
    rewired, repairs = rewired_graph(ctx.graph, seed=ctx.seed + 11, device=ctx.device)
    for name, graph in (("rewired", rewired), ("sign_shuffled", sign_shuffled_graph(ctx.graph, ctx.seed + 13))):
        sim = Simulator(graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
        X, stats = features_for(ctx, sim, setting, ps.looks, {})
        out[name] = {**P.evaluate(X, ps, device=ctx.device, seed=ctx.seed).as_dict(), "stats": stats}
    out["rewire_repairs"] = repairs
    sim = Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
    X, _ = features_for(ctx, sim, setting, ps.looks, {})
    look_train = np.zeros(len(ps.looks), bool)
    look_train[np.unique(np.concatenate([ps.a_idx[~ps.test], ps.b_idx[~ps.test]]))] = True
    out["decodability"] = P.per_slot_decodability(X, ps.looks, ctx.catalog, look_train, ctx.device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def verdict(stage_a_records: list[dict], stage_b_records: list[dict], stage_c_result: dict,
            one_hot: dict, bayes: float) -> dict:
    eligible = [r for r in stage_b_records if r.get("passed")]
    best = max(eligible, key=lambda r: r["probe"]["heldout"], default=None)
    return {
        "gate": GATE,
        "pass": bool(best is not None and best["probe"]["heldout"] >= GATE),
        "best": best,
        "stage_a": {"screened": len(stage_a_records), "passed": sum(r["passed"] for r in stage_a_records)},
        "stage_b_probed": len(stage_b_records),
        "controls": {"one_hot": one_hot, "bayes_ceiling": bayes, **{
            k: v for k, v in stage_c_result.items() if k in ("rewired", "sign_shuffled")}},
    }
```

`lfg_fly/teacher/report.py`:
```python
"""Render data/probe/*.json(l) into docs/PHASE0.md: the committed, honest answer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lfg_fly.teacher.grid import read_jsonl

ATTRIBUTION = (
    "Connectome: Janelia FlyEM MaleCNS v1.0 (CC BY 4.0), Berg et al., Cell 189(18):5504–5526.e15 "
    "(2026), doi:10.1016/j.cell.2026.08.015. Changes: thresholded at ≥3 synapses, signed by predicted "
    "transmitter, photoreceptor columns derived from synaptic partners, simulated. No endorsement by "
    "HHMI/Janelia, Cambridge, MRC-LMB or Google is implied."
)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(results_dir: Path, out_md: Path) -> None:
    v = json.loads((results_dir / "verdict.json").read_text())
    a = read_jsonl(results_dir / "stage_a.jsonl")
    b = read_jsonl(results_dir / "stage_b.jsonl")
    lines = ["# Phase 0: can the fly learn a taste at all?", ""]
    lines += [f"**Verdict: {'PASS' if v['pass'] else 'FAIL'}** (gate: held-out ≥ {v['gate']:.2f} on a "
              "planted additive taste, by a setting that passes every constraint).", ""]
    ctl = v["controls"]
    lines += ["| Reference | Held-out |", "|---|---|",
              f"| Bayes ceiling | {ctl['bayes_ceiling']:.3f} |",
              f"| One-hot, no brain | {ctl['one_hot']['heldout']:.3f} |"]
    if v["best"]:
        pb = v["best"]["probe"]
        lines.append(f"| **Best fly setting** | **{pb['heldout']:.3f}** ({pb['ci_lo']:.3f}–{pb['ci_hi']:.3f}) |")
    for name in ("rewired", "sign_shuffled"):
        if name in ctl:
            lines.append(f"| {name.replace('_', '-')} twin of the best | {ctl[name]['heldout']:.3f} |")
    lines += ["", "## Stage A: screening", "",
              f"{v['stage_a']['passed']} of {v['stage_a']['screened']} settings passed every check.", "",
              "| Brain | Code | Screened | Passed |", "|---|---|---|---|"]
    tally = Counter((r["setting"]["brain"]["kind"], r["setting"]["code"]) for r in a)
    ok = Counter((r["setting"]["brain"]["kind"], r["setting"]["code"]) for r in a if r["passed"])
    for (kind, code), count in sorted(tally.items()):
        lines.append(f"| {kind} | {code} | {count} | {ok[(kind, code)]} |")
    lines += ["", "## Stage B: planted-taste probe", "",
              "| Brain | Code | g_syn | bias | g_in | eligible | held-out | 95% CI |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(b, key=lambda r: -r["probe"]["heldout"]):
        s, pb = r["setting"], r["probe"]
        lines.append(f"| {s['brain']['kind']} | {s['code']} | {s['brain']['g_syn']} | {s['brain']['bias']} | "
                     f"{s['g_in']} | {'yes' if r['passed'] else 'no'} | {pb['heldout']:.3f} | "
                     f"{pb['ci_lo']:.3f}–{pb['ci_hi']:.3f} |")
    if b and b[0].get("dropped_by_cap"):
        lines += ["", f"Stage B cap dropped {b[0]['dropped_by_cap']} passing settings (smoothest kept)."]
    c_path = results_dir / "stage_c.json"
    if c_path.exists():
        c = json.loads(c_path.read_text())
        lines += ["", "## Stage C: per-slot decodability of the best setting", "",
                  "| Slot | Held-out | Chance |", "|---|---|---|"]
        for slot, d in c["decodability"].items():
            lines.append(f"| {slot} | {_pct(d['acc'])} | {_pct(d['chance'])} |")
        lines += ["", f"Rewire repair swaps: {c['rewire_repairs']:,}."]
    lines += ["", "---", "", ATTRIBUTION, ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
```

Add to `lfg_fly/cli.py`:
```python
    p = sub.add_parser("grid", help="run the Phase 0 probe grid (Stages A, B, C) and write a verdict")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--cap", type=int, default=24)
    p.add_argument("--stage", choices=["a", "b", "c", "all"], default="all")
    p.set_defaults(func=_cmd_grid)
    p = sub.add_parser("report", help="write docs/PHASE0.md from data/probe")
    p.set_defaults(func=_cmd_report)
```
```python
def _load_context(args: argparse.Namespace):
    from lfg_fly import paths
    from lfg_fly.brain.senses import build_retina
    from lfg_fly.connectome import build as B
    from lfg_fly.connectome.columns import load_columns
    from lfg_fly.connectome.neurons import populations
    from lfg_fly.teacher.catalog import Catalog
    from lfg_fly.teacher.grid import Context
    from lfg_fly.teacher.render import LayerBank, load_zorder
    from lfg_fly.teacher.sample import make_probe_set

    g = B.load_graph(paths.graph_dir() / f"graph-syn{args.min_syn}.npz")
    pops = populations(g)
    retina = build_retina(load_columns(paths.graph_dir() / f"columns-syn{args.min_syn}.npz"), g)
    cache = paths.network_dir("mainnet") / "catalog"
    cat = Catalog.from_json((cache / "catalog-male.json").read_text())
    bank = LayerBank(cat, cache, size=64, device=args.device)
    return Context(graph=g, pops=pops, retina=retina, catalog=cat, bank=bank, zorder=load_zorder(cache),
                   device=args.device, probe_set=make_probe_set(cat, n_pairs=args.pairs, seed=0), seed=0)


def _cmd_grid(args: argparse.Namespace) -> int:
    import json

    from lfg_fly import env, paths
    from lfg_fly.teacher import grid as G
    from lfg_fly.teacher import probe as P
    from lfg_fly.teacher.sample import bayes_ceiling

    env.configure_libraries()
    ctx = _load_context(args)
    out = paths.repo_root() / "data" / "probe"
    a = G.read_jsonl(out / "stage_a.jsonl")
    if args.stage in ("a", "all"):
        a = G.stage_a(ctx, G.default_grid(), out / "stage_a.jsonl")
        print(f"stage A: {sum(r['passed'] for r in a)}/{len(a)} passed")
    b = G.read_jsonl(out / "stage_b.jsonl")
    if args.stage in ("b", "all"):
        b = G.stage_b(ctx, a, out / "stage_b.jsonl", cap=args.cap)
        print(f"stage B: probed {len(b)}")
    c = json.loads((out / "stage_c.json").read_text()) if (out / "stage_c.json").exists() else {}
    eligible = [r for r in b if r["passed"]]
    if args.stage in ("c", "all") and eligible:
        c = G.stage_c(ctx, max(eligible, key=lambda r: r["probe"]["heldout"]), out / "stage_c.json")
    ps = ctx.probe_set
    one_hot = P.evaluate(P.one_hot(ps.looks, ctx.catalog), ps, device=args.device).as_dict()
    v = G.verdict(a, b, c, one_hot, bayes_ceiling(ps.p[ps.test]))
    v["graph_hash"] = ctx.graph.graph_hash()
    v["catalog_values"] = {slot: len(vals) for slot, vals in ctx.catalog.values.items()}
    (out / "verdict.json").write_text(json.dumps(v, indent=1, sort_keys=True) + "\n")
    print(f"VERDICT: {'PASS' if v['pass'] else 'FAIL'}"
          + (f" (best held-out {v['best']['probe']['heldout']:.3f})" if v["best"] else " (no eligible setting)"))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from lfg_fly import paths
    from lfg_fly.teacher.report import write_report

    write_report(paths.repo_root() / "data" / "probe", paths.repo_root() / "docs" / "PHASE0.md")
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 .venv/bin/pytest -q tests/test_grid.py && nice -n 10 .venv/bin/pytest -q && .venv/bin/ruff check .`
Expected: all PASS; ruff reports no errors.

If `test_learnable_toy_passes_and_chaotic_toy_fails_smoothness` fails on the learnable toy, the pipeline is broken (a near-linear feed-forward rate brain must learn an additive taste). Debug `features_for` and `evaluate`; don't loosen the assert.

- [ ] **Step 5: Commit**

```bash
git add lfg_fly/teacher/grid.py lfg_fly/teacher/report.py lfg_fly/cli.py tests/test_grid.py
git commit -m "feat(teacher): Phase 0 grid (screen, probe, twin controls), verdict and report"
```

---

### Task 12: Run Phase 0 on the real connectome

This is an operations task: no new code, unless a real-data assert fails (then fix, test and commit before continuing). Every command runs from `~/lfg-fly` under `nice -n 10 ionice -c3`.

- [ ] **Step 1: Fetch** (hardlinks the review's copies; no download)

Run: `nice -n 10 ionice -c3 .venv/bin/fly fetch --from ~/fly-data/review-2026-09-22/data`
Expected: three lines at 14,483,314 / 43,282,834 / 508,025,642 bytes.

- [ ] **Step 2: Build the graph and columns, then assert the real-data facts**

Run:
```bash
nice -n 10 ionice -c3 .venv/bin/fly build --device cuda
nice -n 10 ionice -c3 .venv/bin/fly columns
nice -n 10 ionice -c3 .venv/bin/python - <<'EOF'
import lfg_fly, numpy as np
from lfg_fly import paths
from lfg_fly.connectome import build as B, neurons as N, columns as C
g = B.load_graph(paths.graph_dir() / "graph-syn3.npz"); p = N.populations(g)
c = C.load_columns(paths.graph_dir() / "columns-syn3.npz")
facts = {"neurons": (g.n, 165_122), "edges>=3": (len(g.col), 10_511_038), "readout": (len(p.readout), 2_129),
         "photoreceptors": (len(p.photoreceptor), 4_107), "orn": (len(p.orn), 2_635),
         "glomeruli": (len(np.unique(p.orn_glomerulus)), 53), "sensory_brain": (len(p.sensory_brain), 8_982),
         "sensory_vnc": (len(p.sensory_vnc), 6_365), "sensory_any": (int(p.sensory_any.sum()), 15_347),
         "nt_missing": (g.nt_missing, 502)}
for k, (got, want) in facts.items():
    print(f"{k}: {got} (expected {want})", "OK" if got == want else "MISMATCH")
print("columns assigned", c.total - c.dropped, "/", c.total, "(review measured 3,936)")
assert all(got == want for got, want in facts.values())
EOF
```
Expected: every line reads OK, and about 3,936 of 4,107 columns are assigned. On any MISMATCH, stop, find the cause (most likely a selection bug), fix it with a test, commit, and re-run.

- [ ] **Step 3: Catalog** (reads prod's public API on this box, plus the pinned `trait_config.yaml`)

Run: `nice -n 10 ionice -c3 .venv/bin/fly catalog --api http://localhost:8176`
Expected: a per-slot count dict. Head should be above 121 (the male mint list, plus female and universal art), and every non-Body slot includes `None`.

- [ ] **Step 4: Run the grid on the GPU in the background, then check its first output**

Run: `nohup nice -n 10 ionice -c3 .venv/bin/fly grid --device cuda > ~/fly-data/grid.log 2>&1 &`
While it runs, check `nvidia-smi` (GPU busy, CPU near one core) and `tail ~/fly-data/grid.log`.
If it dies, re-running the same command resumes from the JSONL files.
Expected: it ends with `VERDICT: PASS (…)` or `VERDICT: FAIL (…)`.

- [ ] **Step 5: Write the report and commit the results**

Run: `.venv/bin/fly report`
Then:
```bash
git add data/probe docs/PHASE0.md
git commit -m "data: Phase 0 probe results and verdict"
```

- [ ] **Step 6: Push and open the PR**

```bash
git push -u origin phase0-probe
gh pr create --repo joshuahamsa/lfg-fly --base main --head phase0-probe \
  --title "Phase 0: feasibility probe — <PASS|FAIL>" --body-file - <<'EOF'
Implements docs/plans/2026-09-23-phase0-feasibility-probe.md (spec §2 + §3.0).

Verdict and full tables: docs/PHASE0.md. Raw results: data/probe/.
EOF
```
(Replace `<PASS|FAIL>` with the verdict. No AI attribution in the body.)
