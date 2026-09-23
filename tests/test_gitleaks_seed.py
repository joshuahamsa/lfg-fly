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
    cmd = [
        gitleaks,
        "dir",
        str(tmp_path),
        "--config",
        str(config),
        "--no-banner",
        "--exit-code",
        "1",
    ]
    return subprocess.run(cmd, capture_output=True).returncode


def test_rule_catches_generated_seeds(tmp_path):
    xrpl = pytest.importorskip("xrpl")
    from xrpl.core.keypairs import generate_seed

    algorithms = (xrpl.CryptoAlgorithm.SECP256K1, xrpl.CryptoAlgorithm.ED25519)
    seeds = [generate_seed(algorithm=a) for a in algorithms]
    assert _scan(tmp_path, "\n".join(seeds)) == 1


def test_rule_ignores_addresses_and_hashes(tmp_path):
    text = (
        "rrrrrrrrrrrrrrrrrNAMEtxvNvQ\n"
        "4109C6F2045FC7EFF4CDE8F9905D19C28820D86304080FF886B299F0206E42B5\n"
    )
    assert _scan(tmp_path, text) == 0
