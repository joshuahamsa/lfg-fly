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
