"""Where the fly keeps data: FLY_DATA_DIR (default ~/fly-data), never the repo."""

from __future__ import annotations

import os
import re
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


# Per-network data (spec §1: records, sims, snapshots, renders, outbox and the catalog are
# scoped by network; the connectome in raw/ and graph/ is shared).

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _component(kind: str, value: str) -> str:
    """One path component: no separators, no leading dot, so it can never climb out."""
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"invalid {kind} {value!r}")
    return value


def records_dir(network: str) -> Path:
    """`records/<date>.json`, one per UTC day (spec §4.1 step 6)."""
    return network_dir(network) / "records"


def outbox_dir(network: str) -> Path:
    """Alerts for the operator and posts made without X credentials (spec §4.2, §4.5)."""
    return network_dir(network) / "outbox"


def snapshots_dir(network: str) -> Path:
    """Supply snapshots and the rarity heads fitted from them (spec §4, fly-retrain)."""
    return network_dir(network) / "snapshots"


def catalog_dir(network: str) -> Path:
    """The wardrobe catalog and its layer cache.

    Mainnet holds the catalog today. A network without a catalog of its own reads
    mainnet's: the artwork is the same on both stacks, and only mainnet's public API has
    been read for it. Mainnet itself never falls back.
    """
    own = network_dir(network) / "catalog"
    if network == "mainnet" or own.is_dir():
        return own
    return network_dir("mainnet") / "catalog"


def wallet_path(network: str) -> Path:
    """The testnet wallet file (master + RegularKey seeds, chmod 600). Mainnet keeps its
    RegularKey seed in `~/lfg-fly/.env` instead (spec §4.4) and never writes this file."""
    return network_dir(network) / "wallet.json"


def checkpoint_dir(version: str) -> Path:
    """A committed checkpoint: the repo's `checkpoints/<version>/` (spec §2 Checkpoint)."""
    return repo_root() / "checkpoints" / _component("checkpoint version", version)


def rarity_head_path(network: str, snapshot_hash: str) -> Path:
    """Base path of the rarity head fitted from one supply snapshot; the head writes
    `<base>.npz` and `<base>.json` beside each other (contract: save_head/load_head)."""
    return snapshots_dir(network) / f"rarity-head-{_component('snapshot hash', snapshot_hash)}"
