"""The fly's configuration (contract §body/config.py; spec §4.4 FLY_ENABLED, caps, chain identity).

`load_config` reads the environment, or a mapping standing in for it, and never a file:
the CLI loads `~/lfg-fly/.env` into the environment before calling it. The mainnet
RegularKey seed rides on the config for the signer and is excluded from `repr`, so a
logged config never carries it. Everything is validated up front and refused loudly, with
the offending variable named, because the loop runs unattended.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from lfg_fly.paths import NETWORKS

DEFAULT_API_BASE = {"testnet": "http://localhost:8177", "mainnet": "http://localhost:8176"}

# Full-history public endpoints for the §4.4 identity check and every ledger read. The
# public testnet rippled nodes are pruned; only clio still serves ledger 32570.
TESTNET_RPC_URLS = (
    "https://s.altnet.rippletest.net:51234/",
    "https://testnet.xrpl-labs.com/",
    "https://clio.altnet.rippletest.net:51234/",
)
MAINNET_RPC_URLS = ("https://s2.ripple.com:51234/", "https://xrplcluster.com/")
DEFAULT_RPC_URLS = {"testnet": TESTNET_RPC_URLS, "mainnet": MAINNET_RPC_URLS}

# Ledger 32570 (the earliest ledger on record). Mainnet's is a constant of the design
# (spec §4.4); the environment cannot move it. Testnet's is a pin: a testnet reset changes
# it and the identity check then fails closed until FLY_TESTNET_LEDGER_HASH is re-pinned.
MAINNET_LEDGER_HASH = "4109C6F2045FC7EFF4CDE8F9905D19C28820D86304080FF886B299F0206E42B5"
TESTNET_LEDGER_HASH_PIN = "18D82E2D616C76A960DB76D514ED39BCC377381A037DFC0F5AF4F0E22E14FC53"
_HASH_RE = re.compile(r"[0-9A-F]{64}")


@dataclass(frozen=True)
class FlyConfig:
    """Everything the body needs to know, resolved once per process.

    `max_steps` and `tabu_days` are §4.4 caps, not knobs: they have no environment
    variable. `rpc_urls`, `expected_ledger_hash`, `signing_account` and `regular_seed`
    default to "unknown" here so a config can be built by hand; `load_config` fills the
    per-network defaults. An unknown signing account means a mint is refused; an unknown
    hash means the identity check has nothing to match and fails closed.
    """

    network: str  # "testnet" | "mainnet"
    api_base: str  # LFG's public API, no trailing slash
    version: str = "fly-v1"  # checkpoint version
    beta: float = 0.3  # greed, in units of taste sd (spec §2 Decision score)
    lam: float = 0.05  # cost weight, per BRIX
    temperature: float = 0.15  # softmax temperature (spec §4.1 step 5)
    max_steps: int = 3  # at most 3 slots a day (spec §4.4 caps)
    tabu_days: int = 30  # never re-wear a complete look from the last 30 days
    enabled: bool = False  # FLY_ENABLED=1; the kill switch is FLY_ENABLED=0 (spec §4.4)
    setup_max_xrp: float = 0.0  # setup-wide spend cap; setup refuses to exceed it
    donor_sources: tuple[str, ...] = ()  # wallets allowed to transfer NFTs to the fly (§4.3)
    rpc_urls: tuple[str, ...] = ()
    expected_ledger_hash: str | None = None
    signing_account: str | None = None  # LFG's mint destination for this network
    regular_seed: str | None = field(default=None, repr=False)  # mainnet only; never logged
    x_monthly_budget: int = 40

    def __post_init__(self) -> None:
        if self.network not in NETWORKS:
            raise ValueError(f"FLY_NETWORK must be one of {NETWORKS}, not {self.network!r}")
        if not self.api_base:
            raise ValueError("FLY_API_BASE must not be empty")
        if not (self.temperature > 0):
            raise ValueError(f"FLY_TEMPERATURE must be > 0, not {self.temperature!r}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, not {self.max_steps!r}")
        if self.tabu_days < 0:
            raise ValueError(f"tabu_days must be >= 0, not {self.tabu_days!r}")
        if not (self.setup_max_xrp >= 0):
            raise ValueError(f"FLY_SETUP_MAX_XRP must be >= 0, not {self.setup_max_xrp!r}")
        if self.x_monthly_budget < 0:
            raise ValueError(f"FLY_X_MONTHLY_BUDGET must be >= 0, not {self.x_monthly_budget!r}")
        if self.expected_ledger_hash is not None and not _HASH_RE.fullmatch(
            self.expected_ledger_hash
        ):
            raise ValueError("expected_ledger_hash must be 64 upper-case hex characters")


def _get(env: Mapping[str, str], key: str) -> str | None:
    """A stripped value, with an empty or blank variable meaning unset."""
    raw = env.get(key)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, not {raw!r}") from None
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{key} must be finite, not {raw!r}")
    return value


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, not {raw!r}") from None


def _csv(env: Mapping[str, str], key: str) -> tuple[str, ...]:
    raw = _get(env, key)
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _ledger_hash(env: Mapping[str, str], network: str) -> str:
    if network == "mainnet":
        return MAINNET_LEDGER_HASH
    raw = _get(env, "FLY_TESTNET_LEDGER_HASH")
    if raw is None:
        return TESTNET_LEDGER_HASH_PIN
    value = raw.upper()
    if not _HASH_RE.fullmatch(value):
        raise ValueError(f"FLY_TESTNET_LEDGER_HASH must be 64 hex characters, not {raw!r}")
    return value


def load_config(env: Mapping[str, str] | None = None) -> FlyConfig:
    """Resolve a FlyConfig from `env` (default: `os.environ`).

    Reads FLY_NETWORK, FLY_API_BASE, FLY_VERSION, FLY_BETA, FLY_LAMBDA, FLY_TEMPERATURE,
    FLY_ENABLED, FLY_SETUP_MAX_XRP, FLY_DONOR_SOURCES, FLY_RPC_URLS,
    FLY_TESTNET_LEDGER_HASH, FLY_LFG_SIGNING_ACCOUNT, FLY_REGULAR_SEED and
    FLY_X_MONTHLY_BUDGET. `FLY_ENABLED` is on only for the exact string "1" (spec §4.4).
    `FLY_REGULAR_SEED` is read on mainnet only; testnet keeps its keys in wallet.json.
    """
    if env is None:
        env = os.environ
    network = _get(env, "FLY_NETWORK") or "testnet"
    if network not in NETWORKS:
        raise ValueError(f"FLY_NETWORK must be one of {NETWORKS}, not {network!r}")
    api_base = (_get(env, "FLY_API_BASE") or DEFAULT_API_BASE[network]).rstrip("/")
    return FlyConfig(
        network=network,
        api_base=api_base,
        version=_get(env, "FLY_VERSION") or "fly-v1",
        beta=_float(env, "FLY_BETA", 0.3),
        lam=_float(env, "FLY_LAMBDA", 0.05),
        temperature=_float(env, "FLY_TEMPERATURE", 0.15),
        enabled=env.get("FLY_ENABLED") == "1",
        setup_max_xrp=_float(env, "FLY_SETUP_MAX_XRP", 0.0),
        donor_sources=_csv(env, "FLY_DONOR_SOURCES"),
        rpc_urls=_csv(env, "FLY_RPC_URLS") or DEFAULT_RPC_URLS[network],
        expected_ledger_hash=_ledger_hash(env, network),
        signing_account=_get(env, "FLY_LFG_SIGNING_ACCOUNT"),
        regular_seed=_get(env, "FLY_REGULAR_SEED") if network == "mainnet" else None,
        x_monthly_budget=_int(env, "FLY_X_MONTHLY_BUDGET", 40),
    )
