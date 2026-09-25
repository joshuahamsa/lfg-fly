"""FlyConfig / load_config (contract §body/config.py, spec §4.4) and the per-network paths."""

from __future__ import annotations

import dataclasses
import os

import pytest

from lfg_fly import paths
from lfg_fly.body import config as C

TESTNET_RPC = (
    "https://s.altnet.rippletest.net:51234/",
    "https://testnet.xrpl-labs.com/",
    "https://clio.altnet.rippletest.net:51234/",
)
MAINNET_RPC = ("https://s2.ripple.com:51234/", "https://xrplcluster.com/")
MAINNET_HASH = "4109C6F2045FC7EFF4CDE8F9905D19C28820D86304080FF886B299F0206E42B5"
TESTNET_PIN = "18D82E2D616C76A960DB76D514ED39BCC377381A037DFC0F5AF4F0E22E14FC53"


# ---------------------------------------------------------------- defaults


def test_defaults_are_testnet_and_disabled():
    cfg = C.load_config({})
    assert cfg.network == "testnet"
    assert cfg.api_base == "http://localhost:8177"
    assert cfg.version == "fly-v1"
    assert (cfg.beta, cfg.lam, cfg.temperature) == (0.3, 0.05, 0.15)
    assert (cfg.max_steps, cfg.tabu_days) == (3, 30)
    assert cfg.enabled is False
    assert cfg.setup_max_xrp == 0.0
    assert cfg.donor_sources == ()
    assert cfg.rpc_urls == TESTNET_RPC == C.TESTNET_RPC_URLS
    assert cfg.expected_ledger_hash == TESTNET_PIN == C.TESTNET_LEDGER_HASH_PIN
    assert cfg.signing_account is None
    assert cfg.regular_seed is None
    assert cfg.x_monthly_budget == 40


def test_mainnet_defaults():
    cfg = C.load_config({"FLY_NETWORK": "mainnet"})
    assert cfg.network == "mainnet"
    assert cfg.api_base == "http://localhost:8176"
    assert cfg.rpc_urls == MAINNET_RPC == C.MAINNET_RPC_URLS
    assert cfg.expected_ledger_hash == MAINNET_HASH == C.MAINNET_LEDGER_HASH


def test_mainnet_hash_is_a_constant_the_environment_cannot_move():
    cfg = C.load_config({"FLY_NETWORK": "mainnet", "FLY_TESTNET_LEDGER_HASH": "AB" * 32})
    assert cfg.expected_ledger_hash == MAINNET_HASH


def test_testnet_hash_pin_comes_from_env_and_is_normalised():
    cfg = C.load_config({"FLY_TESTNET_LEDGER_HASH": ("ab" * 32).lower()})
    assert cfg.expected_ledger_hash == "AB" * 32
    with pytest.raises(ValueError, match="FLY_TESTNET_LEDGER_HASH"):
        C.load_config({"FLY_TESTNET_LEDGER_HASH": "not-a-hash"})
    with pytest.raises(ValueError, match="FLY_TESTNET_LEDGER_HASH"):
        C.load_config({"FLY_TESTNET_LEDGER_HASH": "AB" * 31})


# ---------------------------------------------------------------- env overrides


def test_every_env_override_is_read():
    env = {
        "FLY_NETWORK": "mainnet",
        "FLY_API_BASE": "https://lfg.example/",
        "FLY_VERSION": "fly-v2",
        "FLY_BETA": "0.5",
        "FLY_LAMBDA": "0.1",
        "FLY_TEMPERATURE": "0.2",
        "FLY_ENABLED": "1",
        "FLY_SETUP_MAX_XRP": "25",
        "FLY_DONOR_SOURCES": "rDonorA, rDonorB,,",
        "FLY_RPC_URLS": "https://a.example/, https://b.example/",
        "FLY_LFG_SIGNING_ACCOUNT": "rLfgSigner",
        "FLY_REGULAR_SEED": "sEdFAKE-not-a-seed",
        "FLY_X_MONTHLY_BUDGET": "12",
    }
    cfg = C.load_config(env)
    assert cfg.api_base == "https://lfg.example"  # trailing slash stripped
    assert cfg.version == "fly-v2"
    assert (cfg.beta, cfg.lam, cfg.temperature) == (0.5, 0.1, 0.2)
    assert cfg.enabled is True
    assert cfg.setup_max_xrp == 25.0
    assert cfg.donor_sources == ("rDonorA", "rDonorB")
    assert cfg.rpc_urls == ("https://a.example/", "https://b.example/")
    assert cfg.signing_account == "rLfgSigner"
    assert cfg.regular_seed == "sEdFAKE-not-a-seed"
    assert cfg.x_monthly_budget == 12


@pytest.mark.parametrize("value", ["true", "yes", "on", "0", "", " 1", "enabled"])
def test_enabled_needs_the_exact_string_1(value):
    assert C.load_config({"FLY_ENABLED": value}).enabled is False


def test_enabled_on_1():
    assert C.load_config({"FLY_ENABLED": "1"}).enabled is True


def test_caps_are_not_configurable_from_the_environment():
    # §4.4: at most 3 slots a day and a 30-day tabu are caps, not knobs.
    cfg = C.load_config({"FLY_MAX_STEPS": "9", "FLY_TABU_DAYS": "0"})
    assert (cfg.max_steps, cfg.tabu_days) == (3, 30)


def test_empty_strings_mean_unset():
    cfg = C.load_config({"FLY_API_BASE": "", "FLY_LFG_SIGNING_ACCOUNT": "  ", "FLY_RPC_URLS": ""})
    assert cfg.api_base == "http://localhost:8177"
    assert cfg.signing_account is None
    assert cfg.rpc_urls == TESTNET_RPC


# ---------------------------------------------------------------- the seed


def test_regular_seed_is_mainnet_only():
    seed = "sEdFAKE-not-a-seed"
    assert C.load_config({"FLY_REGULAR_SEED": seed}).regular_seed is None  # testnet: wallet.json
    assert C.load_config({"FLY_NETWORK": "mainnet", "FLY_REGULAR_SEED": seed}).regular_seed == seed
    assert C.load_config({"FLY_NETWORK": "mainnet", "FLY_REGULAR_SEED": ""}).regular_seed is None


def test_seed_never_appears_in_repr_or_str():
    seed = "sEdFAKE-not-a-seed"
    cfg = C.load_config({"FLY_NETWORK": "mainnet", "FLY_REGULAR_SEED": seed})
    assert seed not in repr(cfg)
    assert seed not in str(cfg)
    assert cfg.regular_seed == seed  # still readable by the signer


# ---------------------------------------------------------------- validation


def test_unknown_network_is_refused():
    with pytest.raises(ValueError, match="FLY_NETWORK"):
        C.load_config({"FLY_NETWORK": "devnet"})
    with pytest.raises(ValueError):
        C.FlyConfig(network="devnet", api_base="http://x")


@pytest.mark.parametrize(
    "var,value",
    [
        ("FLY_BETA", "abc"),
        ("FLY_LAMBDA", "x1"),
        ("FLY_TEMPERATURE", "0"),
        ("FLY_TEMPERATURE", "-1"),
        ("FLY_SETUP_MAX_XRP", "-1"),
        ("FLY_SETUP_MAX_XRP", "nan"),
        ("FLY_X_MONTHLY_BUDGET", "1.5"),
        ("FLY_X_MONTHLY_BUDGET", "-3"),
    ],
)
def test_bad_numbers_are_refused_with_the_variable_named(var, value):
    with pytest.raises(ValueError, match=var):
        C.load_config({var: value})


def test_config_is_frozen():
    cfg = C.load_config({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


def test_direct_construction_needs_only_network_and_api_base():
    cfg = C.FlyConfig(network="testnet", api_base="http://localhost:8177")
    assert cfg.enabled is False and cfg.rpc_urls == () and cfg.expected_ledger_hash is None


def test_load_config_reads_os_environ_by_default(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FLY_") and k != "FLY_DATA_DIR":
            monkeypatch.delenv(k)
    monkeypatch.setenv("FLY_NETWORK", "mainnet")
    monkeypatch.setenv("FLY_ENABLED", "1")
    cfg = C.load_config()
    assert cfg.network == "mainnet" and cfg.enabled is True


# ---------------------------------------------------------------- paths (contract §Data directory)


def test_network_scoped_dirs(_data_dir):
    for net in ("testnet", "mainnet"):
        assert paths.records_dir(net) == _data_dir / net / "records"
        assert paths.outbox_dir(net) == _data_dir / net / "outbox"
        assert paths.snapshots_dir(net) == _data_dir / net / "snapshots"
        assert paths.wallet_path(net) == _data_dir / net / "wallet.json"
    with pytest.raises(ValueError):
        paths.records_dir("devnet")


def test_catalog_dir_falls_back_to_mainnet(_data_dir):
    assert paths.catalog_dir("mainnet") == _data_dir / "mainnet" / "catalog"
    # testnet has no catalog of its own today: it reads mainnet's layer cache
    assert paths.catalog_dir("testnet") == _data_dir / "mainnet" / "catalog"
    (_data_dir / "testnet" / "catalog").mkdir(parents=True)
    assert paths.catalog_dir("testnet") == _data_dir / "testnet" / "catalog"
    # a stray file is not a catalog
    (_data_dir / "mainnet").mkdir(parents=True, exist_ok=True)
    (_data_dir / "mainnet" / "catalog").write_text("")
    # mainnet never falls back
    assert paths.catalog_dir("mainnet") == _data_dir / "mainnet" / "catalog"


def test_checkpoint_dir_lives_in_the_repo():
    d = paths.checkpoint_dir("fly-v1")
    assert d == paths.repo_root() / "checkpoints" / "fly-v1"
    with pytest.raises(ValueError):
        paths.checkpoint_dir("../etc")
    with pytest.raises(ValueError):
        paths.checkpoint_dir("")


def test_rarity_head_path_is_per_network_and_per_snapshot(_data_dir):
    h = "ab" * 32
    p = paths.rarity_head_path("testnet", h)
    assert p.parent == _data_dir / "testnet" / "snapshots"
    assert h in p.name and p.suffix == ""  # a base path: the head writes <base>.npz + <base>.json
    assert paths.rarity_head_path("mainnet", h) != p
    with pytest.raises(ValueError):
        paths.rarity_head_path("testnet", "../x")
