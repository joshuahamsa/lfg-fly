"""`fly setup` against the fake LFG API and a fake JSON-RPC endpoint (spec §5.3, §4.3, §4.4).

No pytest-asyncio mode is relied on: every test drives its coroutine with ``asyncio.run``
while the fake LFG runs in its own thread (``fake_lfg``) and the fake ledger (``FakeRpc``)
runs inside the test's loop. Seeds are generated at run time, never written in a file.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import fake_lfg as FL
import fake_rpc as FR
import pytest
from xrpl.constants import CryptoAlgorithm
from xrpl.core import binarycodec, keypairs
from xrpl.wallet import Wallet

from lfg_fly import paths
from lfg_fly.body import cli_setup
from lfg_fly.body import setup as SU
from lfg_fly.body.chain import Ledger
from lfg_fly.body.client import LfgClient
from lfg_fly.body.config import FlyConfig
from lfg_fly.body.signer import PolicyError, Signer

fake_lfg = FL.fake_lfg  # the shared fixture, bound by name
run = asyncio.run


def seed_for(tag: str) -> str:
    """A deterministic, unfunded family seed for a test role (never a literal in the repo)."""
    return keypairs.generate_seed(hashlib.sha256(tag.encode()).hexdigest()[:32],
                                  CryptoAlgorithm.ED25519)


MASTER_SEED = seed_for("fly-master-wallet")
REGULAR_SEED = seed_for("fly-regular-key")
MASTER = Wallet.from_seed(MASTER_SEED)
REGULAR = Wallet.from_seed(REGULAR_SEED)
FLY = MASTER.address


async def nosleep(_seconds: float) -> None:
    return None


def cfg_for(base: str, state: FL.FakeLfgState, **over) -> FlyConfig:
    kw = dict(network="testnet", api_base=base, signing_account=state.signing_account,
              setup_max_xrp=100.0, rpc_urls=(), expected_ledger_hash=FR.TESTNET_HASH)
    kw.update(over)
    return FlyConfig(**kw)


def bridge_offers(rpc: FR.FakeRpc, state: FL.FakeLfgState, *, fly: str, owner: str) -> None:
    """Serve every pending LFG accept row as an on-ledger sell offer (amount 0, to the fly)."""

    def responder(params: dict) -> dict:
        offers = []
        for row in list(state.sign_requests.values()):
            tx = row.get("txjson") or {}
            if tx.get("TransactionType") == "NFTokenAcceptOffer":
                offers.append({"amount": "0", "flags": 1, "owner": owner, "destination": fly,
                               "nft_offer_index": tx["NFTokenSellOffer"]})
        if not offers:
            raise FR.RpcError("objectNotFound", "The requested object was not found.")
        return {"nft_id": params["nft_id"], "offers": offers}

    rpc.answer("nft_sell_offers", responder)


def apply_regular_key_on_submit(rpc: FR.FakeRpc) -> None:
    """Make the fake ledger reflect a validated SetRegularKey in account_info."""
    inner = rpc.responders["submit"]

    def responder(params: dict) -> dict:
        result = inner(params)
        tx = binarycodec.decode(params["tx_blob"])
        if tx.get("TransactionType") == "SetRegularKey":
            rpc.accounts[tx["Account"]]["RegularKey"] = tx["RegularKey"]
        return result

    rpc.answer("submit", responder)


def write_test_wallet(**over) -> None:
    data = {"network": "testnet", "address": FLY, "master_seed": MASTER_SEED,
            "regular_seed": REGULAR_SEED, "regular_address": REGULAR.address,
            "created_at": "2026-09-25T00:00:00+00:00"}
    data.update(over)
    SU.write_wallet(data)


@asynccontextmanager
async def session(base: str, state: FL.FakeLfgState, cfg: FlyConfig | None = None, *,
                  owner: str | None = None):
    """A signed-in Setup over the fakes: FakeRpc in this loop, fake LFG in its thread."""
    cfg = cfg or cfg_for(base, state)
    state.regular_keys[FLY] = REGULAR.address
    async with FR.FakeRpc() as rpc:
        rpc.add_account(FLY, sequence=10)
        bridge_offers(rpc, state, fly=FLY, owner=owner or state.signing_account)
        ledger = Ledger([rpc.url], "testnet", timeout=5.0, poll_interval=0.0)
        client = LfgClient(base)
        await client.__aenter__()
        client._sleep = nosleep
        try:
            s = SU.Setup(cfg, ledger, client,
                         lambda c: Signer(REGULAR_SEED, FLY, ledger, c),
                         sleep=nosleep, poll_every=0.0)
            await client.sign_in(s.signer, ledger)
            s.rpc = rpc  # type: ignore[attr-defined]  (for assertions)
            yield s
        finally:
            await client.__aexit__(None, None, None)
            await ledger.close()


def submitted_types(rpc: FR.FakeRpc) -> list[str]:
    return [t["TransactionType"] for t in rpc.txs.values()]


# ------------------------------------------------------------------ wallet.json


def test_inside_git_repo_detects_dir_and_file_ancestors(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    assert SU.inside_git_repo(repo / "fly-data" / "testnet" / "wallet.json")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: /somewhere/.git/worktrees/x\n")
    assert SU.inside_git_repo(worktree / "deep" / "wallet.json")
    assert not SU.inside_git_repo(tmp_path / "clean" / "wallet.json")
    # this very checkout is a repo (a worktree: .git is a file)
    assert SU.inside_git_repo(paths.repo_root() / "wallet.json")


def test_write_wallet_refuses_inside_a_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("FLY_DATA_DIR", str(repo / "fly-data"))
    with pytest.raises(SU.InsideRepoError):
        write_test_wallet()
    assert not (repo / "fly-data").exists()
    # the repo's own checkout too
    with pytest.raises(SU.InsideRepoError):
        SU.write_wallet({"network": "testnet"}, path=paths.repo_root() / "wallet.json")
    assert not (paths.repo_root() / "wallet.json").exists()


def test_write_wallet_is_chmod_600_and_reads_back():
    path = SU.write_wallet({"network": "testnet", "address": FLY, "master_seed": MASTER_SEED,
                            "regular_seed": None, "regular_address": None, "created_at": "t"})
    assert path == paths.wallet_path("testnet")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = SU.read_wallet("testnet")
    assert data["address"] == FLY and data["master_seed"] == MASTER_SEED
    assert data["regular_seed"] is None
    assert set(data) == set(SU.WALLET_FIELDS)
    assert not list(path.parent.glob("*.tmp*"))


def test_read_wallet_refusals():
    with pytest.raises(SU.SetupError, match="faucet|keygen"):
        SU.read_wallet("testnet")
    write_test_wallet()
    path = paths.wallet_path("testnet")
    path.chmod(0o644)
    with pytest.raises(SU.SetupError, match="600"):
        SU.read_wallet("testnet")
    path.chmod(0o600)
    path.write_text(json.dumps({"network": "mainnet", "address": FLY}))
    with pytest.raises(SU.SetupError, match="mainnet"):
        SU.read_wallet("testnet")
    with pytest.raises(SU.SetupError, match="mainnet"):
        SU.read_wallet("mainnet")  # setup never writes a mainnet wallet file


# ------------------------------------------------------------------ keygen


def test_keygen_testnet_writes_the_regular_key_into_wallet_json(capsys):
    cfg = FlyConfig(network="testnet", api_base="http://x")
    out = SU.keygen(cfg)
    data = SU.read_wallet("testnet")
    assert data["regular_seed"] and data["regular_address"] == out["regular_address"]
    assert data["address"] is None and data["master_seed"] is None  # faucet not run yet
    assert data["created_at"]
    assert "regular_seed" not in out and data["regular_seed"] not in json.dumps(out)
    assert keypairs.derive_classic_address(
        keypairs.derive_keypair(data["regular_seed"])[0]) == out["regular_address"]
    # a second keygen would orphan the key set on-ledger: refused unless forced
    with pytest.raises(SU.SetupError, match="force"):
        SU.keygen(cfg)
    assert SU.read_wallet("testnet")["regular_address"] == out["regular_address"]
    out2 = SU.keygen(cfg, force=True)
    assert out2["regular_address"] != out["regular_address"]
    assert SU.read_wallet("testnet")["regular_address"] == out2["regular_address"]
    assert data["regular_seed"] not in capsys.readouterr().out


def test_keygen_merges_into_a_faucet_wallet():
    write_test_wallet(regular_seed=None, regular_address=None)
    out = SU.keygen(FlyConfig(network="testnet", api_base="http://x"))
    data = SU.read_wallet("testnet")
    assert data["address"] == FLY and data["master_seed"] == MASTER_SEED
    assert data["regular_address"] == out["regular_address"]


def test_keygen_mainnet_writes_nothing():
    out = SU.keygen(FlyConfig(network="mainnet", api_base="http://x"))
    assert not paths.wallet_path("mainnet").exists()
    assert not paths.wallet_path("testnet").exists()
    assert out["regular_address"].startswith("r")
    assert keypairs.derive_classic_address(
        keypairs.derive_keypair(out["regular_seed"])[0]) == out["regular_address"]


# ------------------------------------------------------------------ faucet


def test_faucet_creates_and_stores_the_master_wallet():
    calls: list[tuple[str, dict]] = []
    fresh = Wallet.create()

    async def post(url: str, body: dict) -> dict:
        calls.append((url, body))
        return {"account": {"classicAddress": fresh.address, "address": fresh.address},
                "seed": fresh.seed, "amount": 100, "balance": 100}

    async def go():
        async with FR.FakeRpc() as rpc:
            rpc.add_account(fresh.address)
            ledger = Ledger([rpc.url], "testnet", timeout=5.0, poll_interval=0.0)
            try:
                return await SU.faucet(FlyConfig(network="testnet", api_base="http://x"),
                                       post=post, ledger=ledger, sleep=nosleep)
            finally:
                await ledger.close()

    out = run(go())
    assert calls == [(SU.FAUCET_URL, {})]
    assert out["address"] == fresh.address and out["created"] is True
    assert out["amount"] == 100 and out["funded"] is True
    assert "seed" not in json.dumps(out)
    data = SU.read_wallet("testnet")
    assert data["address"] == fresh.address and data["master_seed"] == fresh.seed
    assert data["regular_seed"] is None


def test_faucet_tops_up_an_existing_wallet():
    write_test_wallet()
    calls: list[tuple[str, dict]] = []

    async def post(url: str, body: dict) -> dict:
        calls.append((url, body))
        return {"account": {"classicAddress": FLY}, "amount": 100}

    out = run(SU.faucet(FlyConfig(network="testnet", api_base="http://x"), post=post))
    assert calls == [(SU.FAUCET_URL, {"destination": FLY})]
    assert out["address"] == FLY and out["created"] is False and out["funded"] is None
    data = SU.read_wallet("testnet")
    assert data["master_seed"] == MASTER_SEED and data["regular_seed"] == REGULAR_SEED


def test_faucet_refuses_mainnet_and_bad_answers():
    async def post(url: str, body: dict) -> dict:
        return {"error": "nope"}

    with pytest.raises(SU.SetupError, match="testnet"):
        run(SU.faucet(FlyConfig(network="mainnet", api_base="http://x"), post=post))
    with pytest.raises(SU.SetupError, match="faucet"):
        run(SU.faucet(FlyConfig(network="testnet", api_base="http://x"), post=post))
    assert not paths.wallet_path("testnet").exists()


# ------------------------------------------------------------------ regular-key


def test_set_regular_key_is_signed_by_the_master_once():
    write_test_wallet()
    cfg = FlyConfig(network="testnet", api_base="http://x")

    async def go():
        async with FR.FakeRpc() as rpc:
            rpc.add_account(FLY, sequence=7)
            apply_regular_key_on_submit(rpc)
            ledger = Ledger([rpc.url], "testnet", timeout=5.0, poll_interval=0.0)
            try:
                first = await SU.set_regular_key(cfg, ledger, sleep=nosleep)
                second = await SU.set_regular_key(cfg, ledger, sleep=nosleep)
            finally:
                await ledger.close()
            return first, second, rpc

    first, second, rpc = run(go())
    assert first["state"] == "set" and first["regular_key"] == REGULAR.address
    assert len(first["hash"]) == 64
    assert second == {"state": "already_set", "regular_key": REGULAR.address}
    (tx,) = list(rpc.txs.values())
    assert tx["TransactionType"] == "SetRegularKey" and tx["Account"] == FLY
    assert tx["RegularKey"] == REGULAR.address
    assert tx["SigningPubKey"].upper() == MASTER.public_key.upper()  # the master, this once
    assert tx["Sequence"] == 7 and int(tx["Fee"]) <= 10_000 and tx["LastLedgerSequence"]
    assert rpc.calls_for("simulate")  # the §4.3 pre-flight ran
    assert rpc.accounts[FLY]["RegularKey"] == REGULAR.address
    # the wallet file is unchanged (the ledger is the record of the key being set)
    assert SU.read_wallet("testnet")["master_seed"] == MASTER_SEED


def test_set_regular_key_refusals():
    cfg = FlyConfig(network="testnet", api_base="http://x")

    async def go(**over):
        write_test_wallet(**over)
        async with FR.FakeRpc() as rpc:
            rpc.add_account(FLY)
            ledger = Ledger([rpc.url], "testnet", timeout=5.0, poll_interval=0.0)
            try:
                return await SU.set_regular_key(cfg, ledger, sleep=nosleep)
            finally:
                await ledger.close()

    with pytest.raises(SU.SetupError, match="keygen"):
        run(go(regular_seed=None, regular_address=None))
    with pytest.raises(SU.SetupError, match="faucet"):
        run(go(master_seed=None))
    with pytest.raises(SU.SetupError, match="testnet"):
        run(SU.set_regular_key(FlyConfig(network="mainnet", api_base="http://x"), None))

    async def unfunded():
        write_test_wallet()
        async with FR.FakeRpc() as rpc:  # the account is not on the ledger
            ledger = Ledger([rpc.url], "testnet", timeout=5.0, poll_interval=0.0)
            try:
                return await SU.set_regular_key(cfg, ledger, sleep=nosleep)
            finally:
                await ledger.close()

    with pytest.raises(SU.SetupError, match="faucet"):
        run(unfunded())


# ------------------------------------------------------------------ signer from the wallet


def test_build_signer_reads_the_regular_key_from_wallet_json():
    write_test_wallet()
    cfg = FlyConfig(network="testnet", api_base="http://x")
    signer = SU.build_signer(cfg, ledger=None)
    assert signer.account == FLY and signer.key_address == REGULAR.address
    assert REGULAR_SEED not in repr(signer)


def test_build_signer_mainnet_uses_env_seed_and_wallet():
    cfg = FlyConfig(network="mainnet", api_base="http://x", regular_seed=REGULAR_SEED)
    signer = SU.build_signer(cfg, ledger=None, wallet=FLY)
    assert signer.account == FLY and signer.key_address == REGULAR.address
    with pytest.raises(SU.SetupError, match="FLY_WALLET"):
        SU.build_signer(cfg, ledger=None, wallet=None)
    with pytest.raises(SU.SetupError, match="FLY_REGULAR_SEED"):
        SU.build_signer(FlyConfig(network="mainnet", api_base="http://x"), ledger=None,
                        wallet=FLY)


def test_wallet_address_per_network():
    write_test_wallet()
    assert SU.wallet_address(FlyConfig(network="testnet", api_base="http://x")) == FLY
    main = FlyConfig(network="mainnet", api_base="http://x")
    assert SU.wallet_address(main, env={"FLY_WALLET": FLY}) == FLY
    with pytest.raises(SU.SetupError, match="FLY_WALLET"):
        SU.wallet_address(main, env={})
    with pytest.raises(SU.SetupError, match="FLY_WALLET"):
        SU.wallet_address(main, env={"FLY_WALLET": "not-an-address"})


def test_brix_pair_is_pinned_per_network():
    testnet = FlyConfig(network="testnet", api_base="http://x", signing_account=FLY)
    assert SU.brix_pair(testnet, env={}) == (SU.BRIX_CURRENCY_HEX, FLY)
    assert SU.brix_pair(testnet, env={"FLY_BRIX_ISSUER": REGULAR.address}) == (
        SU.BRIX_CURRENCY_HEX, REGULAR.address)
    main = FlyConfig(network="mainnet", api_base="http://x")
    assert SU.brix_pair(main, env={}) == (SU.BRIX_CURRENCY_HEX, SU.MAINNET_BRIX_ISSUER)
    with pytest.raises(SU.SetupError, match="BRIX"):
        SU.brix_pair(FlyConfig(network="testnet", api_base="http://x"), env={})


# ------------------------------------------------------------------ trustline


def test_trustline_signs_the_trustset_and_reports(fake_lfg):
    base, state = fake_lfg
    state.brix_issuer = state.signing_account  # staging: BRIX_ISSUER == SIGNING_ACCOUNT

    async def go():
        async with session(base, state) as s:
            first = await s.trustline()
            second = await s.trustline()
            return first, second, s.rpc

    first, second, rpc = run(go())
    assert first["state"] == "signed" and len(first["tx_hash"]) == 64
    assert second == {"state": "already_set"}
    assert state.brix_trustline_set is True
    (tx,) = list(rpc.txs.values())
    assert tx["TransactionType"] == "TrustSet" and tx["Account"] == FLY
    assert tx["LimitAmount"]["issuer"] == state.signing_account
    assert tx["SigningPubKey"].upper() == REGULAR.public_key.upper()


def test_trustline_refuses_a_foreign_pair(fake_lfg):
    base, state = fake_lfg  # state.brix_issuer is a stranger to the pinned pair

    async def go():
        async with session(base, state) as s:
            await s.trustline()

    with pytest.raises(PolicyError, match="BRIX"):
        run(go())
    assert state.brix_trustline_set is False
    row = next(r for r in state.sign_requests.values() if r["purpose"] == "tx")
    assert row["state"] == "rejected"  # the sign request was told, not left to expire


# ------------------------------------------------------------------ closet


def test_closet_accepts_until_active(fake_lfg):
    base, state = fake_lfg
    state.closet_token = {"status": "none", "nft_id": None}

    async def go():
        async with session(base, state) as s:
            out = await s.closet()
            again = await s.closet()
            return out, again, s.rpc

    out, again, rpc = run(go())
    assert out["status"] == "active" and out["nft_id"] == "C" * 64
    assert out["accepted"] is True and again["accepted"] is False
    assert state.closet_token["status"] == "active"
    assert submitted_types(rpc) == ["NFTokenAcceptOffer"]


def test_closet_retries_a_transient_error_then_gives_up_on_502(fake_lfg):
    base, state = fake_lfg
    state.closet_error = (503, {"error": "closet_mint_transient", "retryable": True})
    naps: list[float] = []

    async def clearing_sleep(seconds: float) -> None:
        naps.append(seconds)
        state.closet_error = None

    async def go():
        async with session(base, state) as s:
            s.sleep = clearing_sleep
            return await s.closet()

    out = run(go())
    assert out["status"] == "active" and naps == [SU.CLOSET_RETRY_SECONDS]

    state.closet_error = (502, {"error": "could not create or retrieve Closet"})

    async def go2():
        async with session(base, state) as s:
            await s.closet()

    with pytest.raises(SU.SetupError, match="502"):
        run(go2())


# ------------------------------------------------------------------ mint


def test_mint_one_at_a_time_until_count(fake_lfg):
    base, state = fake_lfg
    state.mint_queue += [{"nft_id": "1" * 64, "body_type": "male", "traits": {"Head": "Cap"}},
                         {"nft_id": "2" * 64, "body_type": "ape"}]

    async def go():
        async with session(base, state) as s:
            out = await s.mint(count=2)
            return out, s.rpc, s.signer.spend.spent_drops

    out, rpc, spent = run(go())
    assert out["obtained"] == 2 and out["stopped"] == "count"
    assert [m["nft_id"] for m in out["minted"]] == ["1" * 64, "2" * 64]
    assert [m["body"] for m in out["minted"]] == ["male", "ape"]
    assert state.character("1" * 64) is not None and state.character("2" * 64) is not None
    assert spent == 20_000_000  # two payments of 10 XRP, network fees not counted
    assert submitted_types(rpc) == ["Payment", "NFTokenAcceptOffer"] * 2
    payments = [t for t in rpc.txs.values() if t["TransactionType"] == "Payment"]
    assert all(t["Destination"] == state.signing_account and t["Amount"] == "10000000"
               for t in payments)
    assert all(t["SigningPubKey"].upper() == REGULAR.public_key.upper()
               for t in rpc.txs.values())
    assert all(s["state"] == "done" for s in state.mint_sessions.values())


def test_mint_stops_at_the_spend_cap(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with session(base, state, cfg_for(base, state, setup_max_xrp=15.0)) as s:
            out = await s.mint(count=3)
            return out, s.rpc

    out, rpc = run(go())
    assert out["obtained"] == 1 and out["stopped"] == "spend_cap"
    assert "FLY_SETUP_MAX_XRP" in out["reason"]
    assert submitted_types(rpc) == ["Payment", "NFTokenAcceptOffer"]
    assert len(state.characters) == 2  # the default character plus one donor
    # no second session was left dangling on the server
    assert all(s["state"] == "done" for s in state.mint_sessions.values())


def test_mint_spend_cap_zero_refuses_before_any_session(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with session(base, state, cfg_for(base, state, setup_max_xrp=0.0)) as s:
            return await s.mint(count=1)

    out = run(go())
    assert out["obtained"] == 0 and out["stopped"] == "spend_cap"
    assert state.mint_sessions == {}


def test_mint_rejects_the_payment_when_a_session_exceeds_the_cap(fake_lfg):
    base, state = fake_lfg
    state.mint_price_xrp = "40"  # the price is only known once the session exists

    async def go():
        async with session(base, state, cfg_for(base, state, setup_max_xrp=15.0)) as s:
            return await s.mint(count=1)

    out = run(go())
    assert out["obtained"] == 0 and out["stopped"] == "spend_cap"
    (row,) = [r for r in state.sign_requests.values() if r["purpose"] == "tx"]
    assert row["state"] == "rejected"


def test_mint_bulk(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with session(base, state) as s:
            out = await s.mint(count=3, bulk=True)
            return out, s.rpc, s.signer.spend.spent_drops

    out, rpc, spent = run(go())
    assert out["obtained"] == 3 and out["stopped"] == "count"
    assert len(state.bulk_jobs) == 1
    (job,) = state.bulk_jobs.values()
    assert job["quantity"] == 3 and all(u["accepted"] for u in job["units"])
    assert spent == 30_000_000
    assert submitted_types(rpc) == ["Payment"] + ["NFTokenAcceptOffer"] * 3
    payment = next(t for t in rpc.txs.values() if t["TransactionType"] == "Payment")
    assert payment["Amount"] == "30000000" and payment["Destination"] == state.signing_account
    assert len(state.characters) == 4


def test_mint_bulk_splits_jobs_at_the_bulk_max(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with session(base, state, cfg_for(base, state, setup_max_xrp=200.0)) as s:
            return await s.mint(count=12, bulk=True)

    out = run(go())
    assert out["obtained"] == 12
    assert sorted(j["quantity"] for j in state.bulk_jobs.values()) == [2, 10]


def test_mint_counts_only_the_wanted_body(fake_lfg):
    base, state = fake_lfg
    state.mint_queue += [{"body_type": "ape"}, {"body_type": "ape"}, {"body_type": "male"}]

    async def go():
        async with session(base, state) as s:
            return await s.mint(count=1, body="male")

    out = run(go())
    assert out["obtained"] == 1 and out["stopped"] == "count"
    assert [m["body"] for m in out["minted"]] == ["ape", "ape", "male"]
    assert len(out["minted"]) == 3


def test_mint_refuses_without_a_pinned_signing_account_on_mainnet(fake_lfg):
    base, state = fake_lfg

    async def go():
        cfg = FlyConfig(network="mainnet", api_base=base, signing_account=None,
                        setup_max_xrp=100.0)
        async with session(base, state, cfg) as s:
            await s.mint(count=1)

    with pytest.raises(SU.SetupError, match="FLY_LFG_SIGNING_ACCOUNT"):
        run(go())
    assert state.mint_sessions == {}


def test_mint_refuses_without_a_pinned_signing_account_on_testnet_too(fake_lfg):
    """Spec §4.3: the mint destination is pinned per network in the fly's config. It is never
    learned from LFG's own txjson (that would compare LFG's word to itself), on testnet as on
    mainnet; the rehearsal pins staging's SEED address by hand (FLY_LFG_SIGNING_ACCOUNT)."""
    base, state = fake_lfg

    async def go():
        async with session(base, state, cfg_for(base, state, signing_account=None)) as s:
            await s.mint(count=1)

    with pytest.raises(SU.SetupError, match="FLY_LFG_SIGNING_ACCOUNT"):
        run(go())
    assert state.mint_sessions == {}  # refused before any session was opened
    assert not hasattr(SU.Setup, "pinned_signing_account")


def test_mint_refuses_a_destination_that_differs_from_the_pin(fake_lfg):
    """The pin is what the signer compares LFG's txjson against: a session whose Payment
    goes anywhere else is rejected on its sign request and nothing is submitted."""
    base, state = fake_lfg
    cfg = cfg_for(base, state)  # pins today's signing account
    state.signing_account = Wallet.create().address  # LFG now quotes another destination
    seen: dict = {}

    async def go():
        async with session(base, state, cfg) as s:
            seen["rpc"] = s.rpc
            await s.mint(count=1)

    with pytest.raises(PolicyError, match="Destination"):
        run(go())
    (row,) = [r for r in state.sign_requests.values() if r["purpose"] == "tx"]
    assert row["state"] == "rejected"
    assert submitted_types(seen["rpc"]) == []


def test_mint_refuses_a_non_xrp_price(fake_lfg):
    base, state = fake_lfg
    state.mint_pay_with = "LFGO"

    async def go():
        async with session(base, state) as s:
            await s.mint(count=1)

    with pytest.raises(SU.SetupError, match="XRP"):
        run(go())
    (row,) = [r for r in state.sign_requests.values() if r["purpose"] == "tx"]
    assert row["state"] == "rejected"


def test_mint_stops_on_a_server_refusal(fake_lfg):
    base, state = fake_lfg
    state.mint_refusal = (409, {"error": "collection full", "code": "collection_full"})

    async def go():
        async with session(base, state) as s:
            return await s.mint(count=2)

    out = run(go())
    assert out["obtained"] == 0 and out["stopped"] == "refused"
    assert "collection_full" in out["reason"]


def test_mint_resumes_an_active_session(fake_lfg):
    base, state = fake_lfg
    state.mint_queue.append({"nft_id": "9" * 64, "body_type": "male"})

    async def go():
        async with session(base, state) as s:
            first = await s.client.mint()  # left over from a crashed run
            out = await s.mint(count=1)
            return first, out

    first, out = run(go())
    assert out["obtained"] == 1 and out["minted"][0]["nft_id"] == "9" * 64
    assert out["minted"][0]["session"] == first["id"]
    assert len(state.mint_sessions) == 1


async def _crash_after_paying(s: SU.Setup, state: FL.FakeLfgState) -> tuple[str, str, int]:
    """Run `mint` so that the Payment validates on the ledger and is charged, but the
    `sign_result` answer never reaches LFG (the process died there): the session stays
    `awaiting_payment` with its sign row `pending`. Returns (pay_sid, payment hash, spent)."""
    real = s.client.sign_result
    lost = {"once": True}

    async def sign_result(sid, **kw):
        row = state.sign_requests[sid]
        if lost["once"] and kw.get("tx_hash") and row["txjson"]["TransactionType"] == "Payment":
            lost["once"] = False
            raise OSError("connection reset")  # the answer never reached LFG
        return await real(sid, **kw)

    s.client.sign_result = sign_result  # type: ignore[method-assign]
    with pytest.raises(OSError):
        await s.mint(count=1)
    payments = [h for h, t in s.rpc.txs.items() if t["TransactionType"] == "Payment"]
    assert len(payments) == 1
    (sess,) = state.mint_sessions.values()
    assert sess["state"] == "awaiting_payment"
    pay_sid = LfgClient.sign_id(sess["payment_link"])
    assert state.sign_requests[pay_sid]["state"] == "pending"
    assert s.signer.spend.charged(pay_sid)
    return pay_sid, payments[0], s.signer.spend.spent_drops


def test_mint_resume_never_pays_a_charged_session_again(fake_lfg):
    """Spec §4.2 "never blindly re-submit": a resumed `awaiting_payment` session whose payment
    is on the spend ledger is not paid again; the resume waits for LFG to detect the
    payment on-ledger (docs/lfg-api.md §7), then carries on to the accept."""
    base, state = fake_lfg
    state.mint_queue.append({"nft_id": "9" * 64, "body_type": "male"})

    async def go():
        async with session(base, state) as s:
            pay_sid, digest, spent_after_crash = await _crash_after_paying(s, state)

            async def lfg_detects_the_payment(_seconds: float) -> None:
                if state.sign_requests[pay_sid]["state"] == "pending":
                    # LFG's own watcher, off the ledger (the wrapper passes this through)
                    await s.client.sign_result(pay_sid, tx_hash=digest)

            s.sleep = lfg_detects_the_payment
            out = await s.mint(count=1)
            return out, s.rpc, spent_after_crash, s.signer.spend.spent_drops

    out, rpc, spent_after_crash, spent = run(go())
    assert out["obtained"] == 1 and out["minted"][0]["nft_id"] == "9" * 64
    assert submitted_types(rpc) == ["Payment", "NFTokenAcceptOffer"]  # exactly one Payment
    assert spent == spent_after_crash == 10_000_000
    assert len(state.mint_sessions) == 1
    assert state.mint_sessions[out["minted"][0]["session"]]["state"] == "done"


def test_mint_resume_of_a_charged_session_fails_clearly_if_lfg_never_sees_it(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with session(base, state) as s:
            pay_sid, _digest, spent_after_crash = await _crash_after_paying(s, state)
            s.payment_timeout = 0.5  # LFG's watcher never comes
            with pytest.raises(SU.SetupError, match="already charged") as info:
                await s.mint(count=1)
            return pay_sid, str(info.value), s.rpc, spent_after_crash, s.signer.spend.spent_drops

    pay_sid, message, rpc, spent_after_crash, spent = run(go())
    assert pay_sid in message and "cancel" in message
    assert submitted_types(rpc) == ["Payment"]  # still exactly one
    assert spent == spent_after_crash == 10_000_000
    (sess,) = state.mint_sessions.values()
    assert sess["state"] == "awaiting_payment"
    assert state.sign_requests[pay_sid]["state"] == "pending"  # never rejected either


# ------------------------------------------------------------------ accept (§5 step 3(b))

DONOR = Wallet.from_seed(seed_for("operator-donor")).address
STRANGER = Wallet.from_seed(seed_for("a-stranger")).address


def nft(tag: int) -> str:
    """A 64-hex NFTokenID: flags/fee, a 20-byte issuer, taxon, serial `tag`."""
    return f"00080000{'11' * 20}{0:08X}{tag:08X}"


def offer(index: str, *, owner: str, amount: str = "0", destination: str = FLY) -> dict:
    return {"nft_offer_index": index, "amount": amount, "flags": 1, "owner": owner,
            "destination": destination}


def accept_cfg(**over) -> FlyConfig:
    kw = dict(network="testnet", api_base="http://lfg.invalid", donor_sources=(DONOR,),
              expected_ledger_hash=FR.TESTNET_HASH, setup_max_xrp=0.0)
    kw.update(over)
    return FlyConfig(**kw)


async def run_accept(cfg: FlyConfig, nft_ids: list[str], prepare) -> tuple[dict, FR.FakeRpc]:
    async with FR.FakeRpc() as rpc:
        rpc.add_account(FLY, sequence=10)
        prepare(rpc)
        ledger = Ledger([rpc.url], "testnet", timeout=5.0, poll_interval=0.0)
        try:
            signer = Signer(REGULAR_SEED, FLY, ledger, cfg)
            return await SU.accept(cfg, ledger, signer, nft_ids), rpc
        finally:
            await ledger.close()


def test_accept_takes_a_donors_offer_and_skips_every_other():
    """Spec §5 step 3(b): the operator's destination-locked, zero-price offer is accepted
    on-ledger with the fly's key, no LFG session; an offer from a wallet outside
    FLY_DONOR_SOURCES, a priced one, an NFT already owned and a malformed id are reported,
    never signed."""
    donated, foreign, priced, owned, unlocked = nft(1), nft(2), nft(3), nft(4), nft(5)
    o1, o2, o3, o5 = ("A1" * 32, "B2" * 32, "C3" * 32, "E5" * 32)

    def prepare(rpc: FR.FakeRpc) -> None:
        rpc.add_nft(FLY, owned, uri=None)
        rpc.offers[donated] = [offer(o1, owner=DONOR)]
        rpc.offers[foreign] = [offer(o2, owner=STRANGER)]
        rpc.offers[priced] = [offer(o3, owner=DONOR, amount="1000000")]
        rpc.offers[unlocked] = [offer(o5, owner=DONOR, destination=STRANGER)]

    out, rpc = run(run_accept(accept_cfg(), [donated, foreign, priced, owned, unlocked, "nope"],
                              prepare))
    assert [a["nft_id"] for a in out["accepted"]] == [donated]
    (acc,) = out["accepted"]
    assert acc["offer"] == o1 and acc["owner"] == DONOR and len(acc["hash"]) == 64
    assert submitted_types(rpc) == ["NFTokenAcceptOffer"]
    (tx,) = rpc.txs.values()
    assert tx["NFTokenSellOffer"] == o1 and tx["Account"] == FLY
    assert acc["hash"] == tx["hash"]
    skipped = {s["nft_id"]: s["why"] for s in out["skipped"]}
    assert set(skipped) == {foreign, priced, owned, unlocked, "nope"}
    assert STRANGER in skipped[foreign] and "FLY_DONOR_SOURCES" in skipped[foreign]
    assert "1000000" in skipped[priced]
    assert skipped[owned] == "already owned by the fly"
    assert "no sell offers" not in skipped[unlocked] and STRANGER in skipped[unlocked]
    assert "NFTokenID" in skipped["nope"]
    # no LFG was involved, and the sell offers were read on-ledger (twice for the accepted
    # one: once to pick it, once by the signer's own policy)
    assert len(rpc.calls_for("nft_sell_offers")) == 5


def test_accept_refuses_without_donor_sources_or_ids():
    async def go(cfg, ids):
        async with FR.FakeRpc() as rpc:
            rpc.add_account(FLY)
            ledger = Ledger([rpc.url], "testnet", timeout=5.0, poll_interval=0.0)
            try:
                signer = Signer(REGULAR_SEED, FLY, ledger, cfg)
                await SU.accept(cfg, ledger, signer, ids)
            finally:
                await ledger.close()
            return rpc

    with pytest.raises(SU.SetupError, match="FLY_DONOR_SOURCES"):
        run(go(accept_cfg(donor_sources=()), [nft(1)]))
    with pytest.raises(SU.SetupError, match="NFTokenID"):
        run(go(accept_cfg(), []))


def test_accept_lets_the_signer_refuse_a_stale_offer():
    """The signer re-reads the offer on-ledger under the §4.3 policy: an offer that changed
    between the pick and the signature (here: gone) is a PolicyError, not a submission."""
    donated = nft(7)

    def prepare(rpc: FR.FakeRpc) -> None:
        reads = {"n": 0}
        good = [offer("D7" * 32, owner=DONOR)]

        def flaky(params: dict) -> dict:
            reads["n"] += 1
            if reads["n"] == 1:
                return {"nft_id": params["nft_id"], "offers": good}
            raise FR.RpcError("objectNotFound", "The requested object was not found.")

        rpc.answer("nft_sell_offers", flaky)

    with pytest.raises(PolicyError, match="not on the ledger"):
        run(run_accept(accept_cfg(), [donated], prepare))


# ------------------------------------------------------------------ harvest


def test_harvest_all_keeps_the_hero(fake_lfg):
    base, state = fake_lfg
    state.add_character("B" * 64, body="male", traits={"Head": "Cap", "Eyes": "Laser"})
    state.add_character("D" * 64, body="ape", traits={"Head": "Crown"})
    state.add_character("E" * 64, body="male", mutable=False, traits={"Head": "Hat"})

    async def go():
        async with session(base, state) as s:
            return await s.harvest(all_=True)

    out = run(go())
    assert out["hero"] == "A" * 64
    assert [h["nft_id"] for h in out["harvested"]] == ["B" * 64, "D" * 64]
    assert all(h["state"] == "done" for h in out["harvested"])
    assert out["skipped"] == [{"nft_id": "E" * 64, "why": "not mutable"}]
    assert state.character("B" * 64)["blank"] and state.character("D" * 64)["blank"]
    assert not state.character("A" * 64)["blank"]
    assert state.closet_count("Head", "Cap") == 1 and state.closet_count("Body", "ape") == 1
    assert state.closet_count("Background", "None") == 2  # empty donor slots credit None units


def test_harvest_named_ids_and_explicit_hero(fake_lfg):
    base, state = fake_lfg
    state.add_character("B" * 64, body="male", traits={"Head": "Cap"})

    async def go():
        async with session(base, state) as s:
            out = await s.harvest(nft_ids=["A" * 64], hero="B" * 64)
            with pytest.raises(SU.SetupError, match="hero"):
                await s.harvest(nft_ids=["B" * 64], hero="B" * 64)
            with pytest.raises(SU.SetupError, match="all"):
                await s.harvest()
            return out

    out = run(go())
    assert [h["nft_id"] for h in out["harvested"]] == ["A" * 64]
    assert state.character("A" * 64)["blank"] and not state.character("B" * 64)["blank"]


def test_harvest_needs_an_active_closet(fake_lfg):
    base, state = fake_lfg
    state.closet_token = {"status": "none", "nft_id": None}
    state.add_character("B" * 64, body="male")

    async def go():
        async with session(base, state) as s:
            await s.harvest(all_=True)

    with pytest.raises(SU.SetupError, match="closet"):
        run(go())
    assert state.harvest_sessions == {}


def test_harvest_failed_session_is_reported_not_raised(fake_lfg):
    base, state = fake_lfg
    state.add_character("B" * 64, body="male")
    state.harvest_outcome = "failed"

    async def go():
        async with session(base, state) as s:
            return await s.harvest(all_=True)

    out = run(go())
    assert out["harvested"][0]["state"] == "failed" and out["harvested"][0]["error"]


class StubClient:
    """A legacy-path harvest: LFG burns, remints and hands back an accept to sign."""

    def __init__(self, new_nft_id: str):
        self.new_nft_id = new_nft_id
        self.calls: list[tuple] = []
        self.sid = "hv-legacy"
        self.link = "lfg-wc://wc-" + "ab" * 16
        self._polls = 0

    async def economy(self):
        return {"characters": [{"nft_id": "A" * 64, "body": "male", "mutable": False,
                                "blank": False, "attributes": []}],
                "closet": {"assets": [], "token": {"status": "active", "nft_id": "C" * 64}}}

    async def harvest(self, nft_id):
        self.calls.append(("harvest", nft_id))
        return {"id": self.sid, "kind": "harvest", "state": "running", "error": None,
                "moved_assets": [], "accept": self.link, "accept_push": None,
                "new_nft_id": self.new_nft_id, "platform": "web"}

    async def sign_request(self, sid):
        self.calls.append(("sign_request", sid))
        return {"id": sid, "state": "pending", "expires_at": 9e9, "txid": None,
                "txjson": {"TransactionType": "NFTokenAcceptOffer", "Account": FLY,
                           "NFTokenSellOffer": "AB" * 32}}

    async def sign_result(self, sid, *, tx_hash=None, rejected=False, error=None):
        self.calls.append(("sign_result", sid, tx_hash, rejected, error))
        return {"state": "signed", "txid": tx_hash}

    async def harvest_status(self, sid):
        self._polls += 1
        state = "done" if self._polls > 1 else "running"
        return {"id": sid, "kind": "harvest", "state": state, "error": None,
                "moved_assets": [["Head", "Cap"]], "accept": self.link,
                "new_nft_id": self.new_nft_id}

    async def wait(self, getter, sid, terminal, timeout, every=3.0):
        while True:
            status = await getter(sid)
            if status.get("state") in terminal:
                return status


class StubSigner:
    account = FLY

    def __init__(self):
        self.signed: list[tuple[dict, object]] = []

    async def sign_and_submit(self, tx_json, purpose):
        self.signed.append((dict(tx_json), purpose))
        return {"hash": "CD" * 32, "result": {}}


def test_harvest_legacy_accept_is_signed_under_accept_offer():
    new_id = "000B0539C35B55AA096BA6D87A6E6C965A6534150DC56E5E12C5D09E0000000C"
    client, signer = StubClient(new_id), StubSigner()
    cfg = FlyConfig(network="testnet", api_base="http://x", signing_account=FLY)
    s = SU.Setup(cfg, ledger=None, client=client, signer_factory=lambda c: signer,
                 sleep=nosleep, poll_every=0.0)
    out = run(s.harvest(nft_ids=["A" * 64], hero="Z" * 64))
    assert out["harvested"][0]["state"] == "done"
    assert out["harvested"][0]["new_nft_id"] == new_id
    ((tx, purpose),) = signer.signed
    assert tx["TransactionType"] == "NFTokenAcceptOffer"
    assert purpose.kind == "accept_offer" and purpose.nft_id == new_id
    assert FLY in purpose.allowed_owners  # LFG's signing account may own the delivery offer
    assert ("sign_result", "wc-" + "ab" * 16, "CD" * 32, False, None) in client.calls


# ------------------------------------------------------------------ status


def test_status_reports_wallet_ledger_and_wardrobe(fake_lfg):
    base, state = fake_lfg
    state.add_closet("Head", "Cap", 2)
    state.brix["claimable"] = 3

    async def go():
        async with session(base, state) as s:
            s.rpc.accounts[FLY]["RegularKey"] = REGULAR.address
            return await s.status()

    out = run(go())
    assert out["network"] == "testnet" and out["wallet"] == FLY
    assert out["balance_xrp"] == "100" and out["regular_key"] == REGULAR.address
    assert out["regular_key_matches"] is True
    assert out["characters"] == [{
        "nft_id": "A" * 64, "body": "male", "mutable": True, "blank": False,
        "look": ["Blue", "None", "male", "Hoodie", "None", "None", "Laser", "Crown", "None"],
    }]
    assert out["closet"] == {"status": "active", "nft_id": "C" * 64, "units": 2}
    assert out["spend"] == {"spent_xrp": "0", "cap_xrp": "100"}
    assert out["brix"]["claimable"] == 3
    assert "seed" not in json.dumps(out).lower()


def test_status_tolerates_an_unfunded_wallet(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with session(base, state) as s:
            del s.rpc.accounts[FLY]
            return await s.status()

    out = run(go())
    assert out["balance_xrp"] is None and out["regular_key"] is None
    assert out["regular_key_matches"] is False


# ------------------------------------------------------------------ CLI


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fly")
    cli_setup.register(p.add_subparsers(dest="command"))
    return p


@pytest.mark.parametrize("argv, expect", [
    (["setup", "keygen"], {"setup_command": "keygen", "force": False}),
    (["setup", "keygen", "--force"], {"setup_command": "keygen", "force": True}),
    (["setup", "faucet"], {"setup_command": "faucet"}),
    (["setup", "regular-key"], {"setup_command": "regular-key"}),
    (["setup", "trustline"], {"setup_command": "trustline"}),
    (["setup", "closet"], {"setup_command": "closet"}),
    (["setup", "mint", "--count", "3"], {"setup_command": "mint", "count": 3, "bulk": False,
                                          "body": None}),
    (["setup", "mint", "--count", "3", "--bulk", "--body", "male"],
     {"setup_command": "mint", "count": 3, "bulk": True, "body": "male"}),
    (["setup", "harvest", "--all"], {"setup_command": "harvest", "all": True, "nft_ids": []}),
    (["setup", "harvest", "A" * 64, "B" * 64], {"setup_command": "harvest", "all": False,
                                                 "nft_ids": ["A" * 64, "B" * 64]}),
    (["setup", "status"], {"setup_command": "status"}),
])
def test_register_adds_every_subcommand(argv, expect):
    args = parser().parse_args(argv)
    assert args.command == "setup" and callable(args.func)
    for key, value in expect.items():
        assert getattr(args, key) == value


def test_cli_refuses_bad_usage():
    with pytest.raises(SystemExit):
        parser().parse_args(["setup"])
    with pytest.raises(SystemExit):
        parser().parse_args(["setup", "mint"])  # --count is required
    with pytest.raises(SystemExit):
        parser().parse_args(["setup", "harvest", "--all", "A" * 64])


def test_cli_dispatches_to_the_named_step(monkeypatch):
    seen: list[str] = []

    def fake_step(args):
        seen.append(args.setup_command)
        return 0

    for name in cli_setup.COMMANDS:
        monkeypatch.setitem(cli_setup.COMMANDS, name, fake_step)
    for argv in (["setup", "faucet"], ["setup", "mint", "--count", "1"], ["setup", "status"]):
        args = parser().parse_args(argv)
        assert args.func(args) == 0
    assert seen == ["faucet", "mint", "status"]


def test_cli_keygen_testnet_prints_the_address_only(monkeypatch, capsys):
    monkeypatch.setenv("FLY_NETWORK", "testnet")
    args = parser().parse_args(["setup", "keygen"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    data = SU.read_wallet("testnet")
    assert data["regular_address"] in out and data["regular_seed"] not in out
    assert str(paths.wallet_path("testnet")) in out


def test_cli_keygen_mainnet_prints_the_env_line_once(monkeypatch, capsys):
    monkeypatch.setenv("FLY_NETWORK", "mainnet")
    monkeypatch.delenv("FLY_REGULAR_SEED", raising=False)
    args = parser().parse_args(["setup", "keygen"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("FLY_REGULAR_SEED="))
    seed = line.split("=", 1)[1]
    addr = keypairs.derive_classic_address(keypairs.derive_keypair(seed)[0])
    assert addr in out and "Xaman" in out
    assert not paths.wallet_path("mainnet").exists()


def test_cli_exit_codes(monkeypatch, capsys):
    monkeypatch.setenv("FLY_NETWORK", "testnet")
    repo = paths.repo_root()
    monkeypatch.setenv("FLY_DATA_DIR", str(repo / "fly-data-should-not-exist"))
    args = parser().parse_args(["setup", "keygen"])
    assert args.func(args) == 2  # refused: wallet.json inside a repo
    assert not (repo / "fly-data-should-not-exist").exists()
    assert "repo" in capsys.readouterr().err

    async def failing(*_a, **_k):
        raise SU.SetupError("run `fly setup faucet` first")

    monkeypatch.setattr(cli_setup, "_run_session", failing)
    args = parser().parse_args(["setup", "status"])
    assert args.func(args) == 1
    assert "faucet" in capsys.readouterr().err


def test_cli_accept_takes_ids_and_needs_donor_sources(monkeypatch, capsys):
    args = parser().parse_args(["setup", "accept", nft(1), nft(2)])
    assert args.setup_command == "accept" and args.nft_ids == [nft(1), nft(2)]
    with pytest.raises(SystemExit):
        parser().parse_args(["setup", "accept"])
    monkeypatch.setenv("FLY_NETWORK", "testnet")
    monkeypatch.delenv("FLY_DONOR_SOURCES", raising=False)
    assert args.func(args) == 1  # refused before the ledger is opened
    assert "FLY_DONOR_SOURCES" in capsys.readouterr().err

    seen: dict = {}

    async def fake_run_ledger(cfg, fn):
        seen["cfg"] = cfg
        return seen["answer"]

    monkeypatch.setattr(cli_setup, "_run_ledger", fake_run_ledger)
    monkeypatch.setenv("FLY_DONOR_SOURCES", DONOR)
    seen["answer"] = {"accepted": [{"nft_id": nft(1), "offer": "A1" * 32, "owner": DONOR,
                                    "hash": "F" * 64}],
                      "skipped": [{"nft_id": nft(2), "why": "no sell offers on-ledger"}]}
    assert args.func(args) == 1  # a skipped id is never silent
    out = capsys.readouterr().out
    assert nft(1) in out and "no sell offers" in out and seen["cfg"].donor_sources == (DONOR,)
    seen["answer"] = {"accepted": [{"nft_id": nft(1), "offer": "A1" * 32, "owner": DONOR,
                                    "hash": "F" * 64}], "skipped": []}
    assert args.func(args) == 0


def test_cli_tests_never_read_the_operators_dotenv(monkeypatch):
    """The autouse conftest fixture points every CLI's DOTENV at a missing file, so a real
    ~/lfg-fly/.env (the RegularKey seed, FLY_ENABLED=1) can never leak into the test
    process (contract: tests never touch ~/fly-data or the network)."""
    from lfg_fly.body import cli_daily, cli_move

    for module in (cli_setup, cli_daily, cli_move):
        assert not module.DOTENV.exists()
        assert module.DOTENV != Path.home() / "lfg-fly" / ".env"
    monkeypatch.setenv("FLY_NETWORK", "testnet")
    monkeypatch.delenv("FLY_REGULAR_SEED", raising=False)
    parser().parse_args(["setup", "keygen"]).func(parser().parse_args(["setup", "keygen"]))
    assert "FLY_REGULAR_SEED" not in os.environ


def test_load_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# the fly\nFLY_NETWORK=mainnet\nexport FLY_WALLET='rXYZ'\n"
        'FLY_REGULAR_SEED="sEdFAKE"\n\nBAD LINE\nFLY_BETA=0.4 # trailing\n'
    )
    env = {"FLY_NETWORK": "testnet"}
    loaded = cli_setup.load_env_file(env_file, env)
    assert env["FLY_NETWORK"] == "testnet"  # the environment wins over the file
    assert env["FLY_WALLET"] == "rXYZ" and env["FLY_REGULAR_SEED"] == "sEdFAKE"
    assert env["FLY_BETA"] == "0.4"
    assert loaded == ["FLY_WALLET", "FLY_REGULAR_SEED", "FLY_BETA"]
    assert cli_setup.load_env_file(tmp_path / "missing", env) == []


def test_session_wiring_signs_in_and_out(fake_lfg, monkeypatch):
    """`_run_session` builds the ledger, client and signer and always logs out."""
    base, state = fake_lfg
    write_test_wallet()
    state.regular_keys[FLY] = REGULAR.address
    seen: dict = {}

    async def step(s: SU.Setup) -> dict:
        seen["token"] = s.client.token
        seen["wallet"] = s.client.wallet
        return {"ok": True}

    async def go():
        async with FR.FakeRpc() as rpc:
            rpc.add_account(FLY)
            cfg = replace(cfg_for(base, state), rpc_urls=(rpc.url,))
            return await cli_setup._run_session(cfg, step)

    out = run(go())
    assert out == {"ok": True} and seen["wallet"] == FLY and seen["token"]
    assert state.logouts == 1 and state.tokens == {}


def test_session_wiring_refuses_a_wrong_chain(fake_lfg):
    base, state = fake_lfg
    write_test_wallet()

    async def step(s: SU.Setup) -> dict:
        raise AssertionError("must not run")

    async def go():
        async with FR.FakeRpc(hash_32570=FR.OTHER_HASH) as rpc:
            cfg = replace(cfg_for(base, state), rpc_urls=(rpc.url,))
            return await cli_setup._run_session(cfg, step)

    from lfg_fly.body.chain import ChainIdentityError

    with pytest.raises(ChainIdentityError):
        run(go())
    assert state.signin_starts == 0


def test_no_seed_is_ever_written_outside_wallet_json(fake_lfg):
    """Sanity: after a whole rehearsal, the only file under FLY_DATA_DIR holding a seed is
    wallet.json, and no seed appears on stdout."""
    base, state = fake_lfg
    write_test_wallet()
    state.closet_token = {"status": "none", "nft_id": None}
    state.brix_issuer = state.signing_account

    async def go():
        async with session(base, state) as s:
            await s.trustline()
            await s.closet()
            await s.mint(count=1)
            await s.harvest(all_=True)
            await s.status()

    run(go())
    root = paths.data_dir()
    for path in root.rglob("*"):
        if path.is_file() and path != paths.wallet_path("testnet"):
            text = path.read_text(errors="replace")
            assert MASTER_SEED not in text and REGULAR_SEED not in text, path
    assert os.path.exists(paths.network_dir("testnet") / "setup-spend.json")


def test_cli_reports_an_lfg_refusal_cleanly(monkeypatch, capsys):
    """A 503 agent_disabled from LFG is a one-line message and exit 1, not a traceback."""
    from lfg_fly.body.client import LfgError

    monkeypatch.setenv("FLY_NETWORK", "testnet")

    async def refused(*_a, **_k):
        raise LfgError(503, "agent_disabled", {"code": "agent_disabled"})

    monkeypatch.setattr(cli_setup, "_run_session", refused)
    args = parser().parse_args(["setup", "status"])
    assert args.func(args) == 1
    err = capsys.readouterr().err
    assert "503 agent_disabled" in err and "AGENT_SIGNIN_ENABLED" in err
