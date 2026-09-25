"""fly-claim and fly-retrain (spec §4 table; §4.1 step 1 fresh sign-in, logout in finally;
§4.4 chain identity once per process; §5.3 claims_disabled on staging), the `claim` /
`retrain` CLI, `python -m lfg_fly`, and ecosystem.config.js (spec §1 CPU discipline).

Hermetic: LFG is `tests/fake_lfg.py` (a thread), the ledger is `tests/fake_rpc.py` (an
aiohttp TestServer inside the test's own `asyncio.run`), the brain is a fake with one-hot
features, and FLY_DATA_DIR is the autouse tmp dir. No pytest-asyncio: `asyncio.run`.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys

import fake_lfg as FL
import fake_rpc as FR
import numpy as np
import pytest
from xrpl.wallet import Wallet

from lfg_fly import paths
from lfg_fly.body import cli_daily as CD
from lfg_fly.body import daily as D
from lfg_fly.body import outbox
from lfg_fly.body.chain import ChainIdentityError, Ledger
from lfg_fly.body.client import LfgClient
from lfg_fly.body.config import FlyConfig
from lfg_fly.body.records import Stamp, StampMismatch
from lfg_fly.brain import readout as R
from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher.catalog import Catalog

fake_lfg = FL.fake_lfg  # the shared fixture, bound by name (see fake_lfg's docstring)

run = asyncio.run


# ------------------------------------------------------------------ fixtures


@dataclasses.dataclass
class Keys:
    master: Wallet
    regular: Wallet

    @property
    def account(self) -> str:
        return self.master.address


@pytest.fixture
def keys(fake_lfg) -> Keys:
    """A fly wallet with its RegularKey registered on the fake's ledger, and a testnet
    wallet.json holding the RegularKey seed (what `fly setup keygen` writes)."""
    _, state = fake_lfg
    k = Keys(Wallet.create(), Wallet.create())
    state.regular_keys[k.account] = k.regular.address
    path = paths.wallet_path("testnet")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"network": "testnet", "account": k.account,
                                "master_seed": k.master.seed, "regular_seed": k.regular.seed}))
    os.chmod(path, 0o600)
    return k


@pytest.fixture(autouse=True)
def _fresh_identity_cache():
    D.reset_identity_cache()
    yield
    D.reset_identity_cache()


def cfg_for(base: str, rpc: FR.FakeRpc, **over) -> FlyConfig:
    fields = dict(network="testnet", api_base=base, rpc_urls=(rpc.url,),
                  expected_ledger_hash=FR.TESTNET_HASH, enabled=True)
    fields.update(over)
    return FlyConfig(**fields)


def files_under(root) -> list[str]:
    out = []
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            out.append(os.path.join(dirpath, n))
    return out


def assert_no_token_on_disk(state: FL.FakeLfgState, root) -> None:
    """Every token the fake ever issued (it forgets them itself on logout) is absent from
    every file under FLY_DATA_DIR: the alerts embed LFG's answers and the retrain meta embeds
    the stamp, and neither may ever grow a token (§4.4, §7)."""
    tokens = state.issued_tokens
    assert tokens, "no session token was issued"
    files = files_under(root)
    assert files, "nothing was written, so nothing was checked"
    for path in files:
        with open(path, "rb") as f:
            data = f.read()
        for tok in tokens:
            assert tok.encode() not in data, f"session token written to {path}"
        assert b"Bearer" not in data and b"session_token" not in data, path


# ------------------------------------------------------------------ credentials


def test_load_credentials_reads_the_testnet_wallet_json(fake_lfg, keys):
    base, _ = fake_lfg
    cfg = FlyConfig(network="testnet", api_base=base)
    creds = D.load_credentials(cfg)
    assert creds.account == keys.account
    assert creds.seed == keys.regular.seed
    assert keys.regular.seed not in repr(creds) and keys.regular.seed not in str(creds)


def test_load_credentials_refuses_a_missing_or_foreign_wallet_file(fake_lfg):
    base, _ = fake_lfg
    cfg = FlyConfig(network="testnet", api_base=base)
    with pytest.raises(D.CredentialsError) as e:
        D.load_credentials(cfg)
    assert "wallet.json" in str(e.value) and "keygen" in str(e.value)
    path = paths.wallet_path("testnet")
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Wallet.create()
    path.write_text(json.dumps({"network": "mainnet", "account": w.address,
                                "regular_seed": Wallet.create().seed}))
    with pytest.raises(D.CredentialsError):
        D.load_credentials(cfg)
    path.write_text(json.dumps({"network": "testnet", "account": w.address}))
    with pytest.raises(D.CredentialsError):
        D.load_credentials(cfg)
    path.write_text("not json")
    with pytest.raises(D.CredentialsError) as e:
        D.load_credentials(cfg)
    assert "not json" not in str(e.value)  # never echoes the file's contents


def test_load_credentials_mainnet_reads_the_seed_from_config_and_the_wallet_from_env():
    regular = Wallet.create()
    master = Wallet.create()
    cfg = FlyConfig(network="mainnet", api_base="http://localhost:8176", regular_seed=regular.seed)
    creds = D.load_credentials(cfg, env={"FLY_WALLET": master.address})
    assert (creds.account, creds.seed) == (master.address, regular.seed)
    with pytest.raises(D.CredentialsError):
        D.load_credentials(cfg, env={})
    with pytest.raises(D.CredentialsError):
        D.load_credentials(cfg, env={"FLY_WALLET": "not-an-address"})
    no_seed = FlyConfig(network="mainnet", api_base="http://localhost:8176")
    with pytest.raises(D.CredentialsError) as e:
        D.load_credentials(no_seed, env={"FLY_WALLET": master.address})
    assert "FLY_REGULAR_SEED" in str(e.value)


def test_load_env_file_sets_only_missing_keys(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# the fly's secrets\n"
        "FLY_NETWORK=mainnet\n"
        "export FLY_API_BASE='http://localhost:8176'\n"
        'FLY_WALLET="rExample"  \n'
        "\n"
        "BROKEN LINE\n"
        "FLY_BETA=0.4 # trailing comment\n"
    )
    env = {"FLY_NETWORK": "testnet"}
    added = D.load_env_file(env_file, env)
    assert env["FLY_NETWORK"] == "testnet"  # the environment wins over the file
    assert env["FLY_API_BASE"] == "http://localhost:8176"
    assert env["FLY_WALLET"] == "rExample"
    assert env["FLY_BETA"] == "0.4"
    assert set(added) == {"FLY_API_BASE", "FLY_WALLET", "FLY_BETA"}
    assert D.load_env_file(tmp_path / "absent", env) == {}


# ------------------------------------------------------------------ session / identity


def test_session_checks_the_chain_once_per_process_then_signs_in_and_logs_out(fake_lfg, keys):
    base, state = fake_lfg

    async def go():
        async with FR.FakeRpc() as rpc:
            cfg = cfg_for(base, rpc)
            async with D.session(cfg) as (ledger, client):
                assert isinstance(ledger, Ledger) and isinstance(client, LfgClient)
                assert client.token and client.wallet == keys.account
                me = await client.me()
                assert me["wallet"] == keys.account
                token = client.token
            assert client.token is None
            async with D.session(cfg) as (_ledger, client2):
                assert client2.token and client2.token != token
            return len(rpc.calls_for("server_info")), len(rpc.calls_for("ledger"))

    server_info, ledger_calls = run(go())
    assert server_info == 1  # §4.4: once per process, not once per session
    assert state.logouts == 2
    assert state.sessions[-1]["key"] == "regular"
    assert state.sessions[-1]["signer"] == keys.regular.address


def test_session_refuses_the_wrong_chain_before_any_sign_in(fake_lfg, keys):
    base, state = fake_lfg

    async def go():
        async with FR.FakeRpc(hash_32570=FR.OTHER_HASH) as rpc:
            cfg = cfg_for(base, rpc)
            async with D.session(cfg):
                pass

    with pytest.raises(ChainIdentityError):
        run(go())
    assert state.signin_starts == 0

    async def wrong_network():
        async with FR.FakeRpc(network_id=0, hash_32570=FR.MAINNET_HASH) as rpc:
            cfg = cfg_for(base, rpc)
            async with D.session(cfg):
                pass

    with pytest.raises(ChainIdentityError):
        run(wrong_network())
    assert state.signin_starts == 0


def test_session_logs_out_when_the_body_raises(fake_lfg, keys):
    base, state = fake_lfg

    async def go():
        async with FR.FakeRpc() as rpc:
            async with D.session(cfg_for(base, rpc)) as (_ledger, client):
                assert client.token
                raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run(go())
    assert state.logouts == 1
    assert not state.tokens


# ------------------------------------------------------------------ claim


async def do_claim(base: str, **kw) -> tuple[D.ClaimResult, FR.FakeRpc]:
    async with FR.FakeRpc() as rpc:
        result = await D.claim(cfg_for(base, rpc), **kw)
        return result, rpc


def test_claim_confirmed(fake_lfg, keys, _data_dir):
    base, state = fake_lfg
    state.brix_trustline_set = True
    state.brix["claimable"] = 5

    result, _ = run(do_claim(base))
    assert result.outcome == "confirmed" and result.exit_code == 0
    assert result.claim["amount"] == 5 and result.claim["tx_hash"]
    assert result.claim["claim_id"] in result.message and "5" in result.message
    assert result.alert is None
    assert outbox.alerts("testnet") == []
    assert state.brix["claimable"] == 0 and state.brix["claimed_total"] == 5
    # §4.1 step 1: a fresh agent session, RegularKey proof, logout in finally
    assert state.logouts == 1 and not state.tokens
    assert state.sessions[-1]["provider"] == "agent" and state.sessions[-1]["key"] == "regular"
    assert ("POST", "/api/brix/claim") in state.requests
    assert_no_token_on_disk(state, _data_dir)


def test_claim_claims_disabled_alerts_and_exits_zero(fake_lfg, keys):
    """Spec §5.3: staging can't pay BRIX today (503 claims_disabled). That is the operator's
    prerequisite, not the fly's failure: an outbox alert, a clear message, exit 0."""
    base, state = fake_lfg
    state.brix_claims_enabled = False
    state.brix_trustline_set = True
    state.brix["claimable"] = 5

    result, _ = run(do_claim(base))
    assert result.outcome == "disabled" and result.exit_code == 0
    assert "claims_disabled" in result.message and "BRIX_DISTRIBUTOR_SEED" in result.message
    alerts = outbox.alerts("testnet")
    assert len(alerts) == 1 and alerts[0]["path"] == result.alert
    payload = alerts[0]["payload"]
    assert payload["job"] == "fly-claim" and payload["code"] == "claims_disabled"
    assert payload["status"] == 503
    assert alerts[0]["network"] == "testnet"
    assert state.logouts == 1
    for tok in state.sessions:
        assert tok["jti"] not in json.dumps(payload)


def test_claim_refuses_when_not_enabled(fake_lfg, keys):
    """Spec §4.4 kill procedure step 2 / §5 step 6: without FLY_ENABLED=1 the job does
    nothing at all; not the chain, not the key, not LFG."""
    base, state = fake_lfg
    state.brix_trustline_set = True
    state.brix["claimable"] = 5

    async def go():
        async with FR.FakeRpc() as rpc:
            with pytest.raises(D.NotEnabled, match="FLY_ENABLED") as info:
                await D.claim(cfg_for(base, rpc, enabled=False))
            assert isinstance(info.value, D.DailyError)
            return rpc

    rpc = run(go())
    assert rpc.calls == [] and state.requests == [] and state.signin_starts == 0
    assert state.brix["claimable"] == 5 and outbox.alerts("testnet") == []


def test_claim_nothing_to_claim_is_quiet(fake_lfg, keys):
    base, state = fake_lfg
    state.brix_trustline_set = True
    state.brix["claimable"] = 0

    result, _ = run(do_claim(base))
    assert result.outcome == "nothing" and result.exit_code == 0
    assert "nothing" in result.message.lower()
    assert outbox.alerts("testnet") == [] and result.alert is None
    assert state.logouts == 1


def test_claim_polls_a_submitted_claim_to_its_end(fake_lfg, keys):
    base, state = fake_lfg
    state.brix_trustline_set = True
    state.brix["claimable"] = 7
    state.brix_claim_outcome = "submitted"

    async def flip():
        while not state.brix_claims:
            await asyncio.sleep(0.005)
        while not any(m == "GET" and p.startswith("/api/brix/claim/") for m, p in state.requests):
            await asyncio.sleep(0.005)
        for c in state.brix_claims.values():
            c["state"] = "confirmed"

    async def go():
        async with FR.FakeRpc() as rpc:
            result, _ = await asyncio.gather(
                D.claim(cfg_for(base, rpc), timeout=5.0, every=0.01), flip()
            )
            return result

    result = run(go())
    assert result.outcome == "confirmed" and result.exit_code == 0
    assert result.claim["state"] == "confirmed" and result.claim["amount"] == 7
    polls = [p for m, p in state.requests if m == "GET" and p.startswith("/api/brix/claim/")]
    assert polls, "the claim's status was never polled"
    assert outbox.alerts("testnet") == []
    assert state.logouts == 1


def test_claim_in_flight_polls_the_open_claim(fake_lfg, keys):
    base, state = fake_lfg
    state.brix_trustline_set = True
    state.brix["claimable"] = 3
    cid = "claim-open"
    state.brix_claims[cid] = {"claim_id": cid, "state": "confirmed", "amount": 3,
                              "tx_hash": "AB" * 32}
    state.brix["open_claim"] = {"claim_id": cid, "state": "submitted", "tx_hash": "AB" * 32}

    result, _ = run(do_claim(base, every=0.01))
    assert result.outcome == "confirmed" and result.exit_code == 0
    assert result.claim["claim_id"] == cid
    assert ("GET", f"/api/brix/claim/{cid}") in state.requests
    assert outbox.alerts("testnet") == []


def test_claim_failed_alerts_and_exits_nonzero(fake_lfg, keys):
    base, state = fake_lfg
    state.brix_trustline_set = True
    state.brix["claimable"] = 5
    state.brix_claim_outcome = "failed"

    result, _ = run(do_claim(base))
    assert result.outcome == "failed" and result.exit_code == 1
    alerts = outbox.alerts("testnet")
    assert len(alerts) == 1 and alerts[0]["payload"]["code"] == "claim_failed"
    assert alerts[0]["payload"]["claim"]["claim_id"] == result.claim["claim_id"]
    assert state.logouts == 1


def test_claim_that_never_settles_is_unknown(fake_lfg, keys):
    base, state = fake_lfg
    state.brix_trustline_set = True
    state.brix["claimable"] = 5
    state.brix_claim_outcome = "submitted"

    result, _ = run(do_claim(base, timeout=0.05, every=0.01))
    assert result.outcome == "unknown" and result.exit_code == 1
    alerts = outbox.alerts("testnet")
    assert len(alerts) == 1 and alerts[0]["payload"]["code"] == "claim_unknown"
    assert result.claim["state"] == "submitted"
    assert state.logouts == 1


@pytest.mark.parametrize("code,status", [("trustline_required", 409), ("claim_unavailable", 503),
                                         ("claim_unconfirmed", 502)])
def test_claim_other_refusals_alert_and_exit_nonzero(fake_lfg, keys, code, status):
    base, state = fake_lfg
    state.brix_trustline_set = code != "trustline_required"
    state.brix["claimable"] = 5
    # the fake has no switch for the other two codes: the client's call is stubbed to raise
    # exactly what LFG would answer

    async def go():
        async with FR.FakeRpc() as rpc:
            cfg = cfg_for(base, rpc)
            if code != "trustline_required":
                from lfg_fly.body.client import LfgError

                async def refuse(self):
                    raise LfgError(status, code, {"error": code, "code": code})

                orig = LfgClient.brix_claim
                LfgClient.brix_claim = refuse
                try:
                    return await D.claim(cfg)
                finally:
                    LfgClient.brix_claim = orig
            return await D.claim(cfg)

    result = run(go())
    assert result.outcome == "refused" and result.exit_code == 1
    alerts = outbox.alerts("testnet")
    assert len(alerts) == 1
    assert alerts[0]["payload"]["code"] == code and alerts[0]["payload"]["status"] == status
    assert state.logouts == 1


def test_claim_logs_out_even_when_the_call_blows_up(fake_lfg, keys, monkeypatch):
    base, state = fake_lfg

    async def boom(self):
        raise RuntimeError("wire cut")

    monkeypatch.setattr(LfgClient, "brix_claim", boom)
    with pytest.raises(RuntimeError, match="wire cut"):
        run(do_claim(base))
    assert state.logouts == 1 and not state.tokens


# ------------------------------------------------------------------ retrain


def _cat() -> Catalog:
    values = {s: [f"{s}{i}" for i in range(3)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1"]
    return Catalog(body="male", values=values, odds={}, api="x")


def _supply(cat: Catalog, network: str = "testnet") -> dict:
    counts = {}
    for slot, vals in cat.values.items():
        real = [v for v in vals if v != "None"]
        counts[slot] = {v: n for v, n in zip(real, (50, 30, 10), strict=False)}
    return {"network": network, "as_of": 1_700_000_000, "n_live": 100, "counts": counts}


class FakeBrain:
    """`FlyBrain`'s surface for retrain: `.catalog` and `.features(looks, conc)`. Features are
    one-hot (slot, value) columns plus the nine concentrations, so a ridge can recover the
    additive `nft_rarity` exactly and the test can see the live concentrations arrive."""

    def __init__(self, cat: Catalog, state: FL.FakeLfgState | None = None):
        self.catalog = cat
        self.index = {(s, v): i for i, (s, v) in
                      enumerate((s, v) for s in SLOTS for v in cat.values[s])}
        self.calls: list[tuple[list, np.ndarray | None]] = []
        self.state = state
        self.logouts_at_call: list[int] = []

    def features(self, looks, conc=None):
        looks = [tuple(lk) for lk in looks]
        self.calls.append((looks, None if conc is None else np.asarray(conc)))
        if self.state is not None:
            self.logouts_at_call.append(self.state.logouts)
        X = np.zeros((len(looks), len(self.index) + len(SLOTS)), np.float32)
        for j, look in enumerate(looks):
            for slot, value in zip(SLOTS, look, strict=True):
                X[j, self.index[(slot, value)]] = 1.0
            if conc is not None:
                X[j, len(self.index):] = conc[j]
        return X, {"active_frac": 0.1, "readout_active_frac": 0.2, "max_step_frac": 0.3,
                   "neurons_fired": 42, "ms": 1.5, "n_looks": len(looks)}


async def do_retrain(base: str, brain, **kw) -> D.RetrainResult:
    async with FR.FakeRpc() as rpc:
        return await D.retrain(cfg_for(base, rpc), brain, **kw)


def test_snapshot_hash_is_over_the_canonical_json():
    supply = {"network": "testnet", "n_live": 3, "counts": {"Head": {"Cap": 1, "Crown": 2}},
              "as_of": 5}
    shuffled = {"as_of": 5, "counts": {"Head": {"Crown": 2, "Cap": 1}}, "n_live": 3,
                "network": "testnet"}
    expected = hashlib.sha256(
        json.dumps(supply, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert D.snapshot_hash(supply) == expected == D.snapshot_hash(shuffled)
    assert D.snapshot_hash({**supply, "n_live": 4}) != expected
    snapshots = paths.snapshots_dir("testnet")
    assert D.snapshot_path("testnet", expected) == snapshots / f"{expected}.json"
    with pytest.raises(ValueError):
        D.snapshot_path("testnet", "../escape")


def test_retrain_writes_the_snapshot_and_a_head_named_by_it(fake_lfg, keys, _data_dir):
    base, state = fake_lfg
    cat = _cat()
    state.rarity_supply = _supply(cat)
    brain = FakeBrain(cat, state)

    result = run(do_retrain(base, brain, n=300))
    h = D.snapshot_hash(state.rarity_supply)
    assert result.snapshot_hash == h and not result.skipped
    assert result.n_looks == 300 and len(result.looks) == 300

    # the snapshot: FLY_DATA_DIR/<network>/snapshots/<sha256 of the canonical JSON>.json, stamped
    assert result.snapshot_path == paths.snapshots_dir("testnet") / f"{h}.json"
    snap = json.loads(result.snapshot_path.read_text())
    assert snap["supply"] == state.rarity_supply and snap["snapshot_hash"] == h
    assert snap["stamp"] == {"network": "testnet", "lfg_api_base": base, "wallet": keys.account}
    stamp = Stamp("testnet", base, keys.account)
    assert D.read_snapshot(result.snapshot_path, stamp) == state.rarity_supply
    with pytest.raises(StampMismatch):
        D.read_snapshot(result.snapshot_path, Stamp("testnet", base, "rSomeoneElse"))

    # the head: paths.rarity_head_path(network, hash) + .npz/.json, snapshot_hash matching
    base_path = paths.rarity_head_path("testnet", h)
    assert result.head_path == base_path.with_name(base_path.name + ".npz")
    assert result.head_path.exists() and base_path.with_name(base_path.name + ".json").exists()
    head = R.load_head(result.head_path)
    assert isinstance(head, R.RarityHead) and head.snapshot_hash == h
    assert head.w.shape == (len(brain.index) + len(SLOTS),)
    # the head's own files carry no stamp; the stamp lives in the meta and the pointer
    snapshots = paths.snapshots_dir("testnet")
    assert [p.name for p in snapshots.glob("rarity-head-*.json")] == [f"rarity-head-{h}.json"]
    assert (snapshots / f"rarity-meta-{h}.json").exists()

    # fitted on FlyBrain.features(looks, conc=concentrations(looks, supply, catalog)) against
    # nft_rarity: one-hot features recover the additive target
    looks = [tuple(lk) for lk in result.looks]
    (seen_looks, seen_conc), = brain.calls
    assert seen_looks == looks
    np.testing.assert_allclose(seen_conc, R.concentrations(looks, state.rarity_supply, cat))
    X, _ = brain.features(looks, seen_conc)
    target = np.array([R.nft_rarity(lk, state.rarity_supply) for lk in looks])
    pred = head.score(X) * head.target_sd + head.target_mu
    assert np.corrcoef(pred, target)[0, 1] > 0.99
    assert result.stats["train_r2"] > 0.95 and result.stats["heldout_r2"] > 0.9
    assert result.stats["neurons_fired"] == 42

    # the pointer the loop reads, stamped too
    latest = D.latest_head("testnet", stamp)
    assert latest["snapshot_hash"] == h
    assert latest["head"] == str(result.head_path)
    assert latest["snapshot"] == str(result.snapshot_path)
    assert latest["stats"] == result.stats and latest["version"] == "fly-v1"
    with pytest.raises(StampMismatch):
        D.latest_head("testnet", Stamp("testnet", "http://other", keys.account))
    assert D.latest_head("mainnet", stamp) is None

    # §4.1 step 1: signed in fresh, logged out in finally, and the session was closed
    # BEFORE the simulation started (the token lives as briefly as possible, §4.4)
    assert state.logouts == 1 and not state.tokens
    assert brain.logouts_at_call[0] == 1  # [1] is this test's own features() call above
    assert ("GET", "/api/rarity/supply") in state.requests
    assert_no_token_on_disk(state, _data_dir)


def test_retrain_is_reproducible_and_skips_an_existing_head_unless_forced(fake_lfg, keys):
    base, state = fake_lfg
    cat = _cat()
    state.rarity_supply = _supply(cat)

    first = run(do_retrain(base, FakeBrain(cat), n=50))
    brain = FakeBrain(cat)
    again = run(do_retrain(base, brain, n=50))
    assert again.skipped and brain.calls == []
    assert again.snapshot_hash == first.snapshot_hash and again.head_path == first.head_path
    forced = run(do_retrain(base, brain, n=50, force=True))
    assert not forced.skipped and len(brain.calls) == 1
    assert forced.looks == first.looks  # the seed derives from the snapshot hash
    other = run(do_retrain(base, FakeBrain(cat), n=50, force=True, seed=7))
    assert other.looks != first.looks
    assert state.logouts == 4


def test_retrain_refuses_when_not_enabled(fake_lfg, keys):
    base, state = fake_lfg
    cat = _cat()
    state.rarity_supply = _supply(cat)
    brain = FakeBrain(cat)

    async def go():
        async with FR.FakeRpc() as rpc:
            with pytest.raises(D.NotEnabled, match="FLY_ENABLED"):
                await D.retrain(cfg_for(base, rpc, enabled=False), brain, n=20)
            return rpc

    rpc = run(go())
    assert rpc.calls == [] and state.requests == [] and state.signin_starts == 0
    assert brain.calls == [] and not paths.snapshots_dir("testnet").exists()


def test_retrain_refuses_a_supply_from_another_network(fake_lfg, keys):
    base, state = fake_lfg
    cat = _cat()
    state.rarity_supply = _supply(cat, network="mainnet")
    brain = FakeBrain(cat)

    with pytest.raises(StampMismatch):
        run(do_retrain(base, brain, n=20))
    assert brain.calls == []
    assert not paths.snapshots_dir("testnet").exists() or not list(
        paths.snapshots_dir("testnet").iterdir()
    )
    assert state.logouts == 1


def test_sample_looks_are_distinct_and_from_the_catalog():
    cat = _cat()
    looks = D.sample_looks(cat, 200, seed=3)
    assert len(looks) == 200 and len(set(looks)) == 200
    for look in looks:
        assert all(v in cat.values[s] for s, v in zip(SLOTS, look, strict=True))
    assert D.sample_looks(cat, 200, seed=3) == looks


# ------------------------------------------------------------------ CLI


def test_register_adds_claim_and_retrain():
    parser = argparse.ArgumentParser(prog="fly")
    sub = parser.add_subparsers(dest="command")
    CD.register(sub)
    a = parser.parse_args(["claim"])
    assert a.command == "claim" and a.func is CD._cmd_claim
    assert a.timeout == 600.0 and a.every == 3.0
    b = parser.parse_args(["retrain"])
    assert b.command == "retrain" and b.func is CD._cmd_retrain
    assert b.n == D.DEFAULT_N_LOOKS == 2000 and b.device == "cuda" and b.lam == 1.0
    assert b.seed is None and b.force is False
    c = parser.parse_args(["retrain", "--n", "10", "--device", "cpu", "--force", "--seed", "4"])
    assert (c.n, c.device, c.force, c.seed) == (10, "cpu", True, 4)


def test_cmd_claim_loads_the_env_file_and_returns_the_exit_code(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text("FLY_NETWORK=testnet\nFLY_API_BASE=http://localhost:9999\n")
    monkeypatch.setattr(CD, "DOTENV", env_file)
    monkeypatch.delenv("FLY_NETWORK", raising=False)
    monkeypatch.delenv("FLY_API_BASE", raising=False)
    seen = {}

    async def fake_claim(cfg, *, timeout, every):
        seen.update(cfg=cfg, timeout=timeout, every=every)
        return D.ClaimResult(outcome="disabled", message="claims are disabled on this stack")

    monkeypatch.setattr(D, "claim", fake_claim)
    args = argparse.Namespace(timeout=12.0, every=0.5)
    assert CD._cmd_claim(args) == 0
    assert seen["cfg"].api_base == "http://localhost:9999" and seen["cfg"].network == "testnet"
    assert (seen["timeout"], seen["every"]) == (12.0, 0.5)
    assert "claims are disabled" in capsys.readouterr().out

    async def failed_claim(cfg, *, timeout, every):
        return D.ClaimResult(outcome="failed", message="claim failed")

    monkeypatch.setattr(D, "claim", failed_claim)
    assert CD._cmd_claim(args) == 1


def test_cmd_claim_and_retrain_refuse_with_exit_2_when_not_enabled(monkeypatch, capsys):
    """The kill switch at the CLI: one line, exit 2 (as `fly move` does), and for retrain
    the brain is never loaded (the check comes before the GPU)."""
    from lfg_fly.brain.checkpoint import FlyBrain

    monkeypatch.setenv("FLY_NETWORK", "testnet")
    monkeypatch.delenv("FLY_ENABLED", raising=False)
    assert CD._cmd_claim(argparse.Namespace(timeout=1.0, every=0.1)) == 2
    out = capsys.readouterr().out
    assert out.startswith("claim refused:") and "FLY_ENABLED" in out

    def never(*_a, **_k):
        raise AssertionError("the brain must not load when the fly is switched off")

    monkeypatch.setattr(FlyBrain, "load", never)
    args = argparse.Namespace(n=10, device="cpu", lam=1.0, seed=None, force=False)
    assert CD._cmd_retrain(args) == 2
    out = capsys.readouterr().out
    assert out.startswith("retrain refused:") and "FLY_ENABLED" in out

    # switched on, a DailyError from the job itself is still one line and exit 2
    monkeypatch.setenv("FLY_ENABLED", "1")

    async def refusing(cfg, *, timeout, every):
        raise D.CredentialsError("no wallet.json")

    monkeypatch.setattr(D, "claim", refusing)
    assert CD._cmd_claim(argparse.Namespace(timeout=1.0, every=0.1)) == 2
    assert "claim refused: no wallet.json" in capsys.readouterr().out


def test_claim_result_exit_codes():
    for outcome, code in (("confirmed", 0), ("nothing", 0), ("disabled", 0), ("failed", 1),
                          ("unknown", 1), ("refused", 1)):
        assert D.ClaimResult(outcome=outcome, message="").exit_code == code


def test_python_m_lfg_fly_runs_the_cli():
    root = paths.repo_root()
    source = (root / "lfg_fly" / "__main__.py").read_text()
    assert "from lfg_fly.cli import main" in source and "raise SystemExit(main())" in source
    proc = subprocess.run([sys.executable, "-m", "lfg_fly", "--help"], cwd=root,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage: fly")


# ------------------------------------------------------------------ ecosystem.config.js


def test_ecosystem_config_runs_the_three_jobs_niced_on_utc_cron():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    path = paths.repo_root() / "ecosystem.config.js"
    proc = subprocess.run(
        [node, "-e", "console.log(JSON.stringify(require(process.argv[1])))", str(path)],
        capture_output=True, text=True, timeout=60, env={**os.environ, "FLY_PYTHON": ""},
    )
    assert proc.returncode == 0, proc.stderr
    apps = {a["name"]: a for a in json.loads(proc.stdout)["apps"]}
    expected = {"fly-claim": ("10 4 * * *", "claim"), "fly-retrain": ("30 4 * * *", "retrain"),
                "fly-move": ("0 15 * * *", "move")}
    assert set(apps) == set(expected)
    for name, (cron, command) in expected.items():
        app = apps[name]
        assert app["cron_restart"] == cron
        assert app["script"] == "/usr/bin/nice"
        assert app["interpreter"] == "none"
        assert app["autorestart"] is False and app["autostart"] is False
        assert app["args"][:4] == ["-n", "10", "ionice", "-c3"]
        python = app["args"][4]
        assert python.endswith("/.venv/bin/python") and python.startswith(str(paths.repo_root()))
        # `-m lfg_fly` (lfg_fly/__main__.py): cli.py has no __main__ guard, so `-m lfg_fly.cli`
        # would import it and exit 0 without running anything
        assert app["args"][5:] == ["-m", "lfg_fly", command]
        assert app["cwd"] == str(paths.repo_root())
        assert app["env"]["PYTHONUNBUFFERED"] == "1"
