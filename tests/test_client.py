"""LfgClient against the fake LFG API (contract §body/client.py, docs/lfg-api.md, spec §7).

No pytest-asyncio in the venv: every test drives its coroutine with ``asyncio.run``
while the fake server runs in its own thread (see ``fake_lfg``).
"""

from __future__ import annotations

import asyncio
import time

import fake_lfg as FL
import pytest
from xrpl.wallet import Wallet

from lfg_fly.body.client import LfgClient, LfgError

fake_lfg = FL.fake_lfg  # the shared fixture, bound by name
FakeSigner, FakeLedger = FL.ProofSigner, FL.FakeLedger
SOURCE_TAG = 2606160021
PROOF_DESTINATION = "rrrrrrrrrrrrrrrrrNAMEtxvNvQ"


def run(coro):
    return asyncio.run(coro)


async def signed_in(base_url: str, signer: FakeSigner, ledger: FakeLedger | None = None):
    client = LfgClient(base_url)
    await client.__aenter__()
    await client.sign_in(signer, ledger or FakeLedger())
    return client


HASH = "AB" * 32


# ------------------------------------------------------------------ LfgError


def test_lfgerror_carries_status_code_body():
    err = LfgError(409, "tx_mismatch", {"error": "x", "code": "tx_mismatch"})
    assert (err.status, err.code) == (409, "tx_mismatch")
    assert err.body == {"error": "x", "code": "tx_mismatch"}
    assert "409" in str(err) and "tx_mismatch" in str(err)
    assert isinstance(err, RuntimeError)


def test_sign_id_parses_lfg_wc_links():
    sid = "wc-" + "0" * 32
    assert LfgClient.sign_id(f"lfg-wc://{sid}") == sid
    for bad in ("https://x/wc-00", "lfg-wc://", "lfg-wc://nope", "wc-" + "0" * 32, ""):
        with pytest.raises(ValueError):
            LfgClient.sign_id(bad)


# ------------------------------------------------------------------- sign-in


def test_sign_in_builds_the_proof_exactly(fake_lfg):
    base, state = fake_lfg
    wallet = Wallet.create()
    signer = FakeSigner(wallet)
    ledger = FakeLedger(1000)

    async def go():
        async with LfgClient(base) as client:
            token = await client.sign_in(signer, ledger)
            assert token and client.token == token
            assert client.wallet == wallet.address
            me = await client.me()
            assert me["wallet"] == wallet.address
            return token

    token = run(go())
    assert ledger.calls == 1
    (tx, tag), = signer.seen
    assert tag == SOURCE_TAG
    assert tx["TransactionType"] == "Payment"
    assert tx["Account"] == wallet.address
    assert tx["Destination"] == PROOF_DESTINATION
    assert tx["Amount"] == "1" and tx["Fee"] == "12" and tx["Sequence"] == 1
    assert tx["LastLedgerSequence"] == 1020 and tx["SourceTag"] == SOURCE_TAG
    # memos verbatim from the start response, in order
    assert tx["Memos"] == state.proofs[0]["Memos"]
    types = [bytes.fromhex(m["Memo"]["MemoType"]).decode() for m in tx["Memos"]]
    assert types == ["initiator", "platform", "action", "lfg/nonce"]
    assert set(tx) == {
        "TransactionType", "Account", "Destination", "Amount", "Fee", "Sequence",
        "LastLedgerSequence", "SourceTag", "Memos",
    }
    # the fake consumed the row and issued the token; the block logged out
    assert state.tokens == {} and state.logouts == 1
    assert token not in repr(LfgClient(base))


def test_sign_in_with_regular_key(fake_lfg):
    base, state = fake_lfg
    master, regular = Wallet.create(), Wallet.create()
    signer = FakeSigner(regular, account=master.address)

    async def go():
        async with LfgClient(base) as client:
            return await client.sign_in(signer, FakeLedger())

    with pytest.raises(LfgError) as ei:  # not the account's RegularKey yet
        run(go())
    assert (ei.value.status, ei.value.code) == (400, "bad_proof")
    assert state.proof_errors[-1] == "pubkey_account"

    state.regular_keys[master.address] = regular.address
    run(go())
    assert state.proofs[-1]["Account"] == master.address
    assert state.sessions[-1]["wallet"] == master.address
    assert state.sessions[-1]["key"] == "regular"


def test_sign_in_key_lookup_failure_is_503(fake_lfg):
    base, state = fake_lfg
    master, regular = Wallet.create(), Wallet.create()
    state.regular_keys[master.address] = regular.address
    state.key_lookup_ok = False

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(regular, account=master.address), FakeLedger())

    with pytest.raises(LfgError) as ei:
        run(go())
    assert (ei.value.status, ei.value.code) == (503, "regular_key_unverified")


def test_sign_in_refusals(fake_lfg):
    base, state = fake_lfg

    async def go(signer):
        async with LfgClient(base) as client:
            await client.sign_in(signer, FakeLedger())

    state.agent_enabled = False
    with pytest.raises(LfgError) as ei:
        run(go(FakeSigner(Wallet.create())))
    assert (ei.value.status, ei.value.code) == (503, "agent_disabled")
    state.agent_enabled = True

    with pytest.raises(LfgError) as ei:  # a signature that does not cover the tx
        run(go(FakeSigner(Wallet.create(), tamper=True)))
    assert (ei.value.status, ei.value.code) == (400, "bad_proof")
    assert state.proof_errors[-1] == "signature"

    state.validated_ledger = 50  # LLS 70 fine; make the window tiny
    state.proof_lls_window = 10
    with pytest.raises(LfgError) as ei:
        run(go(FakeSigner(Wallet.create())))
    assert state.proof_errors[-1] == "last_ledger"
    assert state.tokens == {}


def test_sign_in_replay_is_409(fake_lfg):
    base, state = fake_lfg
    wallet = Wallet.create()

    async def go():
        client = LfgClient(base)
        async with client:
            await client.sign_in(FakeSigner(wallet), FakeLedger())
            sid = state.proof_rows[-1]["id"]
            tx = state.proofs[-1]
            await client._call("POST", "/api/web/signin/proof", auth=False,
                               json={"sign_id": sid, "tx_json": tx})

    with pytest.raises(LfgError) as ei:
        run(go())
    assert (ei.value.status, ei.value.code) == (409, "proof_replayed")


# ------------------------------------------------------ token, logout, errors


def test_aexit_logs_out_even_when_the_body_raises(fake_lfg):
    base, state = fake_lfg
    signer = FakeSigner(Wallet.create())
    holder: dict = {}

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(signer, FakeLedger())
            holder["client"] = client
            assert client.token in state.tokens
            raise KeyError("boom")

    with pytest.raises(KeyError):
        run(go())
    assert state.logouts == 1 and state.tokens == {}
    assert holder["client"].token is None


def test_aexit_swallows_logout_failure_and_drops_token(fake_lfg):
    base, state = fake_lfg
    holder: dict = {}

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            holder["client"] = client
            state.logout_error = (500, {"error": "db down"})

    run(go())  # no raise
    assert holder["client"].token is None
    assert isinstance(holder["client"].logout_error, LfgError)


def test_aexit_without_sign_in_does_not_call_logout(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            await client.health()

    run(go())
    assert state.logouts == 0


def test_explicit_logout_then_no_second_logout(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            await client.logout()
            assert client.token is None

    run(go())
    assert state.logouts == 1


def test_non_2xx_raises_lfgerror(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            await client.me()  # no token

    with pytest.raises(LfgError) as ei:
        run(go())
    assert ei.value.status == 401
    assert ei.value.code == "unauthorized"
    assert ei.value.body == {"error": "unauthorized"}


def test_revoked_key_is_401_key_revoked(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            state.key_revoked = True
            await client.nfts()

    with pytest.raises(LfgError) as ei:
        run(go())
    assert (ei.value.status, ei.value.code) == (401, "key_revoked")


def test_external_session_is_not_closed(fake_lfg):
    import aiohttp

    base, _ = fake_lfg

    async def go():
        async with aiohttp.ClientSession() as session:
            async with LfgClient(base, session=session) as client:
                assert (await client.health())["ok"] is True
            assert not session.closed
            async with LfgClient(base, session=session) as client:
                assert (await client.health())["ok"] is True

    run(go())


# --------------------------------------------------------------------- reads


def test_reads(fake_lfg):
    base, state = fake_lfg
    state.rarity_supply["counts"] = {"Head": {"Crown": 3}}

    async def go():
        async with LfgClient(base) as client:
            health = await client.health()
            supply = await client.rarity_supply()  # no auth
            rarity = await client.rarity("male")
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            nfts = await client.nfts()
            eco = await client.economy()
            return health, supply, rarity, nfts, eco

    health, supply, rarity, nfts, eco = run(go())
    assert health["ok"] is True
    assert supply["counts"] == {"Head": {"Crown": 3}}
    assert rarity["body"] == "male"
    assert len(nfts["nfts"]) == len(state.characters) == 1
    nft = nfts["nfts"][0]
    assert [a["trait_type"] for a in nft["attributes"]] == list(state.SLOTS)
    assert nft["mutable"] is True and nft["blank"] is False and nft["gender"] == "male"
    assert nfts["swap_fee"] == state.swap_fee
    assert eco["closet"]["token"]["status"] == "active"
    assert eco["z_order"] == state.z_order
    assert eco["characters"][0]["nft_id"] == state.characters[0]["nft_id"]


def test_economy_disabled_is_403(fake_lfg):
    base, state = fake_lfg
    state.economy_enabled = False

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            await client.economy()

    with pytest.raises(LfgError) as ei:
        run(go())
    assert (ei.value.status, ei.value.code) == (403, "economy_disabled")


def test_layer_resolves_200_404_and_memo(fake_lfg):
    base, state = fake_lfg
    state.layers = {("male", "Head", "Crown")}

    async def go():
        async with LfgClient(base) as client:
            a = await client.layer_resolves("male", "Head", "Crown")
            b = await client.layer_resolves("male", "Head", "Pirate Hat")
            c = await client.layer_resolves("male", "Head", "Crown")
            d = await client.layer_resolves("male", "Head", "Pirate Hat")
            return a, b, c, d

    assert run(go()) == (True, False, True, False)
    layer_calls = [p for m, p in state.requests if p.startswith("/api/layer")]
    assert len(layer_calls) == 2  # memoized per client

    async def bad():
        async with LfgClient(base) as client:
            await client.layer_resolves("", "Head", "Crown")

    with pytest.raises(LfgError) as ei:
        run(bad())
    assert ei.value.status == 400


# --------------------------------------------------------------------- equip


def test_equip_done_applies_changes(fake_lfg):
    base, state = fake_lfg
    hero = state.characters[0]["nft_id"]
    state.add_closet("Head", "Pirate Hat")
    state.add_closet("Eyes", "None")
    before_head = state.look(hero)["Head"]

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            started = await client.equip(
                hero, [{"slot": "Head", "value": "Pirate Hat"}, {"slot": "Eyes", "value": "None"}]
            )
            assert started["state"] == "running" and started["kind"] == "equip"
            first = await client.equip_status(started["id"])
            done = await client.wait(client.equip_status, started["id"],
                                     {"done", "failed"}, timeout=5, every=0.01)
            return first, done

    first, done = run(go())
    assert first["state"] == "running"
    assert done["state"] == "done" and done["resolution"] == "committed"
    assert {"slot": "Head", "value": before_head} in done["displaced"]
    look = state.look(hero)
    assert look["Head"] == "Pirate Hat" and look["Eyes"] == "None"
    assert state.closet_count("Head", "Pirate Hat") == 0
    assert state.closet_count("Head", before_head) == 1


@pytest.mark.parametrize(
    "outcome,resolution",
    [("failed_reverted", "reverted"), ("failed_uncertain", "uncertain")],
)
def test_equip_failures(fake_lfg, outcome, resolution):
    base, state = fake_lfg
    hero = state.characters[0]["nft_id"]
    state.add_closet("Head", "Pirate Hat")
    state.equip_outcome = outcome

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            started = await client.equip(hero, [{"slot": "Head", "value": "Pirate Hat"}])
            return await client.wait(client.equip_status, started["id"],
                                     {"done", "failed"}, timeout=5, every=0.01)

    status = run(go())
    assert status["state"] == "failed" and status["resolution"] == resolution
    assert state.look(hero)["Head"] != "Pirate Hat"


def test_equip_hang_times_out_in_wait(fake_lfg):
    base, state = fake_lfg
    hero = state.characters[0]["nft_id"]
    state.add_closet("Head", "Pirate Hat")
    state.equip_outcome = "hang"

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            started = await client.equip(hero, [{"slot": "Head", "value": "Pirate Hat"}])
            clock = {"t": 0.0}
            client._clock = lambda: clock["t"]

            async def sleep(s):
                clock["t"] += s

            client._sleep = sleep
            polls = len([p for _, p in state.requests if p.startswith("/api/equip/")])
            with pytest.raises(TimeoutError):
                await client.wait(client.equip_status, started["id"], {"done", "failed"},
                                  timeout=10, every=3)
            return len([p for _, p in state.requests if p.startswith("/api/equip/")]) - polls

    polls = run(go())
    assert polls == 5  # t = 0, 3, 6, 9, 10


def test_equip_refusals(fake_lfg):
    base, state = fake_lfg
    hero = state.characters[0]["nft_id"]
    state.layers = set()  # nothing resolves

    async def go(changes):
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            await client.equip(hero, changes)

    with pytest.raises(LfgError) as ei:  # not in the closet
        run(go([{"slot": "Head", "value": "Pirate Hat"}]))
    assert ei.value.status == 400
    state.add_closet("Head", "Pirate Hat")
    with pytest.raises(LfgError) as ei:  # does not resolve on a male body
        run(go([{"slot": "Head", "value": "Pirate Hat"}]))
    assert ei.value.status == 400 and "does not fit" in ei.value.body["error"]
    state.layers = None
    state.listed.add(("Head", "Pirate Hat"))
    with pytest.raises(LfgError) as ei:  # listed on the closet market
        run(go([{"slot": "Head", "value": "Pirate Hat"}]))
    assert "listed" in ei.value.body["error"]
    state.listed.clear()
    state.characters[0]["blank"] = True
    with pytest.raises(LfgError) as ei:
        run(go([{"slot": "Head", "value": "Pirate Hat"}]))
    assert (ei.value.status, ei.value.code) == (409, "blank_character")
    state.characters[0]["blank"] = False
    state.characters[0]["mutable"] = False
    with pytest.raises(LfgError) as ei:
        run(go([{"slot": "Head", "value": "Pirate Hat"}]))
    assert ei.value.status == 400 and "not mutable" in ei.value.body["error"]
    state.characters[0]["mutable"] = True
    with pytest.raises(LfgError) as ei:  # Body slot
        run(go([{"slot": "Body", "value": "ape"}]))
    assert ei.value.status == 400
    with pytest.raises(LfgError) as ei:  # empty
        run(go([]))
    assert ei.value.status == 400


# ------------------------------------------------------------------- harvest


def test_harvest_blanks_the_donor_and_credits_the_closet(fake_lfg):
    base, state = fake_lfg
    donor = state.add_character("D" * 64, body="male", traits={"Head": "Crown", "Eyes": "Laser"})

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            started = await client.harvest(donor)
            assert started["kind"] == "harvest" and started["state"] == "running"
            return await client.wait(client.harvest_status, started["id"],
                                     {"done", "failed"}, timeout=5, every=0.01)

    status = run(go())
    assert status["state"] == "done"
    assert ["Head", "Crown"] in status["moved_assets"]
    assert ["Back", "None"] in status["moved_assets"]  # empty donor slots credit None units
    assert ["Body", "male"] in status["moved_assets"]
    ch = state.character(donor)
    assert ch["blank"] is True and all(a["value"] == "None" for a in ch["attributes"])
    assert state.closet_count("Head", "Crown") == 1
    assert state.closet_count("Back", "None") == 1


def test_harvest_needs_an_active_closet(fake_lfg):
    base, state = fake_lfg
    state.closet_token = {"status": "none", "nft_id": None}

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            await client.harvest(state.characters[0]["nft_id"])

    with pytest.raises(LfgError) as ei:
        run(go())
    assert ei.value.status == 400 and "Closet" in ei.value.body["error"]


# ----------------------------------------------------------- sign requests


def test_sign_request_and_result(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            row = state.new_sign_request(
                {"TransactionType": "AccountSet", "Account": client.wallet}, wallet=client.wallet
            )
            got = await client.sign_request(row["id"])
            assert got["state"] == "pending" and got["txjson"]["Account"] == client.wallet
            assert set(got) == {"id", "state", "txjson", "expires_at", "txid"}
            res = await client.sign_result(row["id"], tx_hash=HASH.lower())
            again = await client.sign_result(row["id"], tx_hash=HASH)  # same hash: 200
            with pytest.raises(LfgError) as ei:
                await client.sign_result(row["id"], tx_hash="CD" * 32)
            assert (ei.value.status, ei.value.code) == (409, "already_resolved")
            with pytest.raises(LfgError) as ei:
                await client.sign_request("wc-" + "f" * 32)
            assert ei.value.status == 404
            return res, again

    res, again = run(go())
    assert res == {"state": "signed", "txid": HASH}
    assert again == {"state": "signed", "txid": HASH}


def test_sign_result_rejected_and_error_and_exactly_one(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            r1 = state.new_sign_request({"TransactionType": "AccountSet"}, wallet=client.wallet)
            r2 = state.new_sign_request({"TransactionType": "AccountSet"}, wallet=client.wallet)
            a = await client.sign_result(r1["id"], rejected=True)
            b = await client.sign_result(r2["id"], error="wallet said no")
            with pytest.raises(ValueError):
                await client.sign_result(r2["id"])
            with pytest.raises(ValueError):
                await client.sign_result(r2["id"], tx_hash=HASH, rejected=True)
            return a, b, r1["id"], r2["id"]

    a, b, r1, r2 = run(go())
    assert a == {"state": "rejected"} and b == {"state": "failed"}
    assert state.sign_requests[r1]["state"] == "rejected"
    assert state.sign_requests[r2]["state"] == "failed"


def test_sign_result_retries_202_and_503_with_backoff(fake_lfg):
    base, state = fake_lfg
    state.sign_result_stalls = [202, 503, 202]

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            slept: list[float] = []

            async def sleep(s):
                slept.append(s)

            client._sleep = sleep
            row = state.new_sign_request({"TransactionType": "AccountSet"}, wallet=client.wallet)
            res = await client.sign_result(row["id"], tx_hash=HASH)
            return res, slept

    res, slept = run(go())
    assert res["state"] == "signed"
    assert len(slept) == 3 and slept == sorted(slept) and slept[0] > 0
    posts = [p for m, p in state.requests if m == "POST" and p.endswith("/result")]
    assert len(posts) == 4


def test_sign_result_gives_up_at_expires_at(fake_lfg):
    base, state = fake_lfg
    state.sign_result_always = 202

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            clock = {"t": time.time()}
            client._clock = lambda: clock["t"]

            async def sleep(s):
                clock["t"] += s

            client._sleep = sleep
            row = state.new_sign_request({"TransactionType": "AccountSet"}, wallet=client.wallet)
            with pytest.raises(LfgError) as ei:
                await client.sign_result(row["id"], tx_hash=HASH)
            return ei.value, clock["t"], row["expires_at"]

    err, t, expires_at = run(go())
    assert (err.status, err.code) == (202, "tx_not_found")
    assert t >= expires_at
    assert t - expires_at < 10  # the last sleep was clipped to the deadline


# ---------------------------------------------------------------------- mint


def test_mint_flow_advances_on_sign_results(fake_lfg):
    base, state = fake_lfg
    state.mint_queue.append({"nft_id": "M" * 64, "body_type": "male", "traits": {"Head": "Crown"}})

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            s = await client.mint()
            assert s["state"] == "awaiting_payment" and s["pay_with"] == "XRP"
            assert s["pay_amount"] == "10"
            active = await client.mint_active()
            assert active["id"] == s["id"]
            with pytest.raises(LfgError) as ei:
                await client.mint()
            assert ei.value.status == 409
            pay_id = LfgClient.sign_id(s["payment_link"])
            req = await client.sign_request(pay_id)
            tx = req["txjson"]
            assert tx["TransactionType"] == "Payment" and tx["Account"] == client.wallet
            assert tx["Destination"] == state.signing_account and tx["Amount"] == "10000000"
            assert tx["SourceTag"] == SOURCE_TAG and tx["Memos"]
            await client.sign_result(pay_id, tx_hash="11" * 32)
            s2 = await client.mint_status(s["id"])
            assert s2["state"] == "offer_ready" and s2["nft_id"] == "M" * 64
            acc_id = LfgClient.sign_id(s2["accept_deeplink"])
            acc = await client.sign_request(acc_id)
            assert acc["txjson"]["TransactionType"] == "NFTokenAcceptOffer"
            assert acc["txjson"]["NFTokenSellOffer"]
            await client.sign_result(acc_id, tx_hash="22" * 32)
            s3 = await client.mint_status(s["id"])
            assert s3["state"] == "done" and s3["accept_signed"] is True
            with pytest.raises(LfgError) as ei:
                await client.mint_active()
            assert ei.value.status == 404
            return s3

    run(go())
    assert state.character("M" * 64)["mutable"] is True
    assert state.look("M" * 64)["Head"] == "Crown"


def test_mint_refusal_and_ref(fake_lfg):
    base, state = fake_lfg
    state.mint_refusal = (409, {"error": "collection full", "code": "collection_full"})

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            with pytest.raises(LfgError) as ei:
                await client.mint(ref="abc")
            assert (ei.value.status, ei.value.code) == (409, "collection_full")
            state.mint_refusal = None
            s = await client.mint(ref="abc")
            return s

    s = run(go())
    assert state.mint_sessions[s["id"]]["ref"] == "abc"


def test_bulk_mint_flow(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            job = await client.bulk_mint(12)
            assert job["quantity"] == 10 and job["requested_qty"] == 12
            assert job["pay_amount"] == "100" and job["state"] == "awaiting_payment"
            assert [u["state"] for u in job["units"]] == ["pending"] * 10
            pay_id = LfgClient.sign_id(job["payment_link"])
            tx = (await client.sign_request(pay_id))["txjson"]
            assert tx["Amount"] == "100000000" and tx["Destination"] == state.signing_account
            await client.sign_result(pay_id, tx_hash="33" * 32)
            j2 = await client.bulk_status(job["id"])
            assert j2["state"] == "done" and j2["offered"] == 10
            assert all(u["state"] == "offered" and u["nft_id"] for u in j2["units"])
            link = await client.bulk_unit_accept(job["id"], 3)
            acc_id = LfgClient.sign_id(link["link"])
            acc = await client.sign_request(acc_id)
            assert acc["txjson"]["TransactionType"] == "NFTokenAcceptOffer"
            assert acc["txjson"]["NFTokenSellOffer"] == j2["units"][3]["offer_id"]
            await client.sign_result(acc_id, tx_hash="44" * 32)
            j3 = await client.bulk_status(job["id"])
            return j3

    j3 = run(go())
    assert j3["units"][3]["accepted"] is True
    assert state.character(j3["units"][3]["nft_id"]) is not None


# -------------------------------------------------------------------- closet


def test_closet_none_to_active(fake_lfg):
    base, state = fake_lfg
    state.closet_token = {"status": "none", "nft_id": None}

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            c1 = await client.closet()
            assert c1["status"] == "pending_accept" and c1["accept"].startswith("lfg-wc://")
            c2 = await client.closet()  # not yet signed: same answer
            assert c2["status"] == "pending_accept" and c2["accept"] == c1["accept"]
            sid = LfgClient.sign_id(c1["accept"])
            tx = (await client.sign_request(sid))["txjson"]
            assert tx["TransactionType"] == "NFTokenAcceptOffer"
            await client.sign_result(sid, tx_hash="55" * 32)
            c3 = await client.closet()
            assert c3["status"] == "active" and c3["accept"] is None
            eco = await client.economy()
            return c3, eco

    c3, eco = run(go())
    assert eco["closet"]["token"] == {"status": "active", "nft_id": c3["nft_id"]}


def test_closet_transient_error(fake_lfg):
    base, state = fake_lfg
    state.closet_error = (503, {"error": "closet_mint_transient", "retryable": True})

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            await client.closet()

    with pytest.raises(LfgError) as ei:
        run(go())
    assert ei.value.status == 503 and ei.value.code == "closet_mint_transient"


# ---------------------------------------------------------------------- BRIX


def test_brix_trustline_and_claim(fake_lfg):
    base, state = fake_lfg
    state.brix["claimable"] = 12.5

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            with pytest.raises(LfgError) as ei:
                await client.brix_claim()
            assert (ei.value.status, ei.value.code) == (409, "trustline_required")
            t = await client.brix_trustline()
            assert t["state"] == "pending" and t["xumm_url"] == f"lfg-wc://{t['uuid']}"
            tx = (await client.sign_request(t["uuid"]))["txjson"]
            assert tx["TransactionType"] == "TrustSet" and tx["Flags"] == 131072
            assert tx["LimitAmount"]["currency"] == state.brix_currency
            assert tx["LimitAmount"]["issuer"] == state.brix_issuer
            assert (await client.brix_trustline_status(t["uuid"]))["state"] == "pending"
            await client.sign_result(t["uuid"], tx_hash="66" * 32)
            st = await client.brix_trustline_status(t["uuid"])
            assert st == {"state": "signed", "tx_hash": "66" * 32}
            assert (await client.brix_trustline()) == {"state": "already_set"}
            b = await client.brix()
            assert b["wallet"] == client.wallet and b["claimable"] == 12.5
            claim = await client.brix_claim()
            assert claim["state"] == "confirmed" and claim["amount"] == 12.5
            status = await client.brix_claim_status(claim["claim_id"])
            assert status["claim_id"] == claim["claim_id"]
            with pytest.raises(LfgError) as ei:
                await client.brix_claim()
            assert (ei.value.status, ei.value.code) == (400, "nothing_to_claim")
            state.brix_claims_enabled = False
            with pytest.raises(LfgError) as ei:
                await client.brix_claim()
            assert (ei.value.status, ei.value.code) == (503, "claims_disabled")

    run(go())
    assert state.brix["claimed_total"] == 12.5 and state.brix["claimable"] == 0


# ------------------------------------------------------------ pending offers


def test_pending_offers_accept(fake_lfg):
    base, state = fake_lfg
    state.pending_offers.append(
        {
            "offer_index": "OF" * 32, "nft_id": "N" * 64, "kind": "character", "nft_number": 7,
            "image": None, "amount": "0", "price_label": "free",
            "_character": {"nft_id": "N" * 64, "body": "male", "traits": {"Eyes": "Laser"}},
        }
    )

    async def go():
        async with LfgClient(base) as client:
            await client.sign_in(FakeSigner(Wallet.create()), FakeLedger())
            offers = (await client.pending_offers())["offers"]
            assert len(offers) == 1 and "_character" not in offers[0]
            link = await client.accept_offer(offers[0]["offer_index"])
            sid = LfgClient.sign_id(link["link"])
            tx = (await client.sign_request(sid))["txjson"]
            assert tx["NFTokenSellOffer"] == "OF" * 32
            await client.sign_result(sid, tx_hash="77" * 32)
            assert (await client.pending_offers())["offers"] == []
            with pytest.raises(LfgError) as ei:
                await client.accept_offer("OF" * 32)
            assert (ei.value.status, ei.value.code) == (410, "offer_gone")

    run(go())
    assert state.look("N" * 64)["Eyes"] == "Laser"


# -------------------------------------------------------------------- fake


def test_fake_state_slots_match_the_senses():
    from lfg_fly.brain.senses import SLOTS

    assert FL.FakeLfgState.SLOTS == FL.SLOTS == SLOTS
    assert FL.SOURCE_TAG == SOURCE_TAG and FL.PROOF_DESTINATION == PROOF_DESTINATION


def test_fake_requires_bearer_on_wallet_endpoints(fake_lfg):
    base, state = fake_lfg

    async def go():
        async with LfgClient(base) as client:
            for call in (client.nfts, client.economy, client.closet, client.brix,
                         client.pending_offers, client.mint_active):
                with pytest.raises(LfgError) as ei:
                    await call()
                assert ei.value.status == 401
            client.token = "tok-forged"
            with pytest.raises(LfgError) as ei:
                await client.nfts()
            assert ei.value.status == 401
            client.token = None

    run(go())
