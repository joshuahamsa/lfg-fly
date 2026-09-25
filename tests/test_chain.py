"""lfg_fly.body.chain against a fake JSON-RPC endpoint (spec §4.4, §7 "chain identity")."""

from __future__ import annotations

import asyncio

import pytest
from fake_rpc import (
    MAINNET_HASH,
    OTHER_HASH,
    TESTNET_HASH,
    FakeRpc,
    RpcError,
    tx_hash,
    unreachable_url,
)
from xrpl.core import binarycodec
from xrpl.models.requests import ServerInfo
from xrpl.models.transactions import Payment
from xrpl.transaction import sign
from xrpl.wallet import Wallet

from lfg_fly.body import chain
from lfg_fly.body.chain import (
    AutofillError,
    ChainIdentity,
    ChainIdentityError,
    Ledger,
    LedgerUnavailable,
    SimulateFailed,
    SubmitFailed,
)

PROOF_DEST = "rrrrrrrrrrrrrrrrrNAMEtxvNvQ"


def run(coro):
    return asyncio.run(coro)


def ledger_for(*urls: str, network: str = "testnet", **kw) -> Ledger:
    kw.setdefault("timeout", 5.0)
    kw.setdefault("poll_interval", 0.0)
    return Ledger(list(urls), network, **kw)


def signed_blob(wallet: Wallet, *, sequence: int = 5, lls: int | None = 1_020) -> str:
    tx = Payment(
        account=wallet.address,
        destination=PROOF_DEST,
        amount="1",
        fee="12",
        sequence=sequence,
        last_ledger_sequence=lls,
    )
    return binarycodec.encode(sign(tx, wallet).to_xrpl())


# -- identity_check (§4.4) ------------------------------------------------------------------


def test_identity_skips_pruned_and_unreachable_and_takes_the_hash():
    async def main():
        async with FakeRpc().prune() as pruned, FakeRpc(clio=True) as clio:
            led = ledger_for(pruned.url, unreachable_url(), clio.url)
            try:
                ident = await led.identity_check(TESTNET_HASH)
            finally:
                await led.close()
            expected = ChainIdentity(network_id=1, ledger_32570=TESTNET_HASH, endpoint=clio.url)
            assert ident == expected
            assert isinstance(ident, ChainIdentity)
            assert pruned.calls_for("ledger")[0]["ledger_index"] == 32570

    run(main())


def test_identity_any_mismatch_refuses_even_with_a_matching_endpoint():
    async def main():
        async with FakeRpc(clio=True) as good, FakeRpc(clio=True).wrong_hash() as bad:
            for urls in ([good.url, bad.url], [bad.url, good.url]):
                led = ledger_for(*urls)
                try:
                    with pytest.raises(ChainIdentityError, match=OTHER_HASH[:16]):
                        await led.identity_check(TESTNET_HASH)
                finally:
                    await led.close()

    run(main())


def test_identity_refuses_when_nobody_returns_the_hash():
    async def main():
        async with FakeRpc().prune() as pruned:
            led = ledger_for(pruned.url, unreachable_url())
            try:
                with pytest.raises(ChainIdentityError, match="32570"):
                    await led.identity_check(TESTNET_HASH)
            finally:
                await led.close()

    run(main())


def test_identity_refuses_wrong_network_id():
    async def main():
        # a testnet endpoint (network_id 1) serving a fly configured for mainnet
        async with FakeRpc(network_id=1, hash_32570=MAINNET_HASH) as rpc:
            led = ledger_for(rpc.url, network="mainnet")
            try:
                with pytest.raises(ChainIdentityError, match="network_id"):
                    await led.identity_check(MAINNET_HASH)
            finally:
                await led.close()
        # and a mainnet endpoint is accepted for mainnet
        async with FakeRpc(network_id=0, hash_32570=MAINNET_HASH) as rpc:
            led = ledger_for(rpc.url, network="mainnet")
            try:
                ident = await led.identity_check(MAINNET_HASH)
            finally:
                await led.close()
            assert ident.network_id == 0 and ident.ledger_32570 == MAINNET_HASH

    run(main())


def test_identity_refuses_without_a_pinned_hash():
    async def main():
        async with FakeRpc(clio=True) as rpc:
            led = ledger_for(rpc.url)
            try:
                with pytest.raises(ChainIdentityError):
                    await led.identity_check(None)
            finally:
                await led.close()
            assert rpc.calls == []  # fails closed before asking anyone

    run(main())


def test_identity_hash_comparison_is_case_insensitive():
    async def main():
        async with FakeRpc(clio=True) as rpc:
            led = ledger_for(rpc.url)
            try:
                ident = await led.identity_check(TESTNET_HASH.lower())
            finally:
                await led.close()
            assert ident.ledger_32570 == TESTNET_HASH

    run(main())


def test_unknown_network_is_refused_at_construction():
    with pytest.raises(ValueError):
        Ledger(["http://127.0.0.1:1/"], "devnet")


# -- request ---------------------------------------------------------------------------------


def test_request_falls_through_to_the_first_endpoint_that_answers():
    async def main():
        async with FakeRpc(validated=777) as rpc:
            led = ledger_for(unreachable_url(), rpc.url)
            try:
                r = await led.request({"method": "server_info"})
                assert r["info"]["network_id"] == 1
                r2 = await led.request(ServerInfo())
                assert r2["info"]["validated_ledger"]["seq"] == 777
            finally:
                await led.close()

    run(main())


def test_request_raises_when_no_endpoint_answers():
    async def main():
        led = ledger_for(unreachable_url(), unreachable_url())
        try:
            with pytest.raises(LedgerUnavailable):
                await led.request({"method": "server_info"})
        finally:
            await led.close()

    run(main())


def test_request_skips_http_failures_garbage_and_unknown_commands():
    async def main():
        async with FakeRpc() as bad_status, FakeRpc() as garbage, FakeRpc() as old, FakeRpc() as ok:
            bad_status.http_status = 503
            garbage.garbage = True
            old.answer_error("server_info", "unknownCmd", "Unknown method.")
            led = ledger_for(bad_status.url, garbage.url, old.url, ok.url)
            try:
                r = await led.request({"method": "server_info"})
            finally:
                await led.close()
            assert r["info"]["network_id"] == 1
            assert len(ok.calls) == 1

    run(main())


def test_request_returns_semantic_errors_from_the_first_endpoint():
    async def main():
        async with FakeRpc() as a, FakeRpc() as b:
            led = ledger_for(a.url, b.url)
            try:
                r = await led.request({"method": "account_info", "account": "rNobody"})
            finally:
                await led.close()
            assert r["error"] == "actNotFound"
            assert b.calls == []  # a real answer; no need to ask the next endpoint

    run(main())


def test_request_skips_a_hanging_endpoint_after_the_timeout():
    async def main():
        async with FakeRpc() as slow, FakeRpc() as ok:
            slow.hang = 5.0
            led = ledger_for(slow.url, ok.url, timeout=0.3)
            try:
                r = await led.request({"method": "server_info"})
            finally:
                await led.close()
            assert r["info"]["network_id"] == 1

    run(main())


def test_ledger_is_an_async_context_manager():
    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            assert (await led.validated_ledger_index()) == 1_000

    run(main())


# -- reads -----------------------------------------------------------------------------------


def test_account_info_and_not_found():
    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.add_account("rAlice", sequence=42)
            info = await led.account_info("rAlice")
            assert info["account_data"]["Sequence"] == 42
            assert await led.account_info("rNobody") is None

    run(main())


def test_account_nfts_paginates_and_empty_on_not_found():
    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.page_size = 3
            rpc.add_account("rAlice")
            for i in range(7):
                rpc.add_nft("rAlice", f"{i:064X}", f"ipfs://{i}")
            nfts = await led.account_nfts("rAlice")
            assert [n["NFTokenID"] for n in nfts] == [f"{i:064X}" for i in range(7)]
            assert len(rpc.calls_for("account_nfts")) == 3
            assert await led.account_nfts("rNobody") == []

    run(main())


def test_nft_info_prefers_clio_and_is_none_when_unsupported_or_missing():
    nft = "AB" * 32

    async def main():
        async with FakeRpc() as rippled, FakeRpc(clio=True) as clio:
            clio.add_nft("rAlice", nft, "https://x/meta.json")
            async with ledger_for(rippled.url, clio.url) as led:
                info = await led.nft_info(nft)
                assert info is not None and info["owner"] == "rAlice"
                assert len(rippled.calls_for("nft_info")) == 1  # asked, said unknownCmd, skipped
                assert await led.nft_info("CD" * 32) is None  # objectNotFound
            async with ledger_for(rippled.url) as led:
                assert await led.nft_info(nft) is None

    run(main())


def test_nft_uri_from_nft_info_then_account_nfts_scan():
    nft = "AB" * 32

    async def main():
        async with FakeRpc() as rippled, FakeRpc(clio=True) as clio:
            rippled.add_account("rAlice")
            rippled.add_nft("rAlice", "CD" * 32, "ipfs://other")
            rippled.add_nft("rAlice", nft, "ipfs://from-rippled")
            clio.add_nft("rAlice", nft, "ipfs://from-clio")
            async with ledger_for(rippled.url, clio.url) as led:
                assert await led.nft_uri(nft, "rAlice") == "ipfs://from-clio"
            async with ledger_for(rippled.url) as led:
                assert await led.nft_uri(nft, "rAlice") == "ipfs://from-rippled"
                assert await led.nft_uri("EF" * 32, "rAlice") is None
                assert await led.nft_uri(nft, "rNobody") is None

    run(main())


def test_nft_uri_is_none_for_a_burned_or_uri_less_token():
    nft = "AB" * 32

    async def main():
        async with FakeRpc(clio=True) as clio, ledger_for(clio.url) as led:
            clio.add_account("rAlice")
            clio.add_nft("rAlice", nft, None)
            assert await led.nft_uri(nft, "rAlice") is None
            clio.add_nft("rAlice", "CD" * 32, "ipfs://gone")
            clio.nft_infos["CD" * 32]["is_burned"] = True
            clio.nfts["rAlice"] = [n for n in clio.nfts["rAlice"] if n["NFTokenID"] != "CD" * 32]
            assert await led.nft_uri("CD" * 32, "rAlice") is None

    run(main())


def test_sell_offers_list_and_empty():
    nft = "AB" * 32

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.offers[nft] = [
                {"amount": "0", "flags": 1, "nft_offer_index": "01" * 32, "owner": "rBot",
                 "destination": "rFly"},
            ]
            offers = await led.sell_offers(nft)
            assert offers[0]["owner"] == "rBot" and offers[0]["destination"] == "rFly"
            assert await led.sell_offers("CD" * 32) == []

    run(main())


def test_tx_and_validated_ledger_index():
    async def main():
        async with FakeRpc(validated=4_321) as rpc, ledger_for(rpc.url) as led:
            assert await led.tx("00" * 32) is None
            assert await led.validated_ledger_index() == 4_321

    run(main())


# -- autofill --------------------------------------------------------------------------------


def test_autofill_fee_sequence_lls_and_never_network_id():
    async def main():
        async with FakeRpc(validated=1_000) as rpc, ledger_for(rpc.url) as led:
            rpc.add_account("rAlice", sequence=7)
            tx = {"TransactionType": "Payment", "Account": "rAlice", "Destination": PROOF_DEST,
                  "Amount": "1"}
            out = await led.autofill(tx)
            assert out["Fee"] == "12"  # open_ledger_fee 10 floors at 12
            assert out["Sequence"] == 7
            assert out["LastLedgerSequence"] == 1_020
            assert "NetworkID" not in out
            assert "Fee" not in tx  # the input is not mutated
            rpc.open_ledger_fee = "50"
            assert (await led.autofill(tx))["Fee"] == "50"

    run(main())


def test_autofill_falls_back_to_12_drops_and_keeps_given_fields():
    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.add_account("rAlice", sequence=7)
            rpc.answer_error("fee", "unknownCmd")
            out = await led.autofill({"TransactionType": "Payment", "Account": "rAlice",
                                      "Sequence": 3, "LastLedgerSequence": 999})
            assert out["Fee"] == "12" and out["Sequence"] == 3 and out["LastLedgerSequence"] == 999

    run(main())


def test_autofill_caps_a_fee_spike():
    async def main():
        async with FakeRpc(open_ledger_fee="5000000") as rpc, ledger_for(rpc.url) as led:
            rpc.add_account("rAlice")
            out = await led.autofill({"TransactionType": "Payment", "Account": "rAlice"})
            assert int(out["Fee"]) <= chain.MAX_FEE_DROPS

    run(main())


def test_autofill_refuses_an_unknown_account():
    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            with pytest.raises(AutofillError, match="rNobody"):
                await led.autofill({"TransactionType": "Payment", "Account": "rNobody"})
            with pytest.raises(AutofillError):
                await led.autofill({"TransactionType": "Payment"})

    run(main())


# -- simulate --------------------------------------------------------------------------------


def test_simulate_success_and_failures():
    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.add_account("rAlice")
            tx = {"TransactionType": "Payment", "Account": "rAlice", "Destination": PROOF_DEST,
                  "Amount": "1", "Fee": "12", "Sequence": 1, "LastLedgerSequence": 1_020}
            r = await led.simulate(tx)
            assert r["engine_result"] == "tesSUCCESS"
            assert rpc.calls_for("simulate")[0]["tx_json"] == tx
            rpc.simulate_result = "tecUNFUNDED_PAYMENT"
            with pytest.raises(SimulateFailed, match="tecUNFUNDED_PAYMENT") as ei:
                await led.simulate(tx)
            assert ei.value.engine_result == "tecUNFUNDED_PAYMENT"
            with pytest.raises(SimulateFailed, match="srcActNotFound"):
                await led.simulate({**tx, "Account": "rNobody"})

    run(main())


def test_simulate_fails_closed_when_no_endpoint_supports_it():
    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.answer_error("simulate", "unknownCmd", "Unknown method.")
            with pytest.raises(SimulateFailed):
                await led.simulate({"TransactionType": "Payment", "Account": "rAlice"})

    run(main())


# -- submit_and_wait -------------------------------------------------------------------------


def test_submit_and_wait_returns_the_validated_tx():
    w = Wallet.create()
    blob = signed_blob(w)

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.polls_until_validated = 3
            r = await led.submit_and_wait(blob)
            assert r["validated"] is True
            assert r["hash"] == tx_hash(blob)
            assert r["meta"]["TransactionResult"] == "tesSUCCESS"
            assert rpc.calls_for("submit")[0]["tx_blob"] == blob
            assert len(rpc.calls_for("tx")) == 3

    run(main())


def test_submit_and_wait_raises_on_a_tem_or_tef_preliminary_result():
    w = Wallet.create()

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            for prelim in ("temBAD_FEE", "tefPAST_SEQ"):
                rpc.submit_result = prelim
                with pytest.raises(SubmitFailed, match=prelim) as ei:
                    await led.submit_and_wait(signed_blob(w))
                assert ei.value.engine_result == prelim
            assert rpc.calls_for("tx") == []

    run(main())


def test_submit_and_wait_raises_when_the_last_ledger_passes():
    w = Wallet.create()
    blob = signed_blob(w, lls=1_005)

    async def main():
        async with FakeRpc(validated=1_000) as rpc, ledger_for(rpc.url) as led:
            rpc.submit_result = "terQUEUED"
            rpc.polls_until_validated = None
            rpc.advance_per_ledger_query = 2
            with pytest.raises(SubmitFailed, match="1005") as ei:
                await led.submit_and_wait(blob)
            assert "terQUEUED" in str(ei.value)
            assert ei.value.hash == tx_hash(blob)

    run(main())


def test_submit_and_wait_raises_on_a_tec_final_result():
    w = Wallet.create()

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.final_result = "tecPATH_DRY"
            with pytest.raises(SubmitFailed, match="tecPATH_DRY") as ei:
                await led.submit_and_wait(signed_blob(w))
            assert ei.value.engine_result == "tecPATH_DRY"
            assert ei.value.result is not None and ei.value.result["validated"] is True

    run(main())


def test_submit_and_wait_refuses_a_blob_without_last_ledger_sequence():
    w = Wallet.create()
    blob = signed_blob(w, lls=None)

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            with pytest.raises(SubmitFailed, match="LastLedgerSequence"):
                await led.submit_and_wait(blob)
            assert rpc.calls == []  # never reached the network

    run(main())


def test_submit_and_wait_tolerates_a_transient_outage_while_polling():
    w = Wallet.create()
    blob = signed_blob(w)

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.polls_until_validated = 2
            failures = {"n": 3}
            orig = rpc.responders["tx"]

            def flaky(params):
                if failures["n"] > 0:
                    failures["n"] -= 1
                    raise RpcError("tooBusy", "The server is too busy to help you now.")
                return orig(params)

            rpc.answer("tx", flaky)
            r = await led.submit_and_wait(blob)
            assert r["validated"] is True and failures["n"] == 0

    run(main())


def test_submit_and_wait_gives_up_after_a_sustained_outage():
    w = Wallet.create()
    blob = signed_blob(w)

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            rpc.answer_error("tx", "tooBusy")
            with pytest.raises(LedgerUnavailable):
                await led.submit_and_wait(blob)

    run(main())


def test_submit_and_wait_still_polls_when_the_submit_call_itself_fails():
    """A blob may have reached the network even though the HTTP answer was lost."""
    w = Wallet.create()
    blob = signed_blob(w)

    async def main():
        async with FakeRpc() as rpc, ledger_for(rpc.url) as led:
            h = tx_hash(blob)
            rpc.txs[h] = {**binarycodec.decode(blob), "hash": h, "validated": False}
            rpc.answer_error("submit", "tooBusy")
            r = await led.submit_and_wait(blob)
            assert r["hash"] == h and r["validated"] is True

    run(main())
