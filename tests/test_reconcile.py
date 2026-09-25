"""The reconcile rule (contract §body/reconcile.py; spec §4.2, §7 "every reconcile branch").

Hermetic: the ledger is a stand-in that answers `nft_uri`, the metadata fetch is a
coroutine over a dict, and nothing here talks to LFG (§4.2 step 2: never LFG's index).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from lfg_fly.body import reconcile as RC
from lfg_fly.body import records as R

STAMP = R.Stamp(network="testnet", lfg_api_base="http://localhost:8177", wallet="rFlyWallet")
BEFORE = ("Blue", "None", "male", "Hoodie", "Grin", "Flat", "Laser", "Crown", "None")
AFTER = ("Blue", "None", "male", "Hoodie", "Grin", "Flat", "Monocle", "Pirate Hat", "None")
OTHER = ("Red", "None", "male", "Hoodie", "Grin", "Flat", "Laser", "Crown", "None")
T0 = datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


T0_ISO = T0.isoformat()


def make(state="submitted", submitted_at=T0_ISO, **kw) -> R.Record:
    fields = dict(
        date="2026-09-25", stamp=STAMP, version="fly-v1", rarity_head=None,
        seed=R.date_seed_hex("2026-09-25", "fly-v1"), hero="A" * 64,
        before=list(BEFORE), after=list(AFTER),
        changes=[{"slot": "Eyes", "value": "Monocle"}, {"slot": "Head", "value": "Pirate Hat"}],
        candidates=[], considered=3, total=3, neuron_stats={}, inputs_hash={},
        state=state, equip_id="eq-1", submitted_at=submitted_at,
    )
    fields.update(kw)
    return R.Record(**fields)


def attrs(look) -> list[dict]:
    return [{"trait_type": s, "value": v} for s, v in zip(RC.SLOTS, look, strict=True)]


class FakeLedger:
    """Only `nft_uri`, the one ledger read §4.2 step 2 makes."""

    def __init__(self, uri: str | None = "ipfs://QmHero"):
        self.uri = uri
        self.calls: list[tuple[str, str]] = []

    async def nft_uri(self, nft_id: str, owner: str) -> str | None:
        self.calls.append((nft_id, owner))
        return self.uri


def metadata_of(look):
    async def fetch(uri: str) -> dict:
        fetch.uris.append(uri)
        return {"name": "LFG #1", "image": "ipfs://QmImg", "attributes": attrs(look)}

    fetch.uris = []
    return fetch


# ------------------------------------------------------------------ classify


@pytest.mark.parametrize(
    "status, expected",
    [
        ({"state": "done", "resolution": "committed"}, "done"),
        ({"state": "done", "resolution": None}, "done"),
        ({"state": "failed", "resolution": "reverted"}, "failed"),
        ({"state": "failed", "resolution": None}, "failed"),
        ({"state": "failed"}, "failed"),
        ({"state": "failed", "resolution": "uncertain"}, "UNKNOWN"),
    ],
)
def test_classify_follows_the_4_2_table(status, expected):
    assert RC.classify(status) == expected


def test_classify_refuses_a_non_terminal_or_unknown_state():
    with pytest.raises(ValueError):
        RC.classify({"state": "running", "resolution": None})
    with pytest.raises(ValueError):
        RC.classify({"state": "weird"})
    with pytest.raises(ValueError):
        RC.classify({})


# ------------------------------------------------------------------ attributes_to_look


def test_attributes_to_look_orders_by_slot_and_pads_with_none():
    look = RC.attributes_to_look(
        [{"trait_type": "Head", "value": "Crown"}, {"trait_type": "Body", "value": "male"},
         {"trait_type": "Background", "value": "Blue"}]
    )
    assert look == ("Blue", "None", "male", "None", "None", "None", "None", "Crown", "None")
    assert len(look) == 9


def test_attributes_to_look_ignores_unknown_traits_and_reads_none_values():
    look = RC.attributes_to_look(
        [{"trait_type": "Head", "value": None}, {"trait_type": "Mood", "value": "x"},
         {"trait_type": "Eyes", "value": "Laser"}]
    )
    assert look[RC.SLOTS.index("Head")] == "None"
    assert look[RC.SLOTS.index("Eyes")] == "Laser"


def test_attributes_to_look_round_trips_a_full_look():
    assert RC.attributes_to_look(attrs(AFTER)) == AFTER


# ------------------------------------------------------------------ resolve_unknown


def test_resolve_refuses_before_ten_minutes_have_passed():
    ledger = FakeLedger()
    fetch = metadata_of(AFTER)
    for early in (T0, T0 + timedelta(minutes=9, seconds=59)):
        with pytest.raises(RC.NotYet) as info:
            run(RC.resolve_unknown(ledger, make(), fetch, early))
        assert info.value.ready_at == T0 + RC.UNKNOWN_WAIT
    assert ledger.calls == [] and fetch.uris == []  # nothing read while it is too early


def test_resolve_reads_the_ledger_uri_and_the_intended_look_is_done():
    ledger = FakeLedger("ipfs://QmNew")
    fetch = metadata_of(AFTER)
    verdict = run(RC.resolve_unknown(ledger, make(), fetch, T0 + RC.UNKNOWN_WAIT))
    assert verdict == "done"
    assert ledger.calls == [("A" * 64, "rFlyWallet")]  # the hero, from the record's wallet
    assert fetch.uris == ["ipfs://QmNew"]


def test_resolve_original_look_is_failed():
    assert run(RC.resolve_unknown(FakeLedger(), make(), metadata_of(BEFORE),
                                  T0 + timedelta(hours=1))) == "failed"


def test_resolve_anything_else_stays_unknown():
    assert run(RC.resolve_unknown(FakeLedger(), make(), metadata_of(OTHER),
                                  T0 + timedelta(hours=1))) == "UNKNOWN"


def test_resolve_without_a_ledger_uri_stays_unknown_and_fetches_nothing():
    fetch = metadata_of(AFTER)
    assert run(RC.resolve_unknown(FakeLedger(None), make(), fetch, T0 + timedelta(hours=1))) \
        == "UNKNOWN"
    assert fetch.uris == []


def test_resolve_with_unreadable_metadata_stays_unknown():
    async def broken(uri: str) -> dict:
        return {"name": "no attributes here"}

    assert run(RC.resolve_unknown(FakeLedger(), make(), broken, T0 + timedelta(hours=1))) \
        == "UNKNOWN"


def test_resolve_accepts_a_naive_now_as_utc_and_a_z_suffix():
    rec = make(submitted_at="2026-09-25T15:00:00Z")
    naive = datetime(2026, 9, 25, 15, 11)  # UTC by convention
    assert run(RC.resolve_unknown(FakeLedger(), rec, metadata_of(AFTER), naive)) == "done"


def test_resolve_without_submitted_at_does_not_wait():
    """The loop stamps `submitted_at` in the write that precedes POST /api/equip, so a
    record without one never reached the submit step: the ledger read is safe at once."""
    rec = make(state="pending", equip_id=None, submitted_at=None)
    assert run(RC.resolve_unknown(FakeLedger(), rec, metadata_of(BEFORE), T0)) == "failed"


# ------------------------------------------------------------------ metadata URIs


def test_metadata_url_maps_ipfs_to_the_gateway_and_keeps_http():
    assert RC.metadata_url("ipfs://QmAbc/1.json") == RC.IPFS_GATEWAY + "QmAbc/1.json"
    assert RC.metadata_url("ipfs://ipfs/QmAbc") == RC.IPFS_GATEWAY + "QmAbc"
    assert RC.metadata_url("https://x.test/m.json") == "https://x.test/m.json"
    assert RC.metadata_url("http://x.test/m.json") == "http://x.test/m.json"
    for bad in ("ftp://x", "", "QmAbc", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            RC.metadata_url(bad)


def test_fetch_metadata_reads_json_over_http():
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def go():
        app = web.Application()

        async def meta(request):
            return web.json_response({"attributes": attrs(AFTER)})

        app.router.add_get("/m.json", meta)
        async with TestServer(app) as server:
            got = await RC.fetch_metadata(str(server.make_url("/m.json")))
        return got

    got = run(go())
    assert RC.attributes_to_look(got["attributes"]) == AFTER
