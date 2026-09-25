"""Legality of a single-slot Builder change (spec §4.1 step 4)."""

from __future__ import annotations

import asyncio

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from lfg_fly.body.legality import legal_changes, legal_changes_async
from lfg_fly.brain.senses import NONE, SLOTS

# Background, Back, Body, Clothing, Mouth, Eyebrows, Eyes, Head, Accessory
HERO = ("Blue", "None", "male", "Hoodie", "Smile", "Flat", "Open", "Cap", "None")


def closet(*items: tuple[str, str, int]) -> list[dict]:
    return [{"slot": s, "value": v, "count": c} for s, v, c in items]


def always(slot: str, value: str) -> bool:
    return True


def test_hero_has_nine_slots():
    assert len(HERO) == len(SLOTS)


def test_plain_legal_change():
    assert legal_changes(HERO, closet(("Head", "Beanie", 1)), set(), always) == [("Head", "Beanie")]


def test_body_slot_is_never_legal():
    assert legal_changes(HERO, closet(("Body", "female", 3)), set(), always) == []


def test_unknown_slot_is_ignored():
    assert legal_changes(HERO, closet(("Tail", "Long", 1)), set(), always) == []


def test_count_zero_is_not_legal():
    assert legal_changes(HERO, closet(("Head", "Beanie", 0)), set(), always) == []


def test_duplicate_assets_sum_their_counts():
    assets = closet(("Head", "Beanie", 0), ("Head", "Beanie", 1))
    assert legal_changes(HERO, assets, set(), always) == [("Head", "Beanie")]


def test_listed_unit_is_not_legal():
    assets = closet(("Head", "Beanie", 1), ("Head", "Crown", 1))
    assert legal_changes(HERO, assets, {("Head", "Beanie")}, always) == [("Head", "Crown")]


def test_current_value_is_not_a_change():
    assert legal_changes(HERO, closet(("Head", "Cap", 2)), set(), always) == []


def test_none_is_legal_iff_the_closet_holds_that_slots_none_unit():
    # The hero wears a Cap; the Closet holds a Head None unit → taking the hat off is legal.
    assert legal_changes(HERO, closet(("Head", NONE, 1)), set(), always) == [("Head", NONE)]
    # No None unit for Head → not legal, even though None needs no layer.
    assert legal_changes(HERO, closet(("Head", "Beanie", 0)), set(), always) == []
    # A None unit for another slot does not unlock Head None.
    assert legal_changes(HERO, closet(("Clothing", NONE, 1)), set(), always) == [
        ("Clothing", NONE)
    ]
    # The hero already wears None on Back → Back None is not a change.
    assert legal_changes(HERO, closet(("Back", NONE, 4)), set(), always) == []


def test_none_never_asks_the_resolver():
    asked: list[tuple[str, str]] = []

    def resolves(slot: str, value: str) -> bool:
        asked.append((slot, value))
        return True

    legal_changes(HERO, closet(("Head", NONE, 1), ("Head", "Beanie", 1)), set(), resolves)
    assert asked == [("Head", "Beanie")]


def test_value_that_does_not_resolve_is_not_legal():
    def resolves(slot: str, value: str) -> bool:
        return value != "Tiara"

    assets = closet(("Head", "Tiara", 1), ("Head", "Beanie", 1))
    assert legal_changes(HERO, assets, set(), resolves) == [("Head", "Beanie")]


def test_resolver_is_not_asked_for_illegal_units():
    asked: list[tuple[str, str]] = []

    def resolves(slot: str, value: str) -> bool:
        asked.append((slot, value))
        return True

    assets = closet(
        ("Body", "female", 1),  # Body slot
        ("Head", "Cap", 1),  # current value
        ("Head", "Crown", 0),  # count 0
        ("Head", "Tiara", 1),  # listed
        ("Head", "Beanie", 1),  # legal
    )
    legal_changes(HERO, assets, {("Head", "Tiara")}, resolves)
    assert asked == [("Head", "Beanie")]


def test_order_is_slot_order_then_value_regardless_of_closet_order():
    assets = closet(("Head", "Crown", 1), ("Background", "Red", 1), ("Head", "Beanie", 1))
    assert legal_changes(HERO, assets, set(), always) == [
        ("Background", "Red"),
        ("Head", "Beanie"),
        ("Head", "Crown"),
    ]


def test_resolve_404_over_http_excludes_the_value():
    """A layer that 404s (empty body, image/png, as LFG does) is not legal; a 200 is."""

    async def layer(request: web.Request) -> web.Response:
        assert request.query["body"] == "male"
        if request.query["value"] == "Beanie":
            return web.Response(status=200, body=b"\x89PNG", content_type="image/png")
        return web.Response(status=404, body=b"", content_type="image/png")

    app = web.Application()
    app.router.add_get("/api/layer", layer)

    async def main() -> list[tuple[str, str]]:
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            url = server.make_url("/api/layer")

            async def resolves(slot: str, value: str) -> bool:
                params = {"body": "male", "trait": slot, "value": value}
                async with session.get(url, params=params) as r:
                    return r.status == 200

            assets = closet(("Head", "Tiara", 1), ("Head", "Beanie", 1), ("Head", NONE, 1))
            return await legal_changes_async(HERO, assets, set(), resolves)

    # Tiara 404s and is gone; Beanie (200) and None (never asked) stay, in value order.
    assert asyncio.run(main()) == [("Head", "Beanie"), ("Head", NONE)]


def test_async_accepts_a_sync_resolver_too():
    async def main():
        return await legal_changes_async(HERO, closet(("Head", "Beanie", 1)), set(), always)

    assert asyncio.run(main()) == [("Head", "Beanie")]
