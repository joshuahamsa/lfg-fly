"""Which single-slot Builder changes are legal today (spec §4.1 step 4).

A change ``(slot, value)`` from the hero's current look is legal when:

- the slot is one of the 8 non-Body slots;
- the value is in the Closet with count > 0 and is not listed on the Closet Market;
- the value differs from what the hero wears in that slot;
- ``"None"`` is legal iff the Closet holds that slot's None unit (harvest credits one
  for every empty donor slot; equip skips the affinity check for it);
- any other value must resolve on the hero's body (``GET /api/layer`` → 200, the same
  ``resolve_layer`` the server's equip gate uses).

This is a disclosed non-fly component: it only says what is *possible*; the brain
chooses (``lfg_fly.body.decide``).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable

from lfg_fly.brain.senses import NON_BODY_SLOTS, NONE, SLOTS

Look = tuple[str, ...]
Change = tuple[str, str]

_SLOT_INDEX = {slot: i for i, slot in enumerate(SLOTS)}
_NON_BODY = frozenset(NON_BODY_SLOTS)


def _available(closet_assets: Iterable[dict]) -> dict[Change, int]:
    """Closet units per (slot, value), summed over duplicate rows; non-Body slots only."""
    counts: dict[Change, int] = {}
    for asset in closet_assets:
        slot, value = asset.get("slot"), asset.get("value")
        if slot not in _NON_BODY or not isinstance(value, str):
            continue
        counts[(slot, value)] = counts.get((slot, value), 0) + int(asset.get("count") or 0)
    return counts


def _pre_resolve(hero: Look, closet_assets: Iterable[dict], listed: set[Change]) -> list[Change]:
    """Every rule except the layer resolution, in (slot order, value) order."""
    hero = tuple(hero)
    if len(hero) != len(SLOTS):
        raise ValueError(f"a look has {len(SLOTS)} slots, got {len(hero)}")
    out = []
    for (slot, value), count in _available(closet_assets).items():
        if count <= 0 or (slot, value) in listed:
            continue
        if hero[_SLOT_INDEX[slot]] == value:
            continue
        out.append((slot, value))
    out.sort(key=lambda c: (_SLOT_INDEX[c[0]], c[1]))
    return out


def legal_changes(
    hero: Look,
    closet_assets: list[dict],
    listed: set[Change],
    resolves: Callable[[str, str], bool],
) -> list[Change]:
    """The legal single-slot changes from ``hero`` (spec §4.1 step 4).

    ``closet_assets`` is ``/api/economy``'s ``closet.assets`` (``{slot, value, count}``);
    ``listed`` holds the ``(slot, value)`` units reserved by a Closet Market ask;
    ``resolves(slot, value)`` answers whether ``/api/layer`` returns 200 for the hero's
    body. ``None`` never asks the resolver. The result is ordered by slot (``SLOTS``
    order) then value, so it does not depend on the Closet's row order.
    """
    out = []
    for slot, value in _pre_resolve(hero, closet_assets, listed):
        if value != NONE and not resolves(slot, value):
            continue
        out.append((slot, value))
    return out


async def legal_changes_async(
    hero: Look,
    closet_assets: list[dict],
    listed: set[Change],
    resolves: Callable[[str, str], Awaitable[bool] | bool],
) -> list[Change]:
    """``legal_changes`` with an async resolver (``LfgClient.layer_resolves``).

    The cheap rules run first; only the survivors other than ``None`` hit the resolver,
    all concurrently. A sync resolver is accepted too.
    """
    pre = _pre_resolve(hero, closet_assets, listed)
    ask = [c for c in pre if c[1] != NONE]

    async def one(slot: str, value: str) -> bool:
        r = resolves(slot, value)
        return bool(await r) if inspect.isawaitable(r) else bool(r)

    answers = await asyncio.gather(*(one(s, v) for s, v in ask))
    ok = dict(zip(ask, answers, strict=True))
    return [c for c in pre if c[1] == NONE or ok[c]]
