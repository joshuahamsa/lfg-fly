"""The reconcile rule (spec §4.2; contract §body/reconcile.py).

An equip session's outcome maps to a record state by the §4.2 table:

    done                                    -> done
    failed, resolution reverted or null     -> failed
    failed, resolution uncertain            -> UNKNOWN

An UNKNOWN record, or a `submitted` one whose session LFG no longer holds, is
resolved from the ledger and never from LFG's index (`/api/nfts`, `/api/economy`
lag an indeterminate modify): wait at least ten minutes past the submit, past any
LastLedgerSequence the pending modify could carry; read the hero's current URI
(`Ledger.nft_uri`: clio `nft_info`, else `account_nfts`); fetch that metadata and
compare its attributes. The intended look means done, the original look means
failed, anything else stays UNKNOWN. The loop (`lfg_fly.body.loop`) owns the
waiting, the alert and the stop; this module only decides.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from lfg_fly.body.records import Record
from lfg_fly.brain.senses import NONE, SLOTS

Look = tuple[str, ...]

UNKNOWN_WAIT = timedelta(minutes=10)  # §4.2 step 1
IPFS_GATEWAY = "https://ipfs.io/ipfs/"
METADATA_TIMEOUT = 30.0
TERMINAL = ("done", "failed")
_IPFS_RE = re.compile(r"^ipfs://(?:ipfs/)?(.+)$", re.IGNORECASE)


class NotYet(RuntimeError):
    """Too early to read the ledger: `ready_at` is when the ten minutes are up."""

    def __init__(self, ready_at: datetime):
        super().__init__(f"UNKNOWN can be resolved from {ready_at.isoformat()}")
        self.ready_at = ready_at


def classify(equip_status: dict) -> str:
    """`done` | `failed` | `UNKNOWN` for a terminal `/api/equip/{id}` body (§4.2 table).

    `uncertain` is the only resolution that leaves the outcome open; a `failed` with
    `reverted`, `null` or anything else is a failed equip (LFG treats a null as a generic
    crash before the modify). A non-terminal or unknown state is a ValueError: it is the
    caller's job to poll until the session settles.
    """
    state = equip_status.get("state")
    if state == "done":
        return "done"
    if state == "failed":
        return "UNKNOWN" if equip_status.get("resolution") == "uncertain" else "failed"
    raise ValueError(f"equip session is not terminal: state={state!r}")


def attributes_to_look(attributes: list[dict]) -> Look:
    """The nine values in SLOTS order from a metadata `attributes` list; slots the list
    does not name, or names with a null value, read as "None" (docs/lfg-api.md §2)."""
    got: dict[str, str] = {}
    for a in attributes or ():
        if not isinstance(a, dict):
            continue
        slot, value = a.get("trait_type"), a.get("value")
        if slot in SLOTS:
            got[slot] = NONE if value is None else str(value)
    return tuple(got.get(slot, NONE) for slot in SLOTS)


def _utc(when: datetime) -> datetime:
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(
        timezone.utc)


def parse_when(text: str) -> datetime:
    """An ISO-8601 stamp (a trailing `Z` accepted) as an aware UTC datetime."""
    return _utc(datetime.fromisoformat(text.replace("Z", "+00:00")))


def ready_at(record: Record) -> datetime | None:
    """When the §4.2 wait is over for this record, or None when there is nothing to wait
    for: the loop stamps `submitted_at` in the write that precedes `POST /api/equip`, so a
    record without one never reached the submit step."""
    if not record.submitted_at:
        return None
    return parse_when(record.submitted_at) + UNKNOWN_WAIT


async def resolve_unknown(
    ledger: Any,
    record: Record,
    fetch_metadata: Callable[[str], Awaitable[dict]],
    now: datetime,
) -> str:
    """§4.2 steps 1–3 for one record: `done` | `failed` | `UNKNOWN`.

    Raises `NotYet` while `record.submitted_at + 10 min` lies ahead of `now` (a naive
    `now` is UTC). Then reads the hero's current URI through `ledger.nft_uri(hero, wallet)`
    (the record's stamp names the wallet), fetches the metadata and compares the look:
    the intended `after` is done, the original `before` is failed, and a missing URI,
    unreadable metadata or any other look stays UNKNOWN. LFG is never asked.
    """
    now = _utc(now)
    due = ready_at(record)
    if due is not None and now < due:
        raise NotYet(due)
    uri = await ledger.nft_uri(record.hero, record.stamp.wallet)
    if not uri:
        return "UNKNOWN"
    try:
        meta = await fetch_metadata(uri)
        attributes = meta.get("attributes") if isinstance(meta, dict) else None
        if not isinstance(attributes, list):
            return "UNKNOWN"
        look = attributes_to_look(attributes)
    except (aiohttp.ClientError, OSError, ValueError, TypeError, KeyError):
        return "UNKNOWN"
    if look == tuple(record.after):
        return "done"
    if look == tuple(record.before):
        return "failed"
    return "UNKNOWN"


def metadata_url(uri: str) -> str:
    """The URL to fetch for a token URI: `ipfs://<cid>[/path]` through the public gateway,
    http(s) as is; anything else is a ValueError."""
    if not isinstance(uri, str) or not uri:
        raise ValueError("empty token URI")
    m = _IPFS_RE.match(uri)
    if m:
        return IPFS_GATEWAY + m.group(1)
    if uri.startswith(("https://", "http://")):
        return uri
    raise ValueError(f"unsupported token URI scheme: {uri[:16]!r}")


async def fetch_metadata(uri: str, timeout: float = METADATA_TIMEOUT) -> dict:
    """GET the token's metadata JSON (the default `fetch_metadata` of the loop)."""
    url = metadata_url(uri)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise ValueError(f"metadata {url}: HTTP {resp.status}")
            data = await resp.json(content_type=None)
    if not isinstance(data, dict):
        raise ValueError(f"metadata {url}: not a JSON object")
    return data


__all__ = [
    "IPFS_GATEWAY", "NotYet", "SLOTS", "TERMINAL", "UNKNOWN_WAIT", "attributes_to_look",
    "classify", "fetch_metadata", "metadata_url", "parse_when", "ready_at", "resolve_unknown",
]
