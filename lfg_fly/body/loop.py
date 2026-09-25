"""The move loop: the fly's day (spec §4.1), its safety (§4.4) and its reconcile (§4.2).

`move` runs the eight steps in the spec's order:

1. Refuse unless `FLY_ENABLED=1` or `--dry-run`; take the single-flight lock; prove the
   chain's identity (§4.4) once, before anything is asked of LFG.
2. Sign in fresh with the `agent` provider; the token lives in the client object only
   and `POST /api/logout` runs in a `finally` (the client's `__aexit__`).
3. Reconcile every non-terminal record first (§4.2). A record that stays UNKNOWN, or one
   stamped for another stack (§1), stops the loop with an outbox alert. A done, failed or
   UNKNOWN record for today means no second move: one look change per day (§4.4).
4. Read `/api/nfts`, `/api/economy` (its `z_order` cross-checked against the checkpoint's
   pinned order; any difference refuses) and `/api/rarity/supply`; pick the hero.
5. Candidates and the decision (`legality`, `candidates`, `decide`): every legal
   single-slot change plus stay, taste under `c_ref`, rarity under live concentrations,
   temperature τ with the date seed, at most three greedy steps, the 30-day tabu.
6. Write `records/<date>.json` BEFORE submitting.
7. One `POST /api/equip` with every change, then poll `/api/equip/{id}` and classify.
8. Post (§4.5): the card and text go to X when credentials exist, else to the outbox.

`--dry-run` stops after step 6 with `state="dry_run"` and never calls equip.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import io
import json
import logging
import math
import os
from collections.abc import Awaitable, Callable, Iterator
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from PIL import Image

from lfg_fly import paths
from lfg_fly.body import outbox, reconcile, records
from lfg_fly.body.candidates import Candidate, changes_between
from lfg_fly.body.client import LfgClient, LfgError
from lfg_fly.body.decide import considered_of, plan_day
from lfg_fly.body.legality import legal_changes_async
from lfg_fly.body.records import Record, Stamp, StampMismatch
from lfg_fly.brain import readout
from lfg_fly.voice import (
    before_after,
    compose,
    day_number,
    load_critic_records,
    nearest_name,
    publish,
    with_from,
)

log = logging.getLogger(__name__)

Look = tuple[str, ...]
Fetcher = Callable[[str], Awaitable[Any]]

HERO_BODY = "male"  # the hero is a mutable, non-blank male (spec §4.1 step 4, §5 step 5)
LOCK_FILE = "move.lock"
EQUIP_TIMEOUT = 15 * 60.0  # seconds to wait for /api/equip/{id} to settle
EQUIP_POLL = 3.0
IMAGE_TIMEOUT = 15.0
FINAL_STATES = ("done", "failed", "UNKNOWN")  # a day with one of these never moves again
CARD_PLACEHOLDER = (28, 28, 34)


class LoopError(RuntimeError):
    """Base of the loop's own refusals and stops."""


class LoopRefused(LoopError):
    """The run did not start, or stopped before anything was submitted (no alert)."""


class LoopStopped(LoopError):
    """The run stopped on an unresolved state and wrote an outbox alert (§4.2, §1)."""


# ------------------------------------------------------------------ §4.4 gates


def lock_path(network: str) -> Path:
    return paths.network_dir(network) / LOCK_FILE


@contextlib.contextmanager
def single_flight(network: str) -> Iterator[None]:
    """The §4.4 single-flight lock: `flock` on `FLY_DATA_DIR/<network>/move.lock`, held
    for the whole run; a second runner is refused at once."""
    path = lock_path(network)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise LoopRefused(f"another move holds the lock {path}") from e
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def read_wallet_file(path: Path, network: str | None = None) -> tuple[str, str]:
    """`(address, regular_seed)` from testnet's `wallet.json` (spec §5.3, written by
    `fly setup`). Tolerant of the key names `address|account|wallet|classic_address`
    and `regular_seed|regular_key_seed|regular.seed`; a file stamped for another network
    is refused. The seed is returned, never logged."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a wallet file")
    if network is not None and data.get("network") not in (None, network):
        raise ValueError(f"{path} belongs to {data.get('network')!r}, not {network!r}")
    address = next((data[k] for k in ("address", "account", "wallet", "classic_address",
                                      "master_address") if data.get(k)), None)
    seed = next((data[k] for k in ("regular_seed", "regular_key_seed") if data.get(k)), None)
    if seed is None and isinstance(data.get("regular"), dict):
        seed = data["regular"].get("seed")
    if not address or not seed:
        raise ValueError(f"{path}: needs the wallet address and its RegularKey seed")
    return str(address), str(seed)


def make_signer(cfg: Any, ledger: Any) -> Any:
    """The fly's `Signer` (§4.3): mainnet from `FLY_REGULAR_SEED` + `FLY_WALLET`, testnet
    from `wallet.json`. Imported lazily so a test's stand-in signer needs no xrpl key."""
    from lfg_fly.body.signer import Signer

    if cfg.network == "mainnet":
        seed = cfg.regular_seed
        account = (os.environ.get("FLY_WALLET") or "").strip()
        if not seed:
            raise LoopRefused("FLY_REGULAR_SEED is not set (mainnet signs with the RegularKey)")
        if not account:
            raise LoopRefused("FLY_WALLET (the fly's wallet address) is not set")
    else:
        wallet_file = paths.wallet_path(cfg.network)
        if not wallet_file.exists():
            raise LoopRefused(f"no wallet at {wallet_file}: run `fly setup keygen` first")
        account, seed = read_wallet_file(wallet_file, network=cfg.network)
    return Signer(seed, account, ledger, cfg)


# ------------------------------------------------------------------ helpers


def sha256_json(obj: Any) -> str:
    """sha256 of the canonical JSON of `obj` (sorted keys, compact), for `inputs_hash`."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def zorder_mismatch(api_z_order: Any, pinned: Any) -> str | None:
    """Why `/api/economy`'s `z_order` differs from the checkpoint's pinned `ZOrder`, or None.

    Every slot's layer z must agree (compared as floats) and the set of
    `(trait_type, value, z)` overrides must be identical (spec §4.1 step 3).
    """
    if not isinstance(api_z_order, dict):
        return "z_order missing from /api/economy"
    layers = api_z_order.get("layers") or {}
    for slot, z in pinned.layer_z.items():
        if slot not in layers:
            return f"z_order.layers lacks {slot!r} (pinned z={z})"
        try:
            got = float(layers[slot])
        except (TypeError, ValueError):
            return f"z_order.layers[{slot!r}] is not a number: {layers[slot]!r}"
        if got != float(z):
            return f"z_order.layers[{slot!r}] is {got}, the checkpoint pins {float(z)}"
    api_over = set()
    for o in api_z_order.get("z_overrides") or []:
        try:
            api_over.add((str(o["trait_type"]), str(o["value"]), float(o["z"])))
        except (KeyError, TypeError, ValueError):
            return f"malformed z_override {o!r}"
    pinned_over = {(s, v, float(z)) for (s, v), z in pinned.overrides.items()}
    if api_over != pinned_over:
        return (f"z_order.z_overrides differ: LFG {sorted(api_over)}, "
                f"checkpoint {sorted(pinned_over)}")
    return None


def _eligible(ch: dict) -> bool:
    return bool(ch.get("mutable")) and not ch.get("blank") and ch.get("body") == HERO_BODY


def pick_hero(characters: list[dict], requested: str | None, remembered: str | None) -> dict:
    """The hero (spec §4.1 step 3, §5 step 5): the requested one (`--hero` / `FLY_HERO`),
    else the one the last record names, else the first mutable non-blank male. A requested
    or remembered hero that is missing, immutable, blank or not male is refused rather than
    silently swapped."""
    by_id = {c.get("nft_id"): c for c in characters}
    for label, want in (("requested", requested), ("last record's", remembered)):
        if not want:
            continue
        ch = by_id.get(want)
        if ch is None:
            raise LoopRefused(f"the {label} hero {want} is not among this wallet's characters")
        if not _eligible(ch):
            raise LoopRefused(f"the {label} hero {want} is not a mutable, non-blank "
                              f"{HERO_BODY} character")
        return ch
    for ch in characters:
        if _eligible(ch):
            return ch
    raise LoopRefused(f"no hero: no mutable, non-blank {HERO_BODY} character in this wallet")


def load_rarity_head(network: str) -> readout.RarityHead | None:
    """The newest nightly rarity head under `snapshots/` (`rarity-head-<hash>.npz/.json`,
    spec §2 Readout), or None when no retrain has run yet."""
    directory = paths.snapshots_dir(network)
    if not directory.is_dir():
        return None
    candidates = [p for p in directory.glob("rarity-head-*.json") if p.is_file()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: (p.stat().st_mtime, p.name))
    head = readout.load_head(newest.with_suffix(".npz"))
    if not isinstance(head, readout.RarityHead):
        raise ValueError(f"{newest}: not a rarity head")
    return head


def _record_paths(network: str) -> list[Path]:
    directory = paths.records_dir(network)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("????-??-??.json") if p.is_file())


def last_record(network: str, stamp: Stamp) -> Record | None:
    files = _record_paths(network)
    return records.read(files[-1], stamp) if files else None


def first_move_date(network: str, stamp: Stamp, today: date) -> date:
    """Day 1 of the fly's posts (§4.5): the date of the earliest record that was not a
    dry run; today when there is none."""
    for path in _record_paths(network):
        rec = records.read(path, stamp)
        if rec.state != "dry_run":
            return date.fromisoformat(rec.date)
    return today


async def fetch_image(url: str, timeout: float = IMAGE_TIMEOUT) -> Image.Image:
    """The hero's on-chain image for the card (§4.5); a placeholder when it cannot be
    fetched, so a picture never blocks the day."""
    try:
        target = reconcile.metadata_url(url)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.get(target) as resp:
                if resp.status != 200:
                    raise ValueError(f"HTTP {resp.status}")
                data = await resp.read()
        im = Image.open(io.BytesIO(data))
        im.load()
        return im.convert("RGBA")
    except Exception as e:  # noqa: BLE001 — a missing picture is not a failed move
        log.warning("hero image %s: %s: %s", url, type(e).__name__, e)
        return Image.new("RGB", (480, 480), CARD_PLACEHOLDER)


def _num(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _cand_dict(c: Candidate) -> dict:
    return {
        "look": list(c.look),
        "changes": [{"slot": s, "value": v} for s, v in c.changes],
        "taste": _num(c.taste), "rarity": _num(c.rarity), "cost": _num(c.cost),
        "score": _num(c.score), "considered": bool(c.considered),
    }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class _Stats:
    """Collects the brain's per-batch stats into the record's `neuron_stats`."""

    def __init__(self, brain: Any):
        self.brain = brain
        self.fired: list[int] = []
        self.ms = 0.0
        self.looks = 0
        self.taste_batches = 0
        self.rarity_batches = 0

    def note(self, n: int, kind: str) -> None:
        stats = getattr(self.brain, "stats", None) or {}
        self.ms += float(stats.get("ms", 0.0) or 0.0)
        if kind == "taste":
            self.taste_batches += 1
            if "neurons_fired" in stats:
                self.fired.append(int(stats["neurons_fired"]))
        else:
            self.rarity_batches += 1
        self.looks += n

    def summary(self) -> dict:
        fired = int(round(sum(self.fired) / len(self.fired))) if self.fired else 0
        return {"neurons_fired": fired, "ms": self.ms, "looks": self.looks,
                "taste_batches": self.taste_batches, "rarity_batches": self.rarity_batches}


def _alert(cfg: Any, kind: str, message: str, record: Record | None = None, **extra) -> Path:
    payload = {"kind": kind, "message": message, **extra}
    if record is not None:
        payload.update(date=record.date, hero=record.hero, state=record.state,
                       equip_id=record.equip_id)
    log.error("outbox alert %s: %s", kind, message)
    return outbox.write(cfg.network, "alert", payload)


def _apply_status(record: Record, status: dict) -> None:
    """Fold a terminal `/api/equip/{id}` body into the record (§4.2 table)."""
    record.equip_status = status
    record.resolution = status.get("resolution")
    record.error = status.get("error")
    record.state = reconcile.classify(status)


# ------------------------------------------------------------------ §4.2 reconcile


async def reconcile_record(record: Record, *, cfg: Any, client: LfgClient, ledger: Any,
                           now: Callable[[], datetime], sleep: Callable[[float], Awaitable[None]],
                           fetch_metadata: Fetcher, equip_timeout: float = EQUIP_TIMEOUT,
                           equip_poll: float = EQUIP_POLL) -> Record:
    """Settle one pending / submitted / UNKNOWN record (§4.2) and rewrite it.

    A `submitted` record asks `/api/equip/{id}` first (polling while it still runs); a
    session LFG no longer holds (404), a `pending` record and an UNKNOWN one go to the
    ledger: wait out the ten minutes (sleeping the remainder), read the hero's URI, compare.
    A record that stays UNKNOWN writes an outbox alert and raises `LoopStopped`.
    """
    if record.state in ("pending", "submitted") and record.equip_id:
        try:
            status = await client.equip_status(record.equip_id)
            if status.get("state") not in reconcile.TERMINAL:
                status = await client.wait(client.equip_status, record.equip_id,
                                           set(reconcile.TERMINAL), timeout=equip_timeout,
                                           every=equip_poll)
            _apply_status(record, status)
        except LfgError as e:
            if e.status != 404:
                raise
            record.error = f"equip session {record.equip_id} no longer exists at LFG"
        except TimeoutError as e:
            record.error = str(e)
    if record.state in records.NON_TERMINAL_STATES:
        try:
            verdict = await reconcile.resolve_unknown(ledger, record, fetch_metadata, now())
        except reconcile.NotYet as e:
            wait = max(0.0, (e.ready_at - now()).total_seconds())
            log.info("record %s: waiting %.0fs before the ledger read (§4.2)", record.date, wait)
            await sleep(wait)
            verdict = await reconcile.resolve_unknown(ledger, record, fetch_metadata, e.ready_at)
        note = f"resolved from the ledger as {verdict}"
        record.error = f"{record.error}; {note}" if record.error else note
        record.state = verdict
    records.write(record)
    if record.state == "UNKNOWN":
        _alert(cfg, "unknown_move", f"the move of {record.date} is still UNKNOWN after the "
               "ledger read; the loop stops until the operator clears this alert (§4.2)",
               record)
        raise LoopStopped(f"record {record.date} is UNKNOWN; not moving")
    log.info("record %s reconciled: %s", record.date, record.state)
    return record


# ------------------------------------------------------------------ §4.1 the move


async def move(
    cfg: Any,
    brain: Any,
    *,
    dry_run: bool,
    today: date | None = None,
    ledger: Any = None,
    client: LfgClient | None = None,
    signer: Any = None,
    hero: str | None = None,
    rarity_head: readout.RarityHead | None = None,
    now: Callable[[], datetime] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    fetch_metadata: Fetcher | None = None,
    fetch_image: Fetcher | None = None,
    equip_timeout: float = EQUIP_TIMEOUT,
    equip_poll: float = EQUIP_POLL,
    critic_dir: Path | None = None,
) -> Record:
    """One day of the fly (spec §4.1); returns the day's record.

    `brain` is a `FlyBrain` (or anything with `taste(looks)`, `rarity(looks, conc, head)`,
    `head.taste_sd`, `catalog`, `pinned_zorder()` and `stats`). `ledger`, `client` and
    `signer` default to the real ones for `cfg`; `hero` overrides the pick (else `FLY_HERO`,
    else the last record's hero, else the first mutable non-blank male); `rarity_head`
    defaults to the newest nightly head under `snapshots/` (none: rarity scores 0 and the
    record says so). `now`, `sleep`, `fetch_metadata`, `fetch_image`, `equip_timeout` and
    `equip_poll` exist for tests and the §4.2 wait.
    """
    if not (cfg.enabled or dry_run):
        raise LoopRefused("FLY_ENABLED is not 1 and this is not a dry run (spec §4.4)")
    clock = now or _now_utc
    sleep = sleep or asyncio.sleep
    fetch_metadata = fetch_metadata or reconcile.fetch_metadata
    fetch_image_ = fetch_image or globals()["fetch_image"]
    today = today or clock().date()
    with single_flight(cfg.network):
        own_ledger = ledger is None
        if own_ledger:
            from lfg_fly.body.chain import Ledger

            ledger = Ledger(cfg.rpc_urls, cfg.network)
        try:
            await ledger.identity_check(cfg.expected_ledger_hash)  # §4.4, before any LFG call
            if signer is None:
                signer = make_signer(cfg, ledger)
            client = client if client is not None else LfgClient(cfg.api_base)
            async with client:  # logs out in __aexit__, whatever happens below (§4.1 step 1)
                await client.sign_in(signer, ledger)
                return await _day(cfg, brain, client, ledger, dry_run=dry_run, today=today,
                                  hero=hero, rarity_head=rarity_head, clock=clock, sleep=sleep,
                                  fetch_metadata=fetch_metadata, fetch_image=fetch_image_,
                                  equip_timeout=equip_timeout, equip_poll=equip_poll,
                                  critic_dir=critic_dir)
        finally:
            if own_ledger:
                await ledger.close()


async def _day(cfg: Any, brain: Any, client: LfgClient, ledger: Any, *, dry_run: bool,
               today: date, hero: str | None, rarity_head: readout.RarityHead | None,
               clock: Callable[[], datetime], sleep: Callable[[float], Awaitable[None]],
               fetch_metadata: Fetcher, fetch_image: Fetcher, equip_timeout: float,
               equip_poll: float, critic_dir: Path | None) -> Record:
    network = cfg.network
    stamp = Stamp(network=network, lfg_api_base=cfg.api_base, wallet=str(client.wallet))

    # 2. reconcile first; never proceed while an outcome is unknown (§4.1 step 2, §4.2, §1)
    try:
        pending = records.non_terminal(network, stamp)
    except StampMismatch as e:
        _alert(cfg, "stamp_mismatch", f"{e}; a record from another stack sits in "
               f"{paths.records_dir(network)} and the loop refuses to run beside it (§1)")
        raise LoopStopped(f"record stamp mismatch: {e}") from e
    for rec in pending:
        await reconcile_record(rec, cfg=cfg, client=client, ledger=ledger, now=clock, sleep=sleep,
                               fetch_metadata=fetch_metadata, equip_timeout=equip_timeout,
                               equip_poll=equip_poll)
    today_path = records.path_for(network, today)
    if today_path.exists():
        existing = records.read(today_path, stamp)
        if existing.state in FINAL_STATES:
            raise LoopRefused(f"already moved on {today}: the record is {existing.state} "
                              "(one look change per day, spec §4.4)")

    # 3. read state (§4.1 step 3)
    nfts = await client.nfts()
    economy = await client.economy()
    supply = await client.rarity_supply()
    pinned = getattr(brain, "pinned_zorder", None)
    if pinned is not None:
        why = zorder_mismatch(economy.get("z_order"), pinned())
        if why:
            raise LoopRefused(f"z_order mismatch: {why}; the checkpoint's renderer no longer "
                              "matches LFG's, refusing to move")
    else:
        log.warning("brain has no pinned_zorder(); the z_order cross-check was skipped")
    characters = list(economy.get("characters") or [])
    requested = hero or (os.environ.get("FLY_HERO") or "").strip() or None
    remembered = None
    if requested is None:
        last = last_record(network, stamp)
        remembered = last.hero if last is not None else None
    hero_ch = pick_hero(characters, requested, remembered)
    hero_id = str(hero_ch["nft_id"])
    body = str(hero_ch.get("body") or HERO_BODY)
    before = reconcile.attributes_to_look(hero_ch.get("attributes") or [])
    nft_row = next((n for n in nfts.get("nfts") or [] if n.get("nft_id") == hero_id), {})
    closet = economy.get("closet") or {}
    closet_assets = list(closet.get("assets") or [])
    listed: set[tuple[str, str]] = set()  # the client has no Closet Market read; LFG guards

    # 4–5. candidates and the decision (§4.1 steps 4–5)
    head = rarity_head if rarity_head is not None else load_rarity_head(network)
    tabu = records.recent_looks(network, stamp, cfg.tabu_days, today)
    rng = np.random.default_rng(records.date_seed(today, cfg.version))
    stats = _Stats(brain)
    catalog = getattr(brain, "catalog", None)

    async def legal_fn(look: Look):
        return await legal_changes_async(
            look, closet_assets, listed, lambda s, v: client.layer_resolves(body, s, v))

    def brain_taste(looks: list[Look]):
        out = np.asarray(brain.taste(looks), dtype=np.float64)
        stats.note(len(looks), "taste")
        return out

    def brain_rarity(looks: list[Look]):
        if head is None:
            return np.zeros(len(looks))
        conc = readout.concentrations(looks, supply, catalog)
        out = np.asarray(brain.rarity(looks, conc, head), dtype=np.float64)
        stats.note(len(looks), "rarity")
        return out

    taste_sd = float(getattr(getattr(brain, "head", None), "taste_sd", 1.0))
    final, _chain, steps = await plan_day(
        before, legal_fn=legal_fn, brain_taste=brain_taste, brain_rarity=brain_rarity,
        cost_fn=None, tabu=tabu, cfg=cfg, rng=rng, taste_sd=taste_sd)
    flat = [c for step in steps for c in step]
    considered, total = considered_of(flat)
    changes = [{"slot": s, "value": v} for s, v in changes_between(before, final)]

    # 6. record before submitting (§4.1 step 6)
    record = Record(
        date=today.isoformat(), stamp=stamp, version=cfg.version,
        rarity_head=head.snapshot_hash if head is not None else None,
        seed=records.date_seed_hex(today, cfg.version), hero=hero_id,
        before=list(before), after=list(final), changes=changes,
        candidates=[_cand_dict(c) for c in flat], considered=considered, total=total,
        neuron_stats=stats.summary(),
        inputs_hash={"supply": sha256_json(supply), "economy": sha256_json(economy),
                     "closet": sha256_json(closet_assets)},
        state="dry_run" if dry_run else "pending",
    )
    records.write(record)
    log.info("%s %s: %s -> %s (%d changes, considered %d of %d)", today,
             "dry run" if dry_run else "move", before, final, len(changes), considered, total)
    if dry_run:
        return record
    if not changes:  # stay won: nothing to submit, the day is done (§4.1 step 5)
        record.state = "done"
        records.write(record)
        await _post(cfg, record, client, nft_row, stamp, today, fetch_image, critic_dir)
        return record

    # 7. submit: one POST /api/equip with every change, then poll (§4.1 step 7)
    record.submitted_at = clock().isoformat()
    records.write(record)
    try:
        started = await client.equip(hero_id, changes)
    except LfgError as e:
        if 400 <= e.status < 500:  # refused before anything started
            record.state, record.error = "failed", f"equip refused: {e}"
            records.write(record)
            raise
        record.state, record.error = "UNKNOWN", f"equip answer indeterminate: {e}"
        records.write(record)
        _alert(cfg, "unknown_move", record.error, record)
        raise
    except Exception as e:  # the answer was lost: the equip may or may not have started
        record.state, record.error = "UNKNOWN", f"equip answer lost: {type(e).__name__}: {e}"
        records.write(record)
        _alert(cfg, "unknown_move", record.error, record)
        raise
    record.equip_id = str(started.get("id"))
    record.equip_status = started
    record.state = "submitted"
    records.write(record)
    try:
        status = await client.wait(client.equip_status, record.equip_id, set(reconcile.TERMINAL),
                                   timeout=equip_timeout, every=equip_poll)
    except TimeoutError as e:
        record.state, record.error = "UNKNOWN", f"equip session did not settle: {e}"
        records.write(record)
        _alert(cfg, "unknown_move", record.error, record)
        return record
    _apply_status(record, status)
    records.write(record)
    if record.state == "UNKNOWN":
        _alert(cfg, "unknown_move", f"the equip session of {record.date} ended "
               f"'{record.resolution}' ({record.error}); the next run resolves it from the "
               "ledger (§4.2)", record)
        return record
    if record.state == "done":
        await _post(cfg, record, client, nft_row, stamp, today, fetch_image, critic_dir)
    return record


async def _post(cfg: Any, record: Record, client: LfgClient, nft_row: dict, stamp: Stamp,
                today: date, fetch_image: Fetcher, critic_dir: Path | None) -> None:
    """§4.1 step 8 / §4.5: name the look, draw the card, compose and publish. A failure here
    is recorded in `record.post` and never undoes a done day."""
    try:
        critic = load_critic_records(critic_dir or paths.repo_root() / "data" / "critic")
        name = nearest_name(tuple(record.after), critic) if critic else None
        day = day_number(first_move_date(cfg.network, stamp, today), record.date)
        before_url = str(nft_row.get("image") or "")
        after_url = before_url
        if record.changes:
            with contextlib.suppress(Exception):
                fresh = await client.nfts()
                row = next((n for n in fresh.get("nfts") or [] if n.get("nft_id") == record.hero),
                           {})
                after_url = str(row.get("image") or before_url)
        before_img = await fetch_image(before_url)
        after_img = await fetch_image(after_url)
        card = before_after(before_img, after_img, day, with_from(record.changes, record.before))
        text = compose(record, name, day)
        record.post = publish(cfg, text, card, record, name=name, day=day)
    except Exception as e:  # noqa: BLE001 — the post is the day's last, least step
        log.exception("post failed")
        record.post = {"channel": "none", "error": f"{type(e).__name__}: {e}"}
    records.write(record)


__all__ = [
    "EQUIP_POLL", "EQUIP_TIMEOUT", "HERO_BODY", "LoopError", "LoopRefused", "LoopStopped",
    "fetch_image", "first_move_date", "last_record", "load_rarity_head", "lock_path",
    "make_signer", "move", "pick_hero", "read_wallet_file", "reconcile_record", "sha256_json",
    "single_flight", "zorder_mismatch",
]
