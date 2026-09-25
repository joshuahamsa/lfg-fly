"""Day records: the fly's audit trail, written before anything is submitted.

Contract §body/records.py. Spec §4.1 step 6 (record before submitting), §4.2 (the states a
reconcile resolves) and §1 (every record is stamped `{network, lfg_api_base, wallet}` and
loaders refuse a mismatch, so a stray artifact from another network or wallet is never
resumed by the wrong stack: the 2026-09-08 fixture-leak failure class).

One JSON file per UTC day under `FLY_DATA_DIR/<network>/records/<date>.json`, written
atomically (tmp + rename) and rewritten in place as the state advances.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import date, datetime, timedelta
from pathlib import Path

from lfg_fly import paths

Look = tuple[str, ...]  # nine values in lfg_fly.brain.senses.SLOTS order

STATES = ("dry_run", "pending", "submitted", "done", "failed", "UNKNOWN")
NON_TERMINAL_STATES = ("pending", "submitted", "UNKNOWN")  # a reconcile must resolve these
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class StampMismatch(RuntimeError):
    """A record's `{network, lfg_api_base, wallet}` differs from the running fly's."""


@dataclass
class Stamp:
    """Which stack a record belongs to (spec §1). All three fields must match to load it."""

    network: str
    lfg_api_base: str
    wallet: str


@dataclass
class Record:
    """One day's move (spec §4.1 step 6): inputs, every candidate, the choice, the outcome."""

    date: str  # YYYY-MM-DD (UTC)
    stamp: Stamp
    version: str  # checkpoint version
    rarity_head: str | None  # supply snapshot hash the rarity head was fitted from
    seed: str  # hex of SHA-256(date | version); date_seed() is its first 8 bytes
    hero: str  # nft_id
    before: list[str]  # Look
    after: list[str]  # Look
    changes: list[dict]  # [{"slot", "value"}], in order
    candidates: list[dict]  # [{"look","changes","taste","rarity","cost","score","considered"}]
    considered: int  # "considered N of M"
    total: int
    neuron_stats: dict  # from the brain: neurons_fired, ms, ...
    inputs_hash: dict  # sha256 of the supply snapshot, economy state and closet
    state: str  # one of STATES
    equip_id: str | None = None
    resolution: str | None = None  # the equip session's resolution, verbatim
    error: str | None = None
    post: dict | None = None  # what was posted, or put in the outbox
    submitted_at: str | None = None  # ISO-8601 UTC; §4.2 waits 10 min past it
    equip_status: dict | None = None  # the last /api/equip/{id} body, for the audit trail


def _date_str(day: str | date) -> str:
    """A validated YYYY-MM-DD string (it becomes a file name, so it must be exactly that)."""
    if isinstance(day, datetime):
        return day.date().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    if not isinstance(day, str) or not _DATE_RE.fullmatch(day):
        raise ValueError(f"record date must be YYYY-MM-DD, not {day!r}")
    try:
        date.fromisoformat(day)
    except ValueError:
        raise ValueError(f"record date is not a calendar date: {day!r}") from None
    return day


def path_for(network: str, day: str | date) -> Path:
    """`FLY_DATA_DIR/<network>/records/<date>.json`."""
    return paths.records_dir(network) / f"{_date_str(day)}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` via a same-directory tmp file and `os.replace`.

    A reader never sees a partial record; a failure leaves the old record untouched and no
    tmp file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    with contextlib.suppress(OSError):  # durability of the rename itself; best effort
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write(record: Record) -> Path:
    """Write (or rewrite) the record for its day, atomically, and return the path.

    The network comes from the record's own stamp. The state must be one of STATES, and
    the record must be JSON-serialisable; either failure raises before the disk is touched.
    """
    if record.state not in STATES:
        raise ValueError(f"unknown record state {record.state!r}; expected one of {STATES}")
    path = path_for(record.stamp.network, record.date)
    text = json.dumps(asdict(record), indent=2, sort_keys=True, ensure_ascii=False)
    _atomic_write_text(path, text)
    return path


def read(path: str | Path, expect: Stamp) -> Record:
    """Load a record, refusing it unless its stamp equals `expect` in every field.

    The stamp is checked before anything else is looked at. Unknown keys are ignored (a
    newer writer, an older reader); a missing required key or an unknown state is a
    ValueError.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: not a record")
    found_raw = raw.get("stamp")
    if not isinstance(found_raw, dict):
        raise StampMismatch(f"{path}: record carries no stamp; expected {expect}")
    found = Stamp(**{f.name: found_raw.get(f.name) for f in fields(Stamp)})
    differing = [f.name for f in fields(Stamp) if getattr(found, f.name) != getattr(expect, f.name)]
    if differing:
        raise StampMismatch(
            f"{path}: stamp differs in {', '.join(differing)}: found {found}, expected {expect}"
        )
    names = {f.name for f in fields(Record)}
    required = {
        f.name for f in fields(Record) if f.default is MISSING and f.default_factory is MISSING
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"{path}: record is missing {missing}")
    if raw["state"] not in STATES:
        raise ValueError(f"{path}: unknown record state {raw['state']!r}")
    kwargs = {k: v for k, v in raw.items() if k in names}
    kwargs["stamp"] = found
    return Record(**kwargs)


def _record_files(network: str) -> list[tuple[date, Path]]:
    """Every `<date>.json` in the network's records dir, oldest first. Tmp files, backups
    and anything else are not records and are skipped."""
    directory = paths.records_dir(network)
    if not directory.is_dir():
        return []
    out: list[tuple[date, Path]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix != ".json" or not _DATE_RE.fullmatch(path.stem) or not path.is_file():
            continue
        try:
            day = date.fromisoformat(path.stem)
        except ValueError:
            continue
        out.append((day, path))
    return out


def non_terminal(network: str, expect: Stamp) -> list[Record]:
    """Every pending / submitted / UNKNOWN record, oldest first (spec §4.1 step 2, §4.2).

    Reads every record in the directory, so a foreign record anywhere in it refuses the
    run with StampMismatch, terminal or not.
    """
    return [
        rec
        for rec in (read(path, expect) for _, path in _record_files(network))
        if rec.state in NON_TERMINAL_STATES
    ]


def recent_looks(network: str, expect: Stamp, days: int, today: date) -> set[Look]:
    """The `after` of every done record dated within the last `days` days, inclusive of
    both `today - days` and today: the 30-day tabu source (spec §4.1 step 5)."""
    if isinstance(today, datetime):
        today = today.date()
    cutoff = today - timedelta(days=days)
    looks: set[Look] = set()
    for day, path in _record_files(network):
        if day < cutoff:
            continue
        rec = read(path, expect)
        if rec.state == "done":
            looks.add(tuple(rec.after))
    return looks


def _seed_digest(day: str | date, version: str) -> bytes:
    return hashlib.sha256(f"{_date_str(day)}|{version}".encode()).digest()


def date_seed_hex(day: str | date, version: str) -> str:
    """The full SHA-256(date | version) hex digest recorded as `Record.seed`."""
    return _seed_digest(day, version).hex()


def date_seed(day: str | date, version: str) -> int:
    """The day's RNG seed (spec §4.1 step 5): the first 8 bytes of SHA-256(date | version),
    little-endian, so a day's choice is reproducible from the record alone."""
    return int.from_bytes(_seed_digest(day, version)[:8], "little")
