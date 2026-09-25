"""The outbox: alerts for the operator, and posts made without X credentials.

Contract §body/outbox.py. Spec §1 (a stamp mismatch "stops the loop with an outbox
alert"), §4.2 (an UNKNOWN record "writes an outbox alert until the operator clears it")
and §4.5 ("without credentials, posts go to outbox/").

One JSON file per item under `FLY_DATA_DIR/<network>/outbox/`, named by its UTC instant
so a directory listing is a timeline: `20260925T150000.000000Z-alert.json`. A cleared
alert is renamed with a `.cleared` suffix and kept, never deleted.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from lfg_fly import paths

KINDS = ("alert", "post")
_STAMP_FMT = "%Y%m%dT%H%M%S.%fZ"


def _utc(when: datetime | None) -> datetime:
    """`when` as an aware UTC datetime; None is now, a naive value is taken as UTC."""
    if when is None:
        return datetime.now(timezone.utc)
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def _place(tmp: Path, directory: Path, stamp: str, kind: str) -> Path:
    """Move the finished tmp file to the first free `<stamp>[.n]-<kind>.json`.

    `os.link` creates the name only if it does not exist yet, so two writes at the same
    microsecond both survive. `.n` sorts after the bare stamp ('.' > '-') and keeps the
    `*-<kind>.json` shape that `alerts` lists.
    """
    n = 0
    while True:
        suffix = "" if n == 0 else f".{n}"
        target = directory / f"{stamp}{suffix}-{kind}.json"
        try:
            os.link(tmp, target)
        except FileExistsError:
            n += 1
            continue
        except OSError:  # a filesystem without hard links: settle for create-if-absent
            if target.exists():
                n += 1
                continue
            os.replace(tmp, target)
            return target
        tmp.unlink()
        return target


def write(network: str, kind: str, payload: dict, when: datetime | None = None) -> Path:
    """Write one outbox item and return its path.

    The payload is serialised before anything touches the disk, so an unserialisable
    payload raises TypeError and leaves nothing behind. The file holds
    `{"kind", "network", "when" (ISO-8601 UTC), "payload"}`. A post's PNG goes beside the
    JSON under the same stem; that is the voice's business, hence the returned path.
    """
    if kind not in KINDS:
        raise ValueError(f"outbox kind must be one of {KINDS}, not {kind!r}")
    directory = paths.outbox_dir(network)  # refuses an unknown network
    at = _utc(when)
    text = json.dumps(
        {"kind": kind, "network": network, "when": at.isoformat(), "payload": payload},
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    directory.mkdir(parents=True, exist_ok=True)
    stamp = at.strftime(_STAMP_FMT)
    tmp = directory / f".{stamp}-{kind}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        return _place(tmp, directory, stamp, kind)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def alerts(network: str) -> list[dict]:
    """Every alert not yet cleared, oldest first, each with its `path` added.

    A file that cannot be parsed is still listed (`payload` None, `error` set): an alert
    the operator cannot read is not an alert that has gone away.
    """
    directory = paths.outbox_dir(network)
    if not directory.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(directory.glob("*-alert.json")):
        if not path.is_file():
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(item, dict):
                raise ValueError("outbox item is not a JSON object")
        except (OSError, ValueError) as exc:
            item = {"kind": "alert", "network": network, "when": None, "payload": None,
                    "error": f"corrupt outbox file: {exc}"}
        item.setdefault("kind", "alert")
        item["path"] = path
        out.append(item)
    return out


def clear(network: str, path: str | Path) -> None:
    """Mark an item handled by renaming it to `<name>.cleared`.

    Only a file directly inside this network's outbox may be cleared; anything else is a
    ValueError. A missing file raises FileNotFoundError.
    """
    path = Path(path)
    directory = paths.outbox_dir(network).resolve()
    if path.resolve().parent != directory:
        raise ValueError(f"{path} is not in the {network} outbox {directory}")
    path.rename(path.with_name(path.name + ".cleared"))
