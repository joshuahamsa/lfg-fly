"""Posts: the fly's own account (spec §4.5).

`compose` writes the post text after the spec's example:

    🪰 Day 12. Tried 147 outfits in 61 ms of fly-brain time; 9,812 neurons fired.
    Head: Crown → Pirate Hat · Eyes: Laser → Monocle. Critic's read: "Retired Space Pirate".

The count of outfits is what the fly actually simulated (`considered`); a cap is
disclosed as "Tried N of M outfits" (§4.1 step 4). Below the naming threshold
the look posts as untitled (§3.2). The text is kept within X's 280 weighted
characters, dropping flourishes before it truncates.

`publish` puts the post out. Without X credentials (or without an X poster wired
in) it goes to `FLY_DATA_DIR/<network>/outbox/` through `lfg_fly.body.outbox.write`
as `<stamp>-post.json`, with the card as a PNG of the same stem beside it. The
outbox module is imported lazily, and the writer is injectable, so this module
has no import-time dependency on the body. A session token or an X secret never
appears in a payload or a result.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from lfg_fly.brain.senses import SLOTS
from lfg_fly.voice.card import change_lines, with_from

FLY = "\U0001fab0"  # 🪰
X_MAX_CHARS = 280
UNTITLED = "Untitled."
X_ENV = {
    "api_key": "FLY_X_API_KEY",
    "api_secret": "FLY_X_API_SECRET",
    "access_token": "FLY_X_ACCESS_TOKEN",
    "access_secret": "FLY_X_ACCESS_SECRET",
}
# X counts these code-point ranges as one character and everything else as two.
_SINGLE = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))

Writer = Callable[..., Path]  # outbox.write(network, kind, payload, when=None) -> Path
Poster = Callable[[str, bytes], dict]  # an X client: (text, png bytes) -> its result


def _get(record: Any, name: str, default: Any = None) -> Any:
    """A field of a Record dataclass or of a plain dict."""
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def x_length(text: str) -> int:
    """X's weighted length: 1 for most Latin/Greek/Cyrillic and punctuation code points,
    2 for everything else (emoji, arrows, CJK)."""
    return sum(1 if any(lo <= ord(ch) <= hi for lo, hi in _SINGLE) else 2 for ch in text)


def day_number(first: date | str, today: date | str) -> int:
    """Day 1 is the first move's date."""
    first = date.fromisoformat(first) if isinstance(first, str) else first
    today = date.fromisoformat(today) if isinstance(today, str) else today
    return (today - first).days + 1


def _count(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _text(record: Any, name: str | None, day: int, *, brain_time: bool,
          from_values: bool) -> str:
    considered = int(_get(record, "considered", 0) or 0)
    total = int(_get(record, "total", considered) or considered)
    stats = _get(record, "neuron_stats", None) or {}
    ms = int(round(float(stats.get("ms", 0) or 0)))
    fired = int(stats.get("neurons_fired", 0) or 0)
    changes = list(_get(record, "changes", None) or [])
    before = _get(record, "before", None)

    if considered < total:
        tried = f"Tried {considered:,} of {_count(total, 'outfit')}"
    else:
        tried = f"Tried {_count(considered, 'outfit')}"
    time = f" in {ms:,} ms of fly-brain time" if brain_time else f" in {ms:,} ms"
    moves = " · ".join(change_lines(with_from(changes, before) if from_values else changes))
    read = f'Critic\'s read: "{name}".' if name else UNTITLED
    return (f"{FLY} Day {int(day)}. {tried}{time}; {_count(fired, 'neuron')} fired. "
            f"{moves}. {read}")


def compose(record: Any, name: str | None, day: int) -> str:
    """The §4.5 post text for a record: day number, outfits tried, ms, neurons fired,
    the changes ("Slot: before → after", joined by " · "; a None value reads as
    "nothing") and the critic's read, or "Untitled." when `name` is None.

    Kept within X_MAX_CHARS: first "of fly-brain time" goes, then the before values
    of the changes, and only then the text is truncated with an ellipsis."""
    variants = ((True, True), (False, True), (True, False), (False, False))
    text = ""
    for brain_time, from_values in variants:
        text = _text(record, name, day, brain_time=brain_time, from_values=from_values)
        if x_length(text) <= X_MAX_CHARS:
            return text
    while text and x_length(text + "…") > X_MAX_CHARS:
        text = text[:-1]
    return text.rstrip() + "…"


def x_credentials(env: Mapping[str, str] | None = None) -> dict[str, str] | None:
    """The fly's X OAuth 1.0a credentials from the environment, or None unless all four
    are set and non-empty. Never logged, never written."""
    env = os.environ if env is None else env
    creds = {k: (env.get(v) or "").strip() for k, v in X_ENV.items()}
    return creds if all(creds.values()) else None


def _default_writer() -> Writer:
    from lfg_fly.body import outbox  # lazy: the body is a separate module (and author)

    return outbox.write


def _png(card: Image.Image) -> bytes:
    buf = io.BytesIO()
    card.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def publish(cfg: Any, text: str, card: Image.Image, record: Any, *,
            writer: Writer | None = None, poster: Poster | None = None,
            name: str | None = None, day: int | None = None,
            env: Mapping[str, str] | None = None) -> dict:
    """Put the post out (§4.5). With a `poster` it goes to X and the result is
    {"channel": "x", "text", "result"}. Otherwise, and always without X credentials,
    it goes to the outbox: `writer(network, "post", payload)` (default
    `lfg_fly.body.outbox.write`) with the card saved as a PNG of the same stem beside
    the JSON; the result is {"channel": "outbox", "reason", "path", "card", "text"}
    and is what the loop stores as `record.post`. The payload carries the text, the
    date, the hero, the before/after looks, the changes, the name and the day, and
    never a credential."""
    png = _png(card)
    if poster is not None:
        return {"channel": "x", "text": text, "result": poster(text, png)}
    reason = "no_credentials" if x_credentials(env) is None else "no_poster"
    write = writer if writer is not None else _default_writer()
    payload = {
        "text": text,
        "date": _get(record, "date"),
        "hero": _get(record, "hero"),
        "before": list(_get(record, "before", None) or []),
        "after": list(_get(record, "after", None) or []),
        "changes": list(_get(record, "changes", None) or []),
        "name": name,
        "day": day,
        "slots": list(SLOTS),
        "reason": reason,
    }
    path = Path(write(_get(cfg, "network"), "post", payload))
    png_path = path.with_suffix(".png")
    tmp = png_path.with_name(png_path.name + ".tmp")
    tmp.write_bytes(png)
    os.replace(tmp, png_path)
    return {"channel": "outbox", "reason": reason, "path": str(path), "card": str(png_path),
            "text": text}


__all__ = ["compose", "day_number", "publish", "x_credentials", "x_length"]
