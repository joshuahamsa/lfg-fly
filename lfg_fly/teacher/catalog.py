"""What the hero can wear, read from LFG's public API (spec §3.1, Catalog).

Values are the union of /api/rarity?body=<b> over all five bodies (Body only from
the hero's class), kept iff /api/layer?body=<hero> resolves them. Every non-Body
slot also gets "None", which needs no layer. Layer thumbs are cached on disk; a
cached 404 is an empty file.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

from lfg_fly.brain.senses import NONE, SLOTS

BODIES = ("ape", "female", "male", "milady", "skeleton")


def http_get(url: str, timeout: float = 30) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "lfg-fly/0.0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


@dataclass(frozen=True)
class Catalog:
    body: str
    values: dict[str, list[str]]
    odds: dict[str, dict[str, float]]
    api: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Catalog:
        return cls(**json.loads(text))


def layer_cache_path(cache: Path, body: str, slot: str, value: str) -> Path:
    key = hashlib.sha1(f"{body}|{slot}|{value}".encode()).hexdigest()
    return cache / "layers" / body / slot / f"{key}.img"


def _ensure_layer(api: str, cache: Path, body: str, slot: str, value: str, get) -> bool:
    path = layer_cache_path(cache, body, slot, value)
    if path.exists():
        return path.stat().st_size > 0
    query = urlencode({"body": body, "trait": slot, "value": value, "thumb": "1"})
    status, data = get(f"{api}/api/layer?{query}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = status == 200 and len(data) > 0
    path.write_bytes(data if ok else b"")
    return ok


def build_catalog(api: str, cache: Path, body: str = "male", get=http_get) -> Catalog:
    union: dict[str, set[str]] = {s: set() for s in SLOTS}
    odds: dict[str, dict[str, float]] = {}
    for b in BODIES:
        status, data = get(f"{api}/api/rarity?body={b}")
        if status != 200:
            raise RuntimeError(f"/api/rarity?body={b} -> HTTP {status}")
        for slot, rows in json.loads(data)["slots"].items():
            if slot not in union or (slot == "Body" and b != body):
                continue
            for row in rows:
                union[slot].add(str(row["value"]))
                if b == body and row.get("enabled", True):
                    odds.setdefault(slot, {})[str(row["value"])] = float(row["odds_pct"])
    values: dict[str, list[str]] = {}
    for slot in SLOTS:
        kept = [
            v for v in sorted(union[slot] - {NONE})
            if _ensure_layer(api, cache, body, slot, v, get)
        ]
        if slot != "Body":
            kept.append(NONE)
        values[slot] = kept
    return Catalog(body=body, values=values, odds=odds, api=api)
