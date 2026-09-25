"""Naming at runtime, with no API (spec §3.2, "Naming at runtime").

A post names the fly's look after the nearest critic-named look by slot-weighted
overlap, labelled "critic's read". Below a similarity threshold the look posts as
untitled. The critic's records are the dicts `lfg_fly.teacher.critic.assemble`
writes to `data/critic/<round>.jsonl`: each has the pair's two looks `a` and `b`
(nine values in SLOTS order), the `winner`, and one idea per look (`idea_A`,
`idea_B`). Both looks of a pair are judged looks and each carries its own idea;
`winner` only breaks ties.

Weighting: slots are weighted by visual prominence on the card. Background,
Clothing and Head fill most of the pixels (2 each); Back, Body, Mouth, Eyebrows,
Eyes and Accessory are smaller (1 each). Similarity is the weight of the slots
whose values agree, over the total weight 12. The default threshold 0.6 therefore
needs at least 8 of 12: for instance all three heavy slots plus two light ones.

Pairs the critic flipped on (`flipped: true`) were dropped by the QC rule and are
skipped here too; so are empty ideas and malformed records.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from lfg_fly.brain.senses import SLOTS

Look = tuple[str, ...]

SLOT_WEIGHTS: dict[str, int] = {
    "Background": 2, "Back": 1, "Body": 1, "Clothing": 2, "Mouth": 1,
    "Eyebrows": 1, "Eyes": 1, "Head": 2, "Accessory": 1,
}
_WEIGHTS = tuple(SLOT_WEIGHTS[s] for s in SLOTS)
TOTAL_WEIGHT = sum(_WEIGHTS)
THRESHOLD = 0.6
MAX_NAME_CHARS = 80  # critic.read_results caps ideas at 80; enforced again here


def similarity(look: Sequence[str], other: Sequence[str]) -> float:
    """Slot-weighted overlap in [0, 1]: the weight of agreeing slots over TOTAL_WEIGHT."""
    if len(look) != len(SLOTS) or len(other) != len(SLOTS):
        raise ValueError(f"a look has {len(SLOTS)} slots, got {len(look)} and {len(other)}")
    hit = sum(w for w, x, y in zip(_WEIGHTS, look, other, strict=True) if x == y)
    return hit / TOTAL_WEIGHT


def _idea(record: dict, key: str) -> str | None:
    idea = record.get(key)
    if not isinstance(idea, str):
        return None
    idea = " ".join(idea.split())
    return idea[:MAX_NAME_CHARS] if idea else None


def named_looks(critic_records: Iterable[dict]) -> list[tuple[Look, str, bool]]:
    """Every (look, idea, is_winner) the critic named, in record order.

    Skips flipped pairs, looks of the wrong length and empty ideas."""
    out: list[tuple[Look, str, bool]] = []
    for r in critic_records:
        if not isinstance(r, dict) or r.get("flipped"):
            continue
        winner = r.get("winner")
        for side, key in (("a", "idea_A"), ("b", "idea_B")):
            look = r.get(side)
            if not isinstance(look, (list, tuple)) or len(look) != len(SLOTS):
                continue
            idea = _idea(r, key)
            if idea is None:
                continue
            out.append((tuple(str(v) for v in look), idea, winner == key[-1]))
    return out


def nearest(look: Sequence[str], critic_records: Iterable[dict]) -> tuple[str | None, float]:
    """The idea of the most similar critic-named look and its similarity, before any
    threshold. Ties go to a pair's winner, then to the earlier record. (None, 0.0)
    when nothing is named."""
    best: tuple[float, int, int, str] | None = None  # (score, is_winner, -order, idea)
    for order, (other, idea, is_winner) in enumerate(named_looks(critic_records)):
        s = similarity(look, other)
        key = (s, int(is_winner), -order, idea)
        if best is None or key[:3] > best[:3]:
            best = key
    if best is None:
        return None, 0.0
    return best[3], best[0]


def nearest_name(look: Sequence[str], critic_records: Iterable[dict],
                 threshold: float = THRESHOLD) -> str | None:
    """The critic's read for `look`: the idea of the nearest judged look by slot-weighted
    overlap, or None (untitled) when that overlap is below `threshold` (spec §3.2)."""
    name, score = nearest(look, critic_records)
    if name is None or score < threshold:
        return None
    return name


def load_critic_records(directory: Path) -> list[dict]:
    """Every record in `directory`'s `<round>.jsonl` files (sorted by name; QC json files
    are not rounds). A missing directory is an empty critic."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out
