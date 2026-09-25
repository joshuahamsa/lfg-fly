"""The day's candidates: every legal single-slot change, plus stay (spec §4.1 step 4).

A disclosed non-fly component. The brain scores; this module only enumerates. The
one assist that lives here is the 30-day tabu (§4.1 step 5): a complete look worn in
the last 30 days is removed from the candidates, but *stay* never is, since today's
starting look is by definition recent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import nan

from lfg_fly.brain.senses import SLOTS

Look = tuple[str, ...]
Change = tuple[str, str]

_SLOT_INDEX = {slot: i for i, slot in enumerate(SLOTS)}


@dataclass
class Candidate:
    """One possible look for today and how it was scored (spec §4.1 steps 4–5).

    ``changes`` is relative to the look the step started from: ``()`` for stay, one
    ``(slot, value)`` for a single-slot change. ``considered`` is False for a candidate
    that a cap kept out of the simulation batch ("considered N of M").
    """

    look: Look
    changes: tuple[Change, ...]
    taste: float = nan
    rarity: float = nan
    cost: float = 0.0
    score: float = nan
    considered: bool = True


def as_look(look) -> Look:
    look = tuple(look)
    if len(look) != len(SLOTS):
        raise ValueError(f"a look has {len(SLOTS)} slots, got {len(look)}")
    return look


def apply_change(look: Look, slot: str, value: str) -> Look:
    """``look`` with ``slot`` set to ``value``."""
    look = as_look(look)
    i = _SLOT_INDEX[slot]
    return look[:i] + (value,) + look[i + 1 :]


def changes_between(before: Look, after: Look) -> tuple[Change, ...]:
    """The slot diff from ``before`` to ``after``, in slot order: the body of one
    ``POST /api/equip`` (no duplicate slots, whatever path the greedy steps took)."""
    before, after = as_look(before), as_look(after)
    pairs = enumerate(zip(before, after, strict=True))
    return tuple((SLOTS[i], v) for i, (u, v) in pairs if u != v)


def candidates(hero: Look, legal: list[Change], tabu: set[Look]) -> list[Candidate]:
    """Stay first, then every legal single-slot change from ``hero`` (spec §4.1 step 4).

    Looks in ``tabu`` (complete looks from the last 30 days, §4.1 step 5) are dropped;
    stay never is. ``legal`` is kept in the order given (``legal_changes`` already
    orders it by slot then value, which keeps the seeded sampling reproducible).
    """
    hero = as_look(hero)
    out = [Candidate(look=hero, changes=())]
    for slot, value in legal:
        look = apply_change(hero, slot, value)
        if look in tabu:
            continue
        out.append(Candidate(look=look, changes=((slot, value),)))
    return out
