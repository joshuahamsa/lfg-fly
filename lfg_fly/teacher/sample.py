"""Looks, near/far pairs grouped into families, and the planted taste (spec §3.0, §3.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lfg_fly.brain.senses import NON_BODY_SLOTS, SLOTS
from lfg_fly.teacher.catalog import Catalog

Look = tuple[str, ...]


@dataclass(frozen=True)
class Pair:
    family: int
    a: Look
    b: Look
    near: bool


def _pick(values: list[str], rng) -> str:
    return str(values[int(rng.integers(len(values)))])


def realistic_look(cat: Catalog, rng) -> Look:
    out = []
    for slot in SLOTS:
        odds = {v: w for v, w in cat.odds.get(slot, {}).items() if v in cat.values[slot] and w > 0}
        if odds:
            vals = sorted(odds)
            p = np.array([odds[v] for v in vals], dtype=np.float64)
            out.append(str(vals[int(rng.choice(len(vals), p=p / p.sum()))]))
        else:
            out.append(_pick(cat.values[slot], rng))
    return tuple(out)


def uniform_look(cat: Catalog, rng) -> Look:
    return tuple(_pick(cat.values[s], rng) for s in SLOTS)


def hybrid_look(cat: Catalog, rng) -> Look:
    look = list(realistic_look(cat, rng))
    for slot in rng.choice(NON_BODY_SLOTS, size=3, replace=False):
        look[SLOTS.index(str(slot))] = _pick(cat.values[str(slot)], rng)
    return tuple(look)


def base_look(cat: Catalog, rng) -> Look:
    r = rng.random()
    if r < 0.4:
        return realistic_look(cat, rng)
    return uniform_look(cat, rng) if r < 0.7 else hybrid_look(cat, rng)


def near_variant(look: Look, cat: Catalog, rng, n_slots: int | None = None) -> Look:
    """Change 1-2 non-Body slots (the Builder never changes Body)."""
    candidates = [s for s in NON_BODY_SLOTS if len(cat.values[s]) > 1]
    k = n_slots if n_slots is not None else int(rng.integers(1, 3))
    out = list(look)
    for slot in rng.choice(candidates, size=min(k, len(candidates)), replace=False):
        i = SLOTS.index(str(slot))
        choices = [v for v in cat.values[str(slot)] if v != look[i]]
        out[i] = _pick(choices, rng)
    return tuple(out)


def make_pairs(cat: Catalog, n_families: int, pairs_per_family: int = 3, near_frac: float = 0.6,
               seed: int = 0) -> list[Pair]:
    rng = np.random.default_rng(seed)
    pairs = []
    for fam in range(n_families):
        base = base_look(cat, rng)
        for _ in range(pairs_per_family):
            near = bool(rng.random() < near_frac)
            other = near_variant(base, cat, rng) if near else base_look(cat, rng)
            a, b = (base, other) if rng.random() < 0.5 else (other, base)
            pairs.append(Pair(fam, a, b, near))
    return pairs


@dataclass(frozen=True)
class PlantedTaste:
    theta: dict[tuple[str, str], float]
    k: float = 1.2

    def utility(self, look: Look) -> float:
        return float(sum(self.theta[(s, v)] for s, v in zip(SLOTS, look, strict=True)))


def plant_taste(cat: Catalog, seed: int) -> PlantedTaste:
    rng = np.random.default_rng(seed)
    keys = [(s, v) for s in SLOTS for v in cat.values[s]]
    weights = rng.normal(0.0, 1.0, len(keys)).tolist()
    return PlantedTaste(theta=dict(zip(keys, weights, strict=True)))


@dataclass(frozen=True)
class ProbeSet:
    looks: list[Look]
    a_idx: np.ndarray
    b_idx: np.ndarray
    family: np.ndarray
    near: np.ndarray
    y: np.ndarray
    p: np.ndarray
    test: np.ndarray


def bayes_ceiling(p: np.ndarray) -> float:
    return float(np.mean(np.maximum(p, 1.0 - p)))


def make_probe_set(cat: Catalog, n_pairs: int = 3000, pairs_per_family: int = 3,
                   test_frac: float = 0.2, seed: int = 0) -> ProbeSet:
    pairs = make_pairs(cat, n_pairs // pairs_per_family, pairs_per_family, seed=seed)
    index: dict[Look, int] = {}
    for pr in pairs:
        for look in (pr.a, pr.b):
            index.setdefault(look, len(index))
    taste = plant_taste(cat, seed + 1)
    du = np.array([taste.utility(pr.a) - taste.utility(pr.b) for pr in pairs])
    p = 1.0 / (1.0 + np.exp(-taste.k * du))
    rng = np.random.default_rng(seed + 2)
    y = (rng.random(len(pairs)) < p).astype(np.float32)
    family = np.array([pr.family for pr in pairs])
    fams = rng.permutation(np.unique(family))
    test_fams = set(fams[: math.ceil(test_frac * len(fams))].tolist())
    return ProbeSet(
        looks=list(index),
        a_idx=np.array([index[pr.a] for pr in pairs]),
        b_idx=np.array([index[pr.b] for pr in pairs]),
        family=family,
        near=np.array([pr.near for pr in pairs]),
        y=y,
        p=p,
        test=np.isin(family, list(test_fams)),
    )
