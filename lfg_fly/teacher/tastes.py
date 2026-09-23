"""The Phase 0b planted tastes (spec §3.0b): additive, latent harmony, visual.

Every taste relabels a Phase 0 probe set: the looks, pairs, families and split
stay exactly as they were, and only p and y change. So one simulation of a
setting scores every taste. Each taste draws its additive weights, latent
vectors and label noise from its own seed offset; `additive` at offset 0 is
Phase 0's taste, label for label.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.probe import one_hot
from lfg_fly.teacher.sample import Look, PlantedTaste, ProbeSet, plant_taste

TASTES = ("additive", "latent", "visual")
TASTE_OFFSETS = {"additive": 0, "latent": 100, "visual": 200}
LATENT_DIM = 3


@dataclass(frozen=True)
class Tasted:
    """A probe set relabelled with one taste, and that taste's utility per look."""
    name: str
    ps: ProbeSet
    U: np.ndarray


def additive_utility(looks: list[Look], cat: Catalog, seed: int) -> np.ndarray:
    taste = plant_taste(cat, seed)
    return np.array([taste.utility(look) for look in looks])


def latent_vectors(cat: Catalog, seed: int, dim: int = LATENT_DIM) -> dict[tuple[str, str],
                                                                          np.ndarray]:
    """z ~ N(0, I) for every trait value, centred per slot over the slot's catalog values,
    so a slot's mean vector adds no additive leak to the harmony term."""
    rng = np.random.default_rng(seed)
    out = {}
    for slot in SLOTS:
        vecs = rng.normal(0.0, 1.0, (len(cat.values[slot]), dim))
        vecs -= vecs.mean(0)
        out.update({(slot, v): vecs[i] for i, v in enumerate(cat.values[slot])})
    return out


def latent_harmony(looks: list[Look], z: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
    """Σ over slot pairs i < j of <z_i, z_j>, i.e. (|Σz|² − Σ|z_i|²) / 2."""
    out = np.empty(len(looks))
    for n, look in enumerate(looks):
        vecs = np.array([z[(s, v)] for s, v in zip(SLOTS, look, strict=True)])
        total = vecs.sum(0)
        out[n] = 0.5 * (float(total @ total) - float((vecs * vecs).sum()))
    return out


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def visual_interaction(terms: dict[str, np.ndarray]) -> np.ndarray:
    """Figure-ground contrast plus hue harmony, each standardized over the looks."""
    return _z(np.asarray(terms["contrast"], float)) + _z(np.asarray(terms["harmony"], float))


def mix(A: np.ndarray, interaction: np.ndarray) -> np.ndarray:
    """(A + I)/√2 with I centred and scaled to A's spread: half the variance each."""
    return (A + _z(interaction) * A.std()) / np.sqrt(2.0)


def relabel(ps: ProbeSet, U: np.ndarray, label_seed: int,
            k: float = PlantedTaste.k) -> ProbeSet:
    """The same looks, pairs, families and split, labelled by utility U (one per look).

    The label draw is the first use of `label_seed`'s generator, exactly as in
    `make_probe_set`, so Phase 0's taste reproduces Phase 0's labels."""
    p = 1.0 / (1.0 + np.exp(-k * (U[ps.a_idx] - U[ps.b_idx])))
    y = (np.random.default_rng(label_seed).random(len(p)) < p).astype(np.float32)
    return dataclasses.replace(ps, p=p, y=y)


def taste_set(ps: ProbeSet, cat: Catalog, taste: str, seed: int,
              visual_terms: dict[str, np.ndarray] | None = None) -> Tasted:
    """`ps` (drawn from `seed`) relabelled with `taste`. `visual` needs the per-look
    pixel terms (`pixels.visual_terms`), in `ps.looks` order."""
    if taste not in TASTE_OFFSETS:
        raise ValueError(f"unknown taste {taste!r}; expected one of {TASTES}")
    off = seed + TASTE_OFFSETS[taste]
    A = additive_utility(ps.looks, cat, off + 1)
    if taste == "additive":
        U = A
    elif taste == "latent":
        U = mix(A, latent_harmony(ps.looks, latent_vectors(cat, off + 3)))
    else:
        if visual_terms is None:
            raise ValueError("the visual taste needs its pixel terms (contrast, harmony)")
        U = mix(A, visual_interaction(visual_terms))
    return Tasted(taste, relabel(ps, U, off + 2), U)


def additive_oracle(t: Tasted, cat: Catalog) -> float:
    """The best additive model's expected test accuracy: least squares of the true U on
    one-hot traits over every probe look, scored under p as the Bayes ceiling is."""
    X = np.hstack([one_hot(t.ps.looks, cat), np.ones((len(t.ps.looks), 1), np.float32)])
    coef, *_ = np.linalg.lstsq(X.astype(np.float64), t.U, rcond=None)
    fit = X @ coef
    te = t.ps.test
    d = fit[t.ps.a_idx[te]] - fit[t.ps.b_idx[te]]
    p = t.ps.p[te]
    return float(np.mean(np.where(np.abs(d) < 1e-9, 0.5, np.where(d > 0, p, 1.0 - p))))
