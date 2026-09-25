"""The decision (spec §4.1 step 5; §2 Decision score).

    score(look) = taste(look) + β · taste_sd · rarity(look) − λ · cost_BRIX(action)

Taste is the brain's ``c_ref`` simulation read by the taste head; rarity is a second,
live-concentration simulation read by the rarity head (standardized), so β — the
fly's disclosed greed — is in units of taste's standard deviation. λ is its price
sensitivity per BRIX. Equip costs 0 (``EQUIP_COST_BRIX``) because LFG publishes no
Builder price.

A day is up to ``max_steps`` greedy steps. Each step scores every candidate (one
taste batch, one rarity batch), samples one at temperature τ with the date-seeded
generator (``records.date_seed``), and stops early when *stay* wins.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

import numpy as np

from lfg_fly.body.candidates import Candidate, Change, Look, as_look, candidates

EQUIP_COST_BRIX = 0.0
"""Cost of one equip in BRIX (spec §2 Decision score): LFG publishes no Builder price."""

Scores = Callable[[list[Look]], Any]  # -> array-like of float, or an awaitable of one


def score(
    taste: float, rarity: float, cost: float, beta: float, lam: float, taste_sd: float
) -> float:
    """``taste + beta·taste_sd·rarity − lam·cost`` (spec §2 Decision score)."""
    return float(taste) + beta * taste_sd * float(rarity) - lam * float(cost)


def equip_cost(cand: Candidate) -> float:
    """The default cost: stay is free; any equip costs ``EQUIP_COST_BRIX``."""
    return EQUIP_COST_BRIX if cand.changes else 0.0


def considered_of(cands: Sequence[Candidate]) -> tuple[int, int]:
    """``(N, M)`` for "considered N of M" (spec §4.1 step 4)."""
    return sum(1 for c in cands if c.considered), len(cands)


def choose(cands: list[Candidate], temperature: float, rng: np.random.Generator) -> Candidate:
    """Sample one candidate ∝ ``exp(score / temperature)`` (spec §4.1 step 5).

    Temperature 0 is the argmax (ties → the first, which is stay when it ties).
    Candidates without a finite score, or not considered, are never chosen.
    """
    if temperature < 0:
        raise ValueError("temperature must be ≥ 0")
    pool = [c for c in cands if c.considered and np.isfinite(c.score)]
    if not pool:
        raise ValueError("no scored candidate to choose from")
    s = np.array([c.score for c in pool], dtype=np.float64)
    if temperature == 0:
        return pool[int(np.argmax(s))]
    z = s / temperature
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return pool[int(rng.choice(len(pool), p=p))]


async def _call(fn: Callable[..., Any], *args) -> Any:
    out = fn(*args)
    if inspect.isawaitable(out):
        out = await out
    return out


def _floats(values, n: int, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{what} returned {arr.shape[0]} scores for {n} looks")
    return arr


def _apply_cap(cands: list[Candidate], cap: int | None, rng: np.random.Generator) -> None:
    """Mark the candidates a cap keeps out of the batch; stay is always simulated."""
    if cap is None or cap >= len(cands):
        return
    if cap < 1:
        raise ValueError("cap must be ≥ 1")
    keep = set(rng.choice(np.arange(1, len(cands)), size=cap - 1, replace=False).tolist())
    for i, c in enumerate(cands[1:], start=1):
        c.considered = i in keep


async def plan_day(
    hero: Look,
    *,
    legal_fn: Callable[[Look], Awaitable[Iterable[Change]] | Iterable[Change]],
    brain_taste: Scores,
    brain_rarity: Scores,
    cost_fn: Callable[[Candidate], float] | None,
    tabu: set[Look],
    cfg,
    rng: np.random.Generator,
    max_steps: int | None = None,
    taste_sd: float = 1.0,
    cap: int | None = None,
) -> tuple[Look, list[Candidate], list[list[Candidate]]]:
    """Plan today's look: up to ``max_steps`` greedy steps, stopping when stay wins.

    Spec §4.1 step 5. Per step: ``legal_fn(look)`` (sync or async) lists the legal
    changes from the current look; ``candidates`` adds stay and applies the tabu (the
    30-day set plus every look already visited today, so a day never undoes itself);
    ``brain_taste`` and ``brain_rarity`` (sync or async) are each called once over
    every considered candidate's look, in one batch; ``cost_fn(candidate)`` (default
    ``equip_cost``) prices the action; ``choose`` samples at ``cfg.temperature`` from
    ``rng``, which the caller seeds with ``records.date_seed(date, version)``.

    ``taste_sd`` is the taste head's standard deviation on its training set (β is in
    those units); ``cap`` limits how many candidates are simulated per step (stay
    always is), disclosed through ``Candidate.considered`` / ``considered_of``.

    Returns ``(final look, chain, steps)``: the chosen non-stay candidate of each step
    in order, and every step's full candidate list (the last one is the step where stay
    won, when it did). The equip body is ``changes_between(hero, final)``.
    """
    look = as_look(hero)
    steps_left = cfg.max_steps if max_steps is None else max_steps
    cost_of = equip_cost if cost_fn is None else cost_fn
    visited: set[Look] = {look}
    chain: list[Candidate] = []
    steps: list[list[Candidate]] = []
    for _ in range(int(steps_left)):
        legal = [(str(s), str(v)) for s, v in await _call(legal_fn, look)]
        cands = candidates(look, legal, set(tabu) | visited)
        _apply_cap(cands, cap, rng)
        batch = [c for c in cands if c.considered]
        looks = [c.look for c in batch]
        taste = _floats(await _call(brain_taste, looks), len(looks), "brain_taste")
        rarity = _floats(await _call(brain_rarity, looks), len(looks), "brain_rarity")
        for c, t, r in zip(batch, taste, rarity, strict=True):
            c.taste, c.rarity = float(t), float(r)
            c.cost = float(cost_of(c))
            c.score = score(c.taste, c.rarity, c.cost, cfg.beta, cfg.lam, taste_sd)
        steps.append(cands)
        pick = choose(cands, cfg.temperature, rng)
        if not pick.changes:
            break
        chain.append(pick)
        look = pick.look
        visited.add(look)
    return look, chain, steps
