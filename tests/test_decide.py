"""Candidates and the decision (spec §4.1 step 5, §2 Decision score)."""

from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pytest

from lfg_fly.body.candidates import Candidate, apply_change, candidates, changes_between
from lfg_fly.body.decide import (
    EQUIP_COST_BRIX,
    choose,
    considered_of,
    plan_day,
    score,
)
from lfg_fly.brain.senses import SLOTS

HERO = ("Blue", "None", "male", "Hoodie", "Smile", "Flat", "Open", "Cap", "None")
HEAD = SLOTS.index("Head")
BG = SLOTS.index("Background")


@dataclass(frozen=True)
class Cfg:
    beta: float = 0.3
    lam: float = 0.05
    temperature: float = 0.15
    max_steps: int = 3


def date_seed(date: str, version: str) -> int:
    digest = hashlib.sha256(f"{date}|{version}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def run(coro):
    return asyncio.run(coro)


# --- candidates -------------------------------------------------------------


def test_apply_change_replaces_one_slot():
    look = apply_change(HERO, "Head", "Beanie")
    assert look[HEAD] == "Beanie"
    assert look[:HEAD] == HERO[:HEAD] and look[HEAD + 1 :] == HERO[HEAD + 1 :]


def test_changes_between_is_the_slot_diff_in_slot_order():
    after = apply_change(apply_change(HERO, "Head", "Beanie"), "Background", "Red")
    assert changes_between(HERO, after) == (("Background", "Red"), ("Head", "Beanie"))
    assert changes_between(HERO, HERO) == ()


def test_candidates_stay_first_then_every_legal_change():
    legal = [("Background", "Red"), ("Head", "Beanie")]
    cands = candidates(HERO, legal, set())
    assert [c.changes for c in cands] == [(), (("Background", "Red"),), (("Head", "Beanie"),)]
    assert cands[0].look == HERO
    assert cands[1].look == apply_change(HERO, "Background", "Red")
    assert cands[2].look == apply_change(HERO, "Head", "Beanie")
    assert all(math.isnan(c.taste) and math.isnan(c.score) for c in cands)
    assert all(c.cost == 0.0 and c.considered for c in cands)


def test_tabu_excludes_a_look_but_never_stay():
    legal = [("Background", "Red"), ("Head", "Beanie")]
    tabu = {apply_change(HERO, "Head", "Beanie"), HERO}  # today's look is always "recent"
    cands = candidates(HERO, legal, tabu)
    assert [c.changes for c in cands] == [(), (("Background", "Red"),)]


def test_candidates_accepts_a_list_look_and_rejects_wrong_length():
    cands = candidates(list(HERO), [("Head", "Beanie")], set())
    assert cands[0].look == HERO and isinstance(cands[0].look, tuple)
    with pytest.raises(ValueError):
        candidates(HERO[:8], [], set())


# --- score / choose -----------------------------------------------------------


def test_score_formula():
    # taste + beta·taste_sd·rarity − lam·cost
    assert score(1.0, 2.0, 10.0, beta=0.3, lam=0.05, taste_sd=0.5) == pytest.approx(
        1.0 + 0.3 * 0.5 * 2.0 - 0.05 * 10.0
    )
    assert EQUIP_COST_BRIX == 0.0


def scored(*scores: float) -> list[Candidate]:
    out = []
    for i, s in enumerate(scores):
        changes = () if i == 0 else (("Head", f"H{i}"),)
        look = HERO if i == 0 else apply_change(HERO, "Head", f"H{i}")
        out.append(Candidate(look=look, changes=changes, score=s))
    return out


def test_temperature_zero_is_argmax():
    cands = scored(0.1, 0.9, 0.5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert choose(cands, 0.0, rng) is cands[1]


def test_temperature_zero_ties_go_to_the_first():
    cands = scored(0.9, 0.9, 0.1)
    assert choose(cands, 0.0, np.random.default_rng(0)) is cands[0]


def test_choose_is_reproducible_under_the_same_seed_and_samples_the_softmax():
    cands = scored(0.0, 0.05, 0.1)
    seed = date_seed("2026-09-25", "fly-v1")
    a = [choose(cands, 1.0, np.random.default_rng(seed)) for _ in range(3)]
    assert a[0] is a[1] is a[2]
    picks = {id(choose(cands, 1.0, np.random.default_rng(i))) for i in range(200)}
    assert len(picks) == 3  # at a warm temperature every candidate gets sampled


def test_choose_skips_unscored_and_unconsidered_candidates():
    cands = scored(0.1, 0.9, 0.5)
    cands[1].considered = False
    assert choose(cands, 0.0, np.random.default_rng(0)) is cands[2]
    cands[2].score = math.nan
    assert choose(cands, 0.0, np.random.default_rng(0)) is cands[0]
    with pytest.raises(ValueError):
        choose([Candidate(look=HERO, changes=())], 0.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        choose(cands, -1.0, np.random.default_rng(0))


# --- plan_day -----------------------------------------------------------------

LEGAL = [("Background", "good1"), ("Head", "good2"), ("Accessory", "bad"), ("Eyes", "good3")]


def taste_of(look) -> float:
    return sum(1.0 for v in look if v.startswith("good")) - 0.5 * sum(
        1.0 for v in look if v == "bad"
    )


class Brain:
    """Records every batch; taste counts 'good' values, rarity is 0."""

    def __init__(self, async_: bool):
        self.async_ = async_
        self.taste_calls: list[list[tuple]] = []
        self.rarity_calls: list[list[tuple]] = []

    def taste(self, looks):
        self.taste_calls.append(list(looks))
        out = np.array([taste_of(look) for look in looks])
        return self._wrap(out)

    def rarity(self, looks):
        self.rarity_calls.append(list(looks))
        return self._wrap(np.zeros(len(looks)))

    def _wrap(self, out):
        if not self.async_:
            return out

        async def coro():
            return out

        return coro()


def legal_fn(look):
    return [(s, v) for s, v in LEGAL if look[SLOTS.index(s)] != v]


def test_greedy_takes_up_to_max_steps_and_batches_once_per_step():
    brain = Brain(async_=False)
    final, chain, steps = run(
        plan_day(
            HERO,
            legal_fn=legal_fn,
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
            max_steps=3,
        )
    )
    assert len(chain) == 3 and len(steps) == 3
    assert taste_of(final) == 3.0
    assert final == chain[-1].look
    # each step: one taste batch and one rarity batch over every candidate at that step
    assert len(brain.taste_calls) == 3 and len(brain.rarity_calls) == 3
    for step, batch in zip(steps, brain.taste_calls, strict=True):
        assert batch == [c.look for c in step]
    # step 1 sees stay + 4 legal changes; every candidate is scored
    assert len(steps[0]) == 5 and all(np.isfinite(c.score) for c in steps[0])
    assert steps[0][0].changes == ()
    # the chain's changes are one slot per step, and the flattened diff reaches the final look
    assert all(len(c.changes) == 1 for c in chain)
    assert changes_between(HERO, final) == tuple(sorted(c for cand in chain for c in cand.changes))


def test_greedy_stops_when_stay_wins():
    # Only one improving change exists; step 2's best is stay → stop after one step.
    async def one_good(look):
        pairs = [("Head", "good2"), ("Accessory", "bad")]
        return [(s, v) for s, v in pairs if look[SLOTS.index(s)] != v]

    brain = Brain(async_=True)
    final, chain, steps = run(
        plan_day(
            HERO,
            legal_fn=one_good,
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
            max_steps=3,
        )
    )
    assert [c.changes for c in chain] == [(("Head", "good2"),)]
    assert len(steps) == 2
    assert steps[1][0].changes == () and final == apply_change(HERO, "Head", "good2")
    # the stopping step was still fully scored and recorded
    assert all(np.isfinite(c.score) for c in steps[1])


def test_stay_wins_immediately_means_no_change():
    brain = Brain(async_=False)
    final, chain, steps = run(
        plan_day(
            HERO,
            legal_fn=lambda look: [("Accessory", "bad")],
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
            max_steps=3,
        )
    )
    assert final == HERO and chain == [] and len(steps) == 1


def test_max_steps_defaults_to_cfg():
    brain = Brain(async_=False)
    _, chain, _ = run(
        plan_day(
            HERO,
            legal_fn=legal_fn,
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0, max_steps=2),
            rng=np.random.default_rng(0),
        )
    )
    assert len(chain) == 2


def test_date_seed_makes_the_choice_reproducible():
    # Warm temperature so the sampling actually matters; the same seed → the same day.
    def plan(seed: int):
        brain = Brain(async_=False)
        return run(
            plan_day(
                HERO,
                legal_fn=legal_fn,
                brain_taste=brain.taste,
                brain_rarity=brain.rarity,
                cost_fn=None,
                tabu=set(),
                cfg=Cfg(temperature=1.0),
                rng=np.random.default_rng(seed),
                max_steps=3,
            )
        )

    seed = date_seed("2026-09-25", "fly-v1")
    a, b = plan(seed), plan(seed)
    assert a[0] == b[0]
    assert [c.changes for c in a[1]] == [c.changes for c in b[1]]
    assert [[c.score for c in s] for s in a[2]] == [[c.score for c in s] for s in b[2]]
    # A different date usually plans a different day (checked over several dates).
    others = {plan(date_seed(f"2026-10-{d:02d}", "fly-v1"))[0] for d in range(1, 15)}
    assert len(others) > 1


def test_tabu_excludes_a_complete_look_in_plan_day_but_never_stay():
    # Head=good2 is the best single change but it was worn recently → the fly cannot re-wear it.
    brain = Brain(async_=False)
    tabu = {apply_change(HERO, "Head", "good2"), HERO}
    final, chain, steps = run(
        plan_day(
            HERO,
            legal_fn=lambda look: [
                (s, v)
                for s, v in [("Head", "good2"), ("Accessory", "bad")]
                if look[SLOTS.index(s)] != v
            ],
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=tabu,
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
        )
    )
    assert [c.changes for c in steps[0]] == [(), (("Accessory", "bad"),)]
    assert final == HERO and chain == []


def test_plan_day_never_revisits_a_look_from_earlier_today():
    # Step 1 takes Head=good2. At step 2 "Head=Cap" (back to the start) is legal by the
    # Closet but would undo the day; it must not be offered.
    def legal(look):
        out = []
        if look[HEAD] != "good2":
            out.append(("Head", "good2"))
        if look[HEAD] != "Cap":
            out.append(("Head", "Cap"))
        out.append(("Accessory", "bad"))
        return out

    brain = Brain(async_=False)
    _, chain, steps = run(
        plan_day(
            HERO,
            legal_fn=legal,
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
        )
    )
    assert [c.changes for c in chain] == [(("Head", "good2"),)]
    assert [c.changes for c in steps[1]] == [(), (("Accessory", "bad"),)]


def test_rarity_greed_and_cost_enter_the_score():
    # Two neutral-taste changes; one is rarer, the other costs BRIX. β·taste_sd·rarity − λ·cost.
    def taste(looks):
        return np.zeros(len(looks))

    def rarity(looks):
        return np.array([2.0 if look[HEAD] == "rare" else 0.0 for look in looks])

    def cost(cand: Candidate) -> float:
        return 10.0 if cand.changes and cand.changes[0][1] == "pricey" else 0.0

    _, chain, steps = run(
        plan_day(
            HERO,
            legal_fn=lambda look: [("Head", "rare"), ("Head", "pricey")]
            if look[HEAD] == "Cap"
            else [],
            brain_taste=taste,
            brain_rarity=rarity,
            cost_fn=cost,
            tabu=set(),
            cfg=Cfg(beta=0.3, lam=0.05, temperature=0.0),
            rng=np.random.default_rng(0),
            taste_sd=0.5,
        )
    )
    stay, rare, pricey = steps[0]
    assert stay.score == pytest.approx(0.0)
    assert rare.score == pytest.approx(0.3 * 0.5 * 2.0) and rare.rarity == 2.0
    assert pricey.score == pytest.approx(-0.05 * 10.0) and pricey.cost == 10.0
    assert chain[0] is rare


def test_default_cost_is_the_equip_constant():
    brain = Brain(async_=False)
    _, _, steps = run(
        plan_day(
            HERO,
            legal_fn=lambda look: [("Head", "good2")] if look[HEAD] == "Cap" else [],
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
        )
    )
    assert steps[0][1].cost == EQUIP_COST_BRIX == 0.0 and steps[0][0].cost == 0.0


def test_cap_is_disclosed_as_considered_n_of_m():
    brain = Brain(async_=False)
    _, _, steps = run(
        plan_day(
            HERO,
            legal_fn=legal_fn,
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
            max_steps=1,
            cap=3,
        )
    )
    step = steps[0]
    assert considered_of(step) == (3, 5)
    assert step[0].considered  # stay is always simulated
    assert len(brain.taste_calls[0]) == 3
    assert all(np.isfinite(c.score) for c in step if c.considered)
    assert all(math.isnan(c.score) for c in step if not c.considered)
    # no cap → everything is simulated
    brain = Brain(async_=False)
    _, _, steps = run(
        plan_day(
            HERO,
            legal_fn=legal_fn,
            brain_taste=brain.taste,
            brain_rarity=brain.rarity,
            cost_fn=None,
            tabu=set(),
            cfg=Cfg(temperature=0.0),
            rng=np.random.default_rng(0),
            max_steps=1,
        )
    )
    assert considered_of(steps[0]) == (5, 5)


def test_brain_batch_length_mismatch_is_an_error():
    def bad_taste(looks):
        return np.zeros(len(looks) - 1)

    with pytest.raises(ValueError):
        run(
            plan_day(
                HERO,
                legal_fn=legal_fn,
                brain_taste=bad_taste,
                brain_rarity=lambda looks: np.zeros(len(looks)),
                cost_fn=None,
                tabu=set(),
                cfg=Cfg(),
                rng=np.random.default_rng(0),
            )
        )
