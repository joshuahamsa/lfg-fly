import numpy as np

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog


def _cat():
    values = {s: [f"{s}{i}" for i in range(6)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1"]
    odds = {s: {v: 1.0 for v in vals[:3]} for s, vals in values.items()}
    return Catalog(body="male", values=values, odds=odds, api="x")


def test_realistic_draws_only_odds_values():
    cat, rng = _cat(), np.random.default_rng(0)
    for _ in range(50):
        look = T.realistic_look(cat, rng)
        assert all(v in cat.odds[s] for s, v in zip(SLOTS, look, strict=True))


def test_near_variant_changes_1_or_2_non_body_slots():
    cat, rng = _cat(), np.random.default_rng(1)
    for _ in range(200):
        base = T.base_look(cat, rng)
        other = T.near_variant(base, cat, rng)
        diff = [s for s, x, y in zip(SLOTS, base, other, strict=True) if x != y]
        assert 1 <= len(diff) <= 2 and "Body" not in diff


def test_pairs_are_deterministic_and_grouped():
    cat = _cat()
    a = T.make_pairs(cat, n_families=20, seed=3)
    b = T.make_pairs(cat, n_families=20, seed=3)
    assert a == b and len(a) == 60
    assert sorted({p.family for p in a}) == list(range(20))
    near = [p for p in a if p.near]
    assert 0.3 < len(near) / len(a) < 0.9


def test_probe_set_splits_by_family_and_labels_follow_taste():
    ps = T.make_probe_set(_cat(), n_pairs=600, seed=4)
    test_fams, train_fams = set(ps.family[ps.test]), set(ps.family[~ps.test])
    assert not (test_fams & train_fams)
    assert 0.19 <= ps.test.mean() <= 0.21
    assert set(np.unique(ps.y)) <= {0.0, 1.0}
    assert 0.5 < T.bayes_ceiling(ps.p) <= 1.0
    # labels agree with the planted preference more often than not
    assert ((ps.p > 0.5) == (ps.y == 1)).mean() > 0.6
    assert len(ps.looks) == len(set(ps.looks))
