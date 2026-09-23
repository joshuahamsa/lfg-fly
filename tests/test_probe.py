import numpy as np
import pytest

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import probe as P
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog


def _cat():
    values = {s: [f"{s}{i}" for i in range(8)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1", "B2"]
    return Catalog(body="male", values=values, odds={}, api="x")


def test_one_hot_learns_planted_taste_near_ceiling():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=1500, seed=0)
    res = P.evaluate(P.one_hot(ps.looks, cat), ps, device="cpu")
    assert res.heldout > 0.7
    assert res.ci_lo < res.heldout < res.ci_hi
    assert res.heldout <= T.bayes_ceiling(ps.p[ps.test]) + 0.05


def test_random_features_stay_near_chance():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=1500, seed=1)
    X = np.random.default_rng(0).normal(size=(len(ps.looks), 200)).astype(np.float32)
    res = P.evaluate(X, ps, device="cpu")
    assert abs(res.heldout - 0.5) < 0.08


def test_cv_folds_never_split_a_family():
    groups = np.repeat(np.arange(30), 3)
    folds = P._folds(groups, 5)
    for k in range(5):
        assert not (set(groups[folds == k]) & set(groups[folds != k]))


def test_family_bootstrap_brackets_mean():
    correct = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1] * 10, bool)
    fam = np.repeat(np.arange(50), 2)
    lo, hi = P.family_bootstrap(correct, fam, n_boot=500, seed=0)
    assert lo < correct.mean() < hi


def test_per_slot_decodability_on_one_hot_is_perfect():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=600, seed=2)
    X = P.one_hot(ps.looks, cat)
    train = np.random.default_rng(0).random(len(ps.looks)) < 0.8
    dec = P.per_slot_decodability(X, ps.looks, cat, train)
    assert dec["Head"]["acc"] > 0.95 and dec["Head"]["chance"] < 0.5


def test_checks():
    assert P.smooth_ok(1.0, 20.0, 30.0) and not P.smooth_ok(5.0, 20.0, 30.0)
    assert not P.smooth_ok(1.0, 30.0, 30.0)
    assert P.activity_ok(0.08, 0.06)
    assert not P.activity_ok(0.2, 0.5)
    assert not P.activity_ok(0.05, 0.01)
    assert P.readout_distance(np.ones((3, 4)), np.zeros((3, 4))) == pytest.approx(4.0)
