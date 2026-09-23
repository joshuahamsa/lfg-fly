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


def _xor_probe_set(n_looks=400, n_pairs=1200, seed=0):
    """Looks with two binary features; the taste is their XOR, which no linear
    readout of the two features can represent."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (n_looks, 2))
    X = np.hstack([bits, rng.normal(0, 0.01, (n_looks, 2))]).astype(np.float32)
    u = (bits[:, 0] ^ bits[:, 1]).astype(float) * 4.0
    a, b = rng.integers(0, n_looks, n_pairs), rng.integers(0, n_looks, n_pairs)
    p = 1.0 / (1.0 + np.exp(-(u[a] - u[b])))
    family = np.arange(n_pairs) // 3
    ps = T.ProbeSet(looks=[(str(i),) for i in range(n_looks)], a_idx=a, b_idx=b,
                    family=family, near=np.zeros(n_pairs, bool),
                    y=(rng.random(n_pairs) < p).astype(np.float32), p=p,
                    test=family >= int(0.8 * family.max()))
    return X, ps


def test_evaluate_keeps_per_pair_correctness_out_of_the_record():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=600, seed=0)
    res = P.evaluate(P.one_hot(ps.looks, cat), ps)
    assert res.correct.shape == (res.n_test,) and res.correct.dtype == bool
    assert res.correct.mean() == res.heldout
    assert "correct" not in res.as_dict()


def test_mlp_learns_an_interaction_that_a_linear_readout_cannot():
    X, ps = _xor_probe_set()
    linear = P.evaluate(X, ps)
    mlp = P.evaluate_mlp(X, ps, hidden=16, lambdas=(0.1, 1.0), seed=0, folds=3)
    ceiling = T.bayes_ceiling(ps.p[ps.test])  # ~0.74: equal-utility pairs are coin flips
    assert linear.heldout < 0.6
    assert mlp.heldout > ceiling - 0.05
    assert mlp.correct.shape == (mlp.n_test,)
    assert mlp.ci_lo < mlp.heldout < mlp.ci_hi


def test_mlp_is_deterministic_for_a_seed():
    X, ps = _xor_probe_set(n_pairs=300)
    a = P.evaluate_mlp(X, ps, hidden=8, lambdas=(1.0,), seed=3, folds=2)
    b = P.evaluate_mlp(X, ps, hidden=8, lambdas=(1.0,), seed=3, folds=2)
    assert np.array_equal(a.correct, b.correct)


def test_paired_diff_resamples_families_for_both_models_at_once():
    fam = np.repeat(np.arange(100), 3)
    same = np.random.default_rng(0).random(300) < 0.7
    d = P.paired_diff(same, same, fam, n_boot=300)
    assert d == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    better = same | (np.arange(300) % 2 == 0)
    d = P.paired_diff(better, same, fam, n_boot=300)
    assert d["diff"] == pytest.approx(better.mean() - same.mean())
    assert 0.0 < d["lo"] < d["diff"] < d["hi"]
