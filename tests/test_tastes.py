import numpy as np
import pytest

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import sample as T
from lfg_fly.teacher import tastes as S
from lfg_fly.teacher.catalog import Catalog


def _cat():
    values = {s: [f"{s}{i}" for i in range(8)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1", "B2"]
    return Catalog(body="male", values=values, odds={}, api="x")


def _visual_terms(ps, seed=5):
    rng = np.random.default_rng(seed)
    n = len(ps.looks)
    return {"contrast": rng.random(n), "harmony": rng.normal(size=n)}


def test_additive_taste_reproduces_phase0_labels_exactly():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=600, seed=0)
    t = S.taste_set(ps, cat, "additive", seed=0)
    assert np.array_equal(t.ps.y, ps.y)
    assert np.allclose(t.ps.p, ps.p)
    conf = T.make_probe_set(cat, n_pairs=600, seed=1000)
    assert np.array_equal(S.taste_set(conf, cat, "additive", seed=1000).ps.y, conf.y)


@pytest.mark.parametrize("taste", ["latent", "visual"])
def test_relabel_keeps_looks_pairs_families_and_split(taste):
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=600, seed=0)
    t = S.taste_set(ps, cat, taste, seed=0, visual_terms=_visual_terms(ps))
    assert t.ps.looks is ps.looks
    for field in ("a_idx", "b_idx", "family", "near", "test"):
        assert np.array_equal(getattr(t.ps, field), getattr(ps, field))
    assert not np.array_equal(t.ps.y, ps.y)  # its own taste and its own label noise
    assert t.U.shape == (len(ps.looks),)


def test_latent_vectors_are_centred_per_slot():
    z = S.latent_vectors(_cat(), seed=3)
    for slot in SLOTS:
        vecs = np.array([v for (s, _), v in z.items() if s == slot])
        assert vecs.shape[1] == 3
        assert np.allclose(vecs.mean(0), 0.0, atol=1e-9)


def test_latent_harmony_sums_dot_products_over_slot_pairs():
    cat = _cat()
    z = S.latent_vectors(cat, seed=3)
    look = tuple(cat.values[s][1] for s in SLOTS)
    vecs = [z[(s, v)] for s, v in zip(SLOTS, look, strict=True)]
    expected = sum(float(vecs[i] @ vecs[j]) for i in range(len(SLOTS))
                   for j in range(i + 1, len(SLOTS)))
    assert S.latent_harmony([look], z)[0] == pytest.approx(expected)


def test_mix_scales_the_interaction_to_the_additive_spread():
    rng = np.random.default_rng(0)
    A, interaction = rng.normal(0, 3, 500), rng.normal(10, 0.2, 500)
    U = S.mix(A, interaction)
    scaled = U * np.sqrt(2) - A
    assert scaled.mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled.std() == pytest.approx(A.std())


def test_visual_interaction_standardizes_each_term():
    v = S.visual_interaction({"contrast": np.array([0.0, 1.0, 2.0]),
                              "harmony": np.array([10.0, 10.0, 16.0])})
    zc = np.array([-1.0, 0.0, 1.0]) / np.std([-1.0, 0.0, 1.0])
    zh = (np.array([10.0, 10.0, 16.0]) - 12.0) / np.std([10.0, 10.0, 16.0])
    assert np.allclose(v, zc + zh)


def test_visual_taste_needs_its_pixel_terms():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=90, seed=0)
    with pytest.raises(ValueError, match="visual"):
        S.taste_set(ps, cat, "visual", seed=0)
    with pytest.raises(ValueError, match="unknown taste"):
        S.taste_set(ps, cat, "salty", seed=0)


def test_additive_oracle_reaches_the_ceiling_only_for_the_additive_taste():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=3000, seed=0)
    add = S.taste_set(ps, cat, "additive", seed=0)
    assert S.additive_oracle(add, cat) == pytest.approx(T.bayes_ceiling(add.ps.p[add.ps.test]))
    lat = S.taste_set(ps, cat, "latent", seed=0)
    assert S.additive_oracle(lat, cat) < T.bayes_ceiling(lat.ps.p[lat.ps.test]) - 0.03


def test_each_seed_draws_a_fresh_taste():
    cat = _cat()
    sel = T.make_probe_set(cat, n_pairs=300, seed=0)
    a = S.taste_set(sel, cat, "latent", seed=0)
    b = S.taste_set(sel, cat, "latent", seed=1000)
    assert not np.allclose(a.U, b.U)
