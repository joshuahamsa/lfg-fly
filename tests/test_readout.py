"""The heads (spec §2 Readout): Bradley–Terry taste on standardized features with the
critic's strength as the sample weight, ridge rarity, concentrations, save/load."""

import numpy as np
import pytest

from lfg_fly.brain import readout as R
from lfg_fly.brain.senses import C_REF, SLOTS
from lfg_fly.teacher import probe as P
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog


def _cat():
    values = {s: [f"{s}{i}" for i in range(3)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1"]
    return Catalog(body="male", values=values, odds={}, api="x")


def _planted(n_looks=300, d=10, n_pairs=1200, k=3.0, seed=0):
    """Looks with gaussian features, a planted linear taste and noisy pair labels."""
    rng = np.random.default_rng(seed)
    X = (rng.normal(size=(n_looks, d)) * rng.uniform(0.5, 4.0, d) + rng.normal(size=d))
    X = X.astype(np.float32)
    w_true = rng.normal(size=d)
    u = X @ w_true
    a = rng.integers(0, n_looks, n_pairs)
    b = (a + rng.integers(1, n_looks, n_pairs)) % n_looks
    du = u[a] - u[b]
    p = 1.0 / (1.0 + np.exp(-k * du / du.std()))
    y = (rng.random(n_pairs) < p).astype(np.float32)
    family = np.arange(n_pairs) // 3
    q1, q2 = np.quantile(np.abs(du), [1 / 3, 2 / 3])
    strength = 1 + (np.abs(du) > q1).astype(int) + (np.abs(du) > q2).astype(int)
    return X, w_true, a, b, y, family, strength.astype(np.float32)


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fit_taste_recovers_a_planted_readout():
    X, w_true, a, b, y, family, strength = _planted()
    head = R.fit_taste(X, a, b, y, strength, family, "cpu")
    assert isinstance(head, R.TasteHead)
    assert head.w.shape == (X.shape[1],) and head.mu.shape == head.sd.shape == (X.shape[1],)
    # the head lives in standardized coordinates: u = ((X-mu)/sd) @ (w_true * sd) + const
    assert _cos(head.w, w_true * head.sd) > 0.95
    s = head.score(X)
    assert s.shape == (len(X),)
    assert np.corrcoef(s, X @ w_true)[0, 1] > 0.95
    # fresh pairs: the head orders them like the planted taste
    rng = np.random.default_rng(9)
    a2, b2 = rng.integers(0, len(X), 500), rng.integers(0, len(X), 500)
    keep = a2 != b2
    agree = ((s[a2] - s[b2] > 0) == ((X @ w_true)[a2] - (X @ w_true)[b2] > 0))[keep]
    assert agree.mean() > 0.9
    assert head.lam in P.LAMBDAS
    # taste_sd is the standard deviation of the score over the fitted looks (§2 Decision score)
    rows = np.unique(np.concatenate([a, b]))
    assert head.taste_sd == pytest.approx(float(s[rows].std()), rel=1e-5)
    assert head.taste_sd > 0


def test_fit_taste_uses_the_critics_strength_as_sample_weight():
    """Two halves of the pairs are labelled by opposite tastes. With the critic's strength
    3 on one half and ~0 on the other, the head follows the heavy half; with flat weights the
    halves cancel."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6)).astype(np.float32)
    w_true = rng.normal(size=6)
    u = X @ w_true
    a, b = rng.integers(0, 200, 900), rng.integers(0, 200, 900)
    keep = a != b
    a, b = a[keep], b[keep]
    y = (u[a] > u[b]).astype(np.float32)
    half = np.arange(len(a)) % 2 == 0
    y[~half] = 1.0 - y[~half]
    family = np.arange(len(a)) // 3
    weight = np.where(half, 3.0, 0.01).astype(np.float32)
    heavy = R.fit_taste(X, a, b, y, weight, family, "cpu")
    flat = R.fit_taste(X, a, b, y, np.ones(len(a), np.float32), family, "cpu")
    assert _cos(heavy.w, w_true * heavy.sd) > 0.9
    assert abs(_cos(flat.w, w_true * flat.sd)) < 0.6 or np.linalg.norm(flat.w) < 0.1 * (
        np.linalg.norm(heavy.w))


def test_fit_taste_takes_weight_none_as_flat():
    X, _, a, b, y, family, _ = _planted(n_pairs=600)
    ones = R.fit_taste(X, a, b, y, np.ones(len(a), np.float32), family, "cpu")
    none = R.fit_taste(X, a, b, y, None, family, "cpu")
    np.testing.assert_allclose(none.w, ones.w, atol=1e-5)
    assert none.lam == ones.lam


def test_probe_evaluate_accepts_weights_and_flat_weights_change_nothing():
    cat = _cat()
    ps = T.make_probe_set(cat, n_pairs=600, seed=0)
    X = P.one_hot(ps.looks, cat)
    plain = P.evaluate(X, ps, device="cpu")
    flat = P.evaluate(X, ps, device="cpu", w=np.full(len(ps.y), 2.0, np.float32))
    assert flat.heldout == plain.heldout and flat.lam == plain.lam
    assert np.array_equal(flat.correct, plain.correct)


def _flipped_probe_set(seed=0, n_looks=200, d=6, n_pairs=900, flip_frac=0.6):
    """Training pairs: `flip_frac` of them carry the OPPOSITE label at weight 0.01, the rest
    the true label at weight 3. Test pairs (their own families) are all true."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_looks, d)).astype(np.float32)
    u = X @ rng.normal(size=d)
    a, b = rng.integers(0, n_looks, n_pairs), rng.integers(0, n_looks, n_pairs)
    b = np.where(a == b, (b + 1) % n_looks, b)
    y = (u[a] > u[b]).astype(np.float32)
    family = np.arange(n_pairs) // 3
    test = family >= int(0.7 * family.max())
    flipped = (~test) & (rng.random(n_pairs) < flip_frac)
    y[flipped] = 1.0 - y[flipped]
    w = np.where(flipped, 0.01, 3.0).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(u[a] - u[b])))
    ps = T.ProbeSet(looks=[(str(i),) for i in range(n_looks)], a_idx=a, b_idx=b,
                    family=family, near=np.zeros(n_pairs, bool), y=y, p=p, test=test)
    return X, ps, w


def test_probe_evaluate_weights_let_the_true_labels_win():
    X, ps, w = _flipped_probe_set()
    weighted = P.evaluate(X, ps, device="cpu", w=w)
    unweighted = P.evaluate(X, ps, device="cpu")
    assert weighted.heldout > 0.85
    assert unweighted.heldout < 0.5  # the flipped majority taught it the opposite taste


def test_probe_evaluate_mlp_weights_let_the_true_labels_win():
    X, ps, w = _flipped_probe_set(n_pairs=600)
    weighted = P.evaluate_mlp(X, ps, device="cpu", w=w, hidden=8, lambdas=(1.0,), folds=2)
    unweighted = P.evaluate_mlp(X, ps, device="cpu", hidden=8, lambdas=(1.0,), folds=2)
    assert weighted.heldout > 0.8
    assert unweighted.heldout < 0.5


def test_fit_rarity_recovers_a_planted_linear_map():
    rng = np.random.default_rng(2)
    X = (rng.normal(size=(200, 8)) * rng.uniform(0.5, 3.0, 8) + 2.0).astype(np.float32)
    v = rng.normal(size=8)
    target = X @ v + 3.0
    head = R.fit_rarity(X, target, "snap1", lam=1e-3)
    assert isinstance(head, R.RarityHead)
    assert head.snapshot_hash == "snap1"
    assert head.target_mu == pytest.approx(float(target.mean()), rel=1e-6)
    assert head.target_sd == pytest.approx(float(target.std()), rel=1e-6)
    z = (target - target.mean()) / target.std()
    np.testing.assert_allclose(head.score(X), z, atol=1e-3)
    X2 = (rng.normal(size=(50, 8)) * 2.0 + 1.0).astype(np.float32)
    z2 = (X2 @ v + 3.0 - target.mean()) / target.std()
    np.testing.assert_allclose(head.score(X2), z2, atol=1e-2)
    assert head.score(X2).shape == (50,)


def test_fit_rarity_ridge_shrinks():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    target = X @ rng.normal(size=5)
    small = R.fit_rarity(X, target, "s", lam=1e-3)
    big = R.fit_rarity(X, target, "s", lam=1e4)
    assert np.linalg.norm(big.w) < 0.1 * np.linalg.norm(small.w)


SUPPLY = {"network": "mainnet", "as_of": 1, "n_live": 100,
          "counts": {"Head": {"Head0": 50, "Head1": 30, "Head2": 20},
                     "Eyes": {"Eyes0": 100},
                     "Body": {"B0": 60, "B1": 40}}}


def _look(**kw):
    return tuple(kw.get(s, f"{s}0" if s != "Body" else "B0") for s in SLOTS)


def test_concentrations_follow_the_rarity_percentile_within_the_slot():
    cat = _cat()
    looks = [_look(Head="Head0"), _look(Head="Head1"), _look(Head="Head2"), _look(Head="None")]
    c = R.concentrations(looks, SUPPLY, cat)
    assert c.shape == (4, len(SLOTS)) and c.dtype == np.float32
    h = SLOTS.index("Head")
    # Head: 4 catalog values, rarities 2 < 3.3 < 5 < inf (None has no live supply -> p = 1)
    np.testing.assert_allclose(c[:, h], [0.4 + 0.6 * 0.25, 0.4 + 0.6 * 0.5,
                                         0.4 + 0.6 * 0.75, 1.0], atol=1e-6)
    assert np.all(c >= 0.4 - 1e-6) and np.all(c <= 1.0 + 1e-6)
    # Eyes: Eyes0 is worn by every live NFT, the other three catalog values by none
    e = SLOTS.index("Eyes")
    assert c[0, e] == pytest.approx(0.4 + 0.6 * 0.25)
    assert R.concentrations([_look(Eyes="Eyes1")], SUPPLY, cat)[0, e] == pytest.approx(1.0)
    # a slot absent from the supply: every value has zero supply, p = 1 everywhere
    assert c[0, SLOTS.index("Mouth")] == pytest.approx(1.0)
    # Body: B1 (40) is rarer than B0 (60)
    assert c[0, SLOTS.index("Body")] == pytest.approx(0.4 + 0.6 * 0.5)
    assert R.concentrations([_look(Body="B1")], SUPPLY, cat)[0, SLOTS.index("Body")] == 1.0


def test_concentrations_rank_a_live_value_outside_the_catalog_too():
    cat = _cat()
    supply = {"n_live": 100, "counts": {"Head": {"Head0": 50, "Head1": 30, "Head2": 15,
                                                 "Zebra": 5}}}
    c = R.concentrations([_look(Head="Zebra"), _look(Head="Head2")], supply, cat)
    h = SLOTS.index("Head")
    # five values ranked: Head0 < Head1 < Head2 < Zebra < None(0 supply)
    assert c[1, h] == pytest.approx(0.4 + 0.6 * 3 / 5)
    assert c[0, h] == pytest.approx(0.4 + 0.6 * 4 / 5)


def test_c_ref_is_the_taste_pin():
    assert C_REF == 0.7
    # a flat c_ref row is what taste simulations use; concentrations never depend on it
    c = R.concentrations([_look()], SUPPLY, _cat())
    assert not np.allclose(c, C_REF)


def test_nft_rarity_sums_n_live_over_freq():
    look = _look(Head="Head1", Eyes="Eyes0", Body="B1")
    # Head1 100/30, Eyes0 100/100, Body B1 100/40, six slots with no supply -> n_live each
    expected = 100 / 30 + 1.0 + 100 / 40 + 6 * 100
    assert R.nft_rarity(look, SUPPLY) == pytest.approx(expected)
    assert R.nft_rarity(_look(Head="None"), SUPPLY) > R.nft_rarity(_look(Head="Head0"), SUPPLY)


def test_save_load_head_round_trip_taste_and_rarity(tmp_path):
    X, _, a, b, y, family, strength = _planted(n_pairs=600)
    taste = R.fit_taste(X, a, b, y, strength, family, "cpu")
    rarity = R.fit_rarity(X, X @ np.arange(X.shape[1]), "snapshotdeadbeef")
    for head, name in ((taste, "taste"), (rarity, "rarity")):
        # a directory ...
        d = tmp_path / name
        R.save_head(d, head)
        assert (d / "head.npz").exists() and (d / "head.json").exists()
        back = R.load_head(d)
        # ... or an .npz file with its .json beside it
        f = tmp_path / f"{name}-file.npz"
        R.save_head(f, head)
        assert f.exists() and f.with_suffix(".json").exists()
        back2 = R.load_head(f)
        for got in (back, back2):
            assert type(got) is type(head)
            np.testing.assert_array_equal(got.w, head.w)
            np.testing.assert_array_equal(got.mu, head.mu)
            np.testing.assert_array_equal(got.sd, head.sd)
            np.testing.assert_allclose(got.score(X), head.score(X), rtol=1e-6)
    t2 = R.load_head(tmp_path / "taste")
    assert t2.lam == taste.lam and t2.taste_sd == taste.taste_sd
    r2 = R.load_head(tmp_path / "rarity")
    assert r2.snapshot_hash == "snapshotdeadbeef"
    assert r2.b == rarity.b and r2.target_mu == rarity.target_mu
    assert r2.target_sd == rarity.target_sd


def test_load_head_refuses_a_missing_or_foreign_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        R.load_head(tmp_path / "nothing")
    d = tmp_path / "bad"
    d.mkdir()
    (d / "head.json").write_text('{"kind": "smell"}')
    np.savez(d / "head.npz", w=np.zeros(2))
    with pytest.raises(ValueError):
        R.load_head(d)
