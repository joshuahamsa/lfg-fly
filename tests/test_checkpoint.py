"""The checkpoint (spec §2 Checkpoint and versioning) and FlyBrain, on a tiny graph, CPU."""

import io
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from lfg_fly import paths
from lfg_fly.brain import checkpoint as C
from lfg_fly.brain import readout as R
from lfg_fly.brain.senses import C_REF, SLOTS, InputBuilder, build_retina
from lfg_fly.brain.sim import BrainParams, Simulator
from lfg_fly.connectome import build as B
from lfg_fly.connectome.columns import Columns, save_columns
from lfg_fly.connectome.neurons import populations
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import render as RN
from lfg_fly.teacher.catalog import Catalog, layer_cache_path
from lfg_fly.teacher.sample import uniform_look

COMMIT = "0123456789abcdef0123456789abcdef01234567"
Z_CFG = {"layers": [{"name": s, "z": 10 * (i + 1)} for i, s in enumerate(SLOTS)],
         "z_overrides": []}


def _toy_graph(with_eyes: bool = False, seed: int = 0):
    """ORNs (and, with eyes, three photoreceptors) -> a middle layer -> readout."""
    rng = np.random.default_rng(seed)
    n_pr = 3 if with_eyes else 0
    n_orn, n_mid, n_ro = 24, 40, 30
    n = n_pr + n_orn + n_mid + n_ro
    sc = (["ol_sensory"] * n_pr + ["cb_sensory"] * n_orn + ["cb_intrinsic"] * n_mid
          + ["descending_neuron"] * n_ro)
    types = (["R1-R6", "R8y", "R7p"][:n_pr] + [f"ORN_G{i % 6}" for i in range(n_orn)]
             + ["x"] * (n_mid + n_ro))
    nts = (["histamine"] * n_pr + ["acetylcholine"] * (n - n_pr))
    pre, post = [], []
    for o in range(n_pr + n_orn):
        for m in rng.choice(n_mid, 6, replace=False):
            pre.append(o)
            post.append(n_pr + n_orn + m)
    for m in range(n_mid):
        for r in rng.choice(n_ro, 5, replace=False):
            pre.append(n_pr + n_orn + m)
            post.append(n_pr + n_orn + n_mid + r)
    edges = pd.DataFrame({"pre": pre, "post": post}).drop_duplicates()
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": sc, "type": types, "rootSide": ["L"] * n})
    nt = pd.DataFrame({"body": np.arange(1, n + 1), "consensus_nt": nts})
    g = B.build_graph(ann, nt, pd.DataFrame({"body_pre": edges.pre + 1,
                                             "body_post": edges.post + 1,
                                             "weight": 5}), min_syn=1)
    if with_eyes:
        cols = Columns(neuron=np.array([0, 1, 2]), eye=np.array(["L", "L", "L"]),
                       q=np.array([0, 4, 0], np.int32), r=np.array([0, 0, 4], np.int32),
                       share=np.ones(3, np.float32), dropped=0, total=3)
    else:
        cols = Columns(neuron=np.array([], np.int64), eye=np.array([], "<U1"),
                       q=np.array([], np.int32), r=np.array([], np.int32),
                       share=np.array([], np.float32), dropped=0, total=0)
    return g, cols


def _png(rgba):
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), rgba).save(buf, format="PNG")
    return buf.getvalue()


def _catalog(n_values: int = 3) -> Catalog:
    values = {s: [f"{s}{i}" for i in range(n_values)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0"]
    return Catalog(body="male", values=values, odds={}, api="http://lfg")


def _install(tmp_path, g, cols, cat, layers: bool = False):
    """Put the graph and columns where FlyBrain.load reads them (FLY_DATA_DIR is the autouse
    tmp dir) and the catalog (plus, optionally, layer PNGs and the pinned trait_config) in a
    catalog dir."""
    B.save_graph(g, paths.graph_dir() / "graph-syn1.npz")
    save_columns(cols, paths.graph_dir() / "columns-syn1.npz")
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    (catalog_dir / "catalog-male.json").write_text(cat.to_json())
    if layers:
        rng = np.random.default_rng(5)
        for slot in SLOTS:
            for value in cat.values[slot]:
                if value == "None":
                    continue
                p = layer_cache_path(catalog_dir, "male", slot, value)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(_png(tuple(int(x) for x in rng.integers(0, 256, 3)) + (200,)))
        (catalog_dir / f"trait_config-{COMMIT}.yaml").write_text(yaml.safe_dump(Z_CFG))
    return catalog_dir


def _setting(code="all-sensory-brain"):
    return G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.1, steps=8, burn_in=5),
                     code, 2.0)


def _manifest(g, cols, cat, setting, **over):
    fields = dict(
        version="fly-test", graph_hash=g.graph_hash(),
        feathers={"annotations": "a" * 64, "neurotransmitters": "b" * 64, "weights": "c" * 64},
        min_syn=1, sign_map=dict(B.SIGN), nt_missing=g.nt_missing,
        columns_hash=C.columns_hash(cols), setting=setting.as_dict(), c_ref=C_REF,
        codes={"odor": "v2", "allsens": "v1"}, lfg_trait_config_commit=COMMIT,
        catalog_hash=C.catalog_hash(cat),
        taste={"heldout": 0.76, "ci_lo": 0.72, "ci_hi": 0.79, "lam": 0.01, "n_train": 2400,
               "n_test": 600, "gate_pass": True},
        created_at="2026-09-25T00:00:00+00:00")
    fields.update(over)
    return C.Manifest(**fields)


def _features(g, cols, cat, setting, looks, conc=None, bank=None, zorder=None):
    pops = populations(g)
    ctx = G.Context(graph=g, pops=pops, retina=build_retina(cols, g), catalog=cat, bank=bank,
                    zorder=zorder, device="cpu", probe_set=None)
    sim = Simulator(g, pops.sensory_any, pops.readout, "cpu")
    return G.features_for(ctx, sim, setting, looks, {}, conc=conc)


def _planted_head(X, seed=0):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=X.shape[1])
    u = X @ w_true
    a, b = rng.integers(0, len(X), 600), rng.integers(0, len(X), 600)
    b = np.where(a == b, (b + 1) % len(X), b)
    y = (u[a] > u[b]).astype(np.float32)
    family = np.arange(600) // 3
    return R.fit_taste(X, a, b, y, np.ones(600, np.float32), family, "cpu")


def test_manifest_round_trips_through_json(tmp_path):
    g, cols = _toy_graph()
    cat = _catalog()
    m = _manifest(g, cols, cat, _setting())
    d = m.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert C.Manifest.from_dict({**d, "future_field": 1}) == m  # unknown keys are ignored
    assert C.setting_from_dict(m.setting) == _setting()
    assert m.c_ref == 0.7 and m.codes == {"odor": "v2", "allsens": "v1"}


def test_write_and_read_checkpoint(tmp_path):
    g, cols = _toy_graph()
    cat = _catalog()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 30)).astype(np.float32)
    head = _planted_head(X)
    m = _manifest(g, cols, cat, _setting())
    out = tmp_path / "checkpoints" / "fly-test"
    C.write_checkpoint(out, m, head)
    assert (out / "manifest.json").exists()
    on_disk = json.loads((out / "manifest.json").read_text())
    for key in ("version", "graph_hash", "feathers", "min_syn", "sign_map", "nt_missing",
                "columns_hash", "setting", "c_ref", "codes", "lfg_trait_config_commit",
                "catalog_hash", "taste", "created_at"):
        assert key in on_disk
    m2, head2 = C.read_checkpoint(out)
    assert m2 == m
    assert isinstance(head2, R.TasteHead)
    np.testing.assert_array_equal(head2.w, head.w)
    assert head2.taste_sd == head.taste_sd and head2.lam == head.lam
    np.testing.assert_allclose(head2.score(X), head.score(X), rtol=1e-6)


def test_read_checkpoint_refuses_a_rarity_head(tmp_path):
    g, cols = _toy_graph()
    m = _manifest(g, cols, _catalog(), _setting())
    X = np.random.default_rng(0).normal(size=(50, 5)).astype(np.float32)
    out = tmp_path / "ck"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps(m.to_dict()))
    R.save_head(out, R.fit_rarity(X, X @ np.ones(5), "snap"))
    with pytest.raises(ValueError):
        C.read_checkpoint(out)


def test_fly_brain_reproduces_features_for_on_a_tiny_graph(tmp_path):
    g, cols = _toy_graph()
    cat = _catalog()
    catalog_dir = _install(tmp_path, g, cols, cat)
    setting = _setting()
    rng = np.random.default_rng(1)
    looks = [uniform_look(cat, rng) for _ in range(40)]
    X, _ = _features(g, cols, cat, setting, looks)
    assert X.shape == (40, 30) and X.std() > 0
    head = _planted_head(X)
    out = tmp_path / "checkpoints" / "fly-test"
    C.write_checkpoint(out, _manifest(g, cols, cat, setting), head)

    brain = C.FlyBrain.load(out, "cpu", catalog_dir)
    assert brain.manifest.version == "fly-test"
    assert brain.setting == setting
    assert isinstance(brain.sim, Simulator) and isinstance(brain.builder, InputBuilder)
    assert brain.builder.name == "all-sensory-brain" and brain.builder.g_in == 2.0
    assert brain.head.taste_sd == head.taste_sd
    assert brain.graph.graph_hash() == g.graph_hash()
    assert brain.catalog == cat

    # taste: the c_ref simulation through the head, exactly what features_for gives
    taste = brain.taste(looks)
    assert taste.shape == (40,)
    np.testing.assert_allclose(taste, head.score(X), rtol=1e-5, atol=1e-6)
    assert brain.stats["neurons_fired"] > 0 and brain.stats["ms"] >= 0.0
    assert brain.stats["n_looks"] == 40
    assert set(brain.stats) >= {"neurons_fired", "ms", "active_frac", "readout_active_frac",
                                "max_step_frac"}

    # features under c_ref equal features(conc=None) and a flat c_ref matrix
    X0, stats0 = brain.features(looks)
    np.testing.assert_allclose(X0, X, rtol=1e-6)
    flat = np.full((40, len(SLOTS)), C_REF, np.float32)
    X1, _ = brain.features(looks, conc=flat)
    np.testing.assert_allclose(X1, X, rtol=1e-6)
    assert "active_frac" in stats0

    # rarity: a live-concentration simulation through a rarity head
    supply = {"n_live": 1000, "counts": {s: {f"{s}{i}": 100 * (i + 1) for i in range(3)}
                                         for s in SLOTS if s != "Body"}}
    supply["counts"]["Body"] = {"B0": 1000}
    conc = R.concentrations(looks, supply, cat)
    assert not np.allclose(conc, C_REF)
    X_live, _ = _features(g, cols, cat, setting, looks, conc=conc)
    assert not np.allclose(X_live, X)  # concentration reaches the readout
    target = np.array([R.nft_rarity(lk, supply) for lk in looks])
    rhead = R.fit_rarity(X_live, target, "snap1")
    rar = brain.rarity(looks, conc, rhead)
    np.testing.assert_allclose(rar, rhead.score(X_live), rtol=1e-5, atol=1e-6)
    # the taste path is pinned to c_ref: it never sees live concentrations
    np.testing.assert_allclose(brain.taste(looks), head.score(X), rtol=1e-5, atol=1e-6)


def test_fly_brain_is_deterministic_and_batches_like_one_call(tmp_path):
    g, cols = _toy_graph()
    cat = _catalog()
    catalog_dir = _install(tmp_path, g, cols, cat)
    setting = _setting()
    rng = np.random.default_rng(2)
    looks = [uniform_look(cat, rng) for _ in range(12)]
    X, _ = _features(g, cols, cat, setting, looks)
    out = tmp_path / "ck"
    C.write_checkpoint(out, _manifest(g, cols, cat, setting), _planted_head(X))
    brain = C.FlyBrain.load(out, "cpu", catalog_dir)
    once = brain.taste(looks)
    parts = np.concatenate([brain.taste(looks[:5]), brain.taste(looks[5:])])
    np.testing.assert_allclose(once, parts, rtol=1e-6)
    np.testing.assert_array_equal(once, brain.taste(looks))


def test_fly_brain_refuses_a_graph_or_columns_mismatch(tmp_path):
    g, cols = _toy_graph()
    cat = _catalog()
    catalog_dir = _install(tmp_path, g, cols, cat)
    X = np.random.default_rng(0).normal(size=(50, 30)).astype(np.float32)
    head = _planted_head(X)
    bad_graph = tmp_path / "bad-graph"
    C.write_checkpoint(bad_graph, _manifest(g, cols, cat, _setting(), graph_hash="0" * 64), head)
    with pytest.raises(C.CheckpointMismatch, match="graph"):
        C.FlyBrain.load(bad_graph, "cpu", catalog_dir)
    bad_cols = tmp_path / "bad-cols"
    C.write_checkpoint(bad_cols, _manifest(g, cols, cat, _setting(), columns_hash="f" * 64), head)
    with pytest.raises(C.CheckpointMismatch, match="column"):
        C.FlyBrain.load(bad_cols, "cpu", catalog_dir)
    # a manifest without a columns hash pins nothing about the columns
    ok = tmp_path / "ok"
    C.write_checkpoint(ok, _manifest(g, cols, cat, _setting(), columns_hash=None), head)
    assert C.FlyBrain.load(ok, "cpu", catalog_dir).head is not None


def test_fly_brain_warns_on_a_catalog_change_but_loads(tmp_path, caplog):
    g, cols = _toy_graph()
    cat = _catalog()
    catalog_dir = _install(tmp_path, g, cols, cat)
    X = np.random.default_rng(0).normal(size=(50, 30)).astype(np.float32)
    out = tmp_path / "ck"
    C.write_checkpoint(out, _manifest(g, cols, cat, _setting(), catalog_hash="1" * 64),
                       _planted_head(X))
    with caplog.at_level("WARNING", logger="lfg_fly.brain.checkpoint"):
        brain = C.FlyBrain.load(out, "cpu", catalog_dir)
    assert brain.catalog_matches is False
    assert "catalog" in caplog.text


def test_fly_brain_with_eyes_builds_the_bank_and_pinned_zorder(tmp_path):
    g, cols = _toy_graph(with_eyes=True)
    cat = _catalog(n_values=2)
    catalog_dir = _install(tmp_path, g, cols, cat, layers=True)
    setting = _setting("eyes+nose")
    rng = np.random.default_rng(3)
    looks = [uniform_look(cat, rng) for _ in range(10)]
    bank = RN.LayerBank(cat, catalog_dir, size=64, device="cpu")
    zorder = RN.zorder_from_config(Z_CFG)
    X, _ = _features(g, cols, cat, setting, looks, bank=bank, zorder=zorder)
    out = tmp_path / "ck"
    C.write_checkpoint(out, _manifest(g, cols, cat, setting), _planted_head(X))
    brain = C.FlyBrain.load(out, "cpu", catalog_dir)
    assert brain.builder.uses_eyes and brain.bank is not None
    assert brain.zorder.key("Background", "x") < brain.zorder.key("Head", "x")
    np.testing.assert_allclose(brain.taste(looks), brain.head.score(X), rtol=1e-5, atol=1e-6)
    # the retina came from the columns file: three photoreceptors see the render
    assert len(brain.retina.neurons) == 3


def test_fly_brain_needs_no_layers_when_the_code_has_no_eyes(tmp_path):
    g, cols = _toy_graph()
    cat = _catalog()
    catalog_dir = _install(tmp_path, g, cols, cat, layers=False)  # no PNGs, no trait_config
    X = np.random.default_rng(0).normal(size=(50, 30)).astype(np.float32)
    out = tmp_path / "ck"
    C.write_checkpoint(out, _manifest(g, cols, cat, _setting("nose")), _planted_head(X))
    brain = C.FlyBrain.load(out, "cpu", catalog_dir)
    assert brain.bank is None
    assert brain.taste([uniform_look(cat, np.random.default_rng(0))]).shape == (1,)
