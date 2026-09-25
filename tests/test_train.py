"""Training on the critic's labels (spec §3.3), the controls and REPORT.md (spec §3.4), on a
tiny synthetic graph, CPU only."""

import argparse
import dataclasses
import json

import numpy as np
import pytest
from test_grid import _toy_context

from lfg_fly import paths
from lfg_fly.brain import checkpoint as C
from lfg_fly.brain import readout as R
from lfg_fly.brain.senses import C_REF
from lfg_fly.brain.sim import BrainParams
from lfg_fly.connectome import build as B
from lfg_fly.connectome.columns import Columns, save_columns
from lfg_fly.teacher import cli_train
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import probe as P
from lfg_fly.teacher import rewire as RW
from lfg_fly.teacher import sample as T
from lfg_fly.teacher import train as TR
from lfg_fly.teacher.render import LFG_TRAIT_CONFIG_COMMIT

SETTING = G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.0, steps=8, burn_in=5),
                    "all-sensory-brain", 1.0)
NOSE = G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.0, steps=8, burn_in=5), "nose", 1.0)
EYES = G.Setting(BrainParams(kind="rate", g_syn=2.0, bias=0.1, steps=8, burn_in=5), "eyes", 0.5)


def _ctx(tmp_path):
    """test_grid's learnable toy, with no probe set: the labels come from the critic."""
    ctx = _toy_context(tmp_path, feedforward=True)
    return dataclasses.replace(ctx, probe_set=None, confirm_set=None)


def _critic_round(cat, n_families=120, seed=2000, flip_every=10):
    """A round-1 jsonl as critic.assemble writes it: a planted additive taste plays the critic,
    strength follows the utility gap, and every `flip_every`-th pair is a position-bias flip."""
    pairs = T.make_pairs(cat, n_families, 3, seed=seed)
    taste = T.plant_taste(cat, seed + 1)
    du = np.array([taste.utility(p.a) - taste.utility(p.b) for p in pairs])
    q1, q2 = np.quantile(np.abs(du), [1 / 3, 2 / 3])
    rows = []
    for i, (p, d) in enumerate(zip(pairs, du, strict=True)):
        winner = "A" if d > 0 else "B"
        strength = 1 + int(abs(d) > q1) + int(abs(d) > q2)
        rec = {"pair_id": i, "family": p.family, "near": p.near, "a": list(p.a), "b": list(p.b),
               "winner": winner, "strength": strength, "idea_A": f"idea {i}a",
               "idea_B": f"idea {i}b", "reason": "planted", "batch": f"b{i // 40:03d}"}
        if i % flip_every == 0:
            other = "B" if winner == "A" else "A"
            rec["flip"] = {"winner": other, "strength": 1}
            rec["flipped"] = True
        elif i % flip_every == 5:
            rec["flip"] = {"winner": winner, "strength": strength}
            rec["flipped"] = False
        if i % 7 == 3:
            rec["dup"] = {"winner": winner, "strength": strength}
            rec["agree"] = True
        rows.append(rec)
    return rows


def _write_round(root, rows, round_name="r1"):
    d = root / "data" / "critic"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{round_name}.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (d / f"{round_name}-qc.json").write_text(json.dumps(QC, indent=1))
    return path


QC = {"n_pairs": 360, "n_flip_checked": 72, "flip_rate": 0.5, "n_dropped": 36,
      "n_dup_checked": 51, "agreement": 0.84, "implied_ceiling": 0.912,
      "a_wins": 0.49, "strength_hist": {"1": 120, "2": 120, "3": 120}}


def _fast(monkeypatch):
    """One L2 strength and a small MLP: the same protocol, cheaper on the CPU. The lambdas must
    forward `w`: the trainer always passes the critic's strength."""
    real_fit_bt, real_mlp = P.fit_bt, P.evaluate_mlp
    monkeypatch.setattr(P, "fit_bt", lambda D, y, groups, device, **kw: real_fit_bt(
        D, y, groups, device, lambdas=(1.0,), **kw))
    monkeypatch.setattr(P, "evaluate_mlp", lambda X, ps, device="cpu", seed=0, **kw: real_mlp(
        X, ps, device=device, seed=seed, hidden=8, lambdas=(1.0,), folds=2, **kw))


def _install_graph(ctx, tmp_path):
    """The toy graph and an empty column file where FlyBrain.load reads them, and a catalog
    dir holding catalog-male.json."""
    g = ctx.graph
    B.save_graph(g, paths.graph_dir() / f"graph-syn{g.min_syn}.npz")
    cols = Columns(neuron=np.array([], np.int64), eye=np.array([], "<U1"),
                   q=np.array([], np.int32), r=np.array([], np.int32),
                   share=np.array([], np.float32), dropped=0, total=0)
    save_columns(cols, paths.graph_dir() / f"columns-syn{g.min_syn}.npz")
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir(exist_ok=True)
    (catalog_dir / "catalog-male.json").write_text(ctx.catalog.to_json())
    return catalog_dir


# ---------------------------------------------------------------- labels


def test_load_labels_drops_flipped_pairs_and_splits_by_family(tmp_path):
    ctx = _ctx(tmp_path)
    rows = _critic_round(ctx.catalog)
    path = _write_round(tmp_path, rows)
    labels = TR.load_labels(path, ctx.catalog, test_frac=0.2, seed=0)
    kept = [r for r in rows if not r.get("flipped")]
    assert labels.n_dropped == sum(1 for r in rows if r.get("flipped")) == 36
    assert len(labels.y) == len(kept) == len(labels.a_idx) == len(labels.b_idx)
    assert len(labels.family) == len(labels.near) == len(labels.w) == len(labels.test) == len(kept)
    # looks are indexed once each; pair i is (looks[a_idx[i]], looks[b_idx[i]])
    assert len(set(labels.looks)) == len(labels.looks)
    for i, r in enumerate(kept):
        assert labels.looks[labels.a_idx[i]] == tuple(r["a"])
        assert labels.looks[labels.b_idx[i]] == tuple(r["b"])
        assert labels.y[i] == (1.0 if r["winner"] == "A" else 0.0)
        assert labels.w[i] == r["strength"]
        assert labels.family[i] == r["family"] and bool(labels.near[i]) == r["near"]
    assert labels.y.dtype == np.float32 and labels.w.dtype == np.float32
    assert labels.test.dtype == bool and labels.pair_id.tolist() == [r["pair_id"] for r in kept]
    # the test split is by family: >= 20% of families, and no family straddles the split
    fams = np.unique(labels.family)
    test_fams = np.unique(labels.family[labels.test])
    assert len(test_fams) >= int(np.ceil(0.2 * len(fams)))
    assert not (set(test_fams) & set(labels.family[~labels.test]))
    assert labels.n_test == int(labels.test.sum()) and labels.n_train == int((~labels.test).sum())
    assert labels.n_pairs == len(kept) and labels.source == str(path)
    # deterministic in the seed, different for another seed
    again = TR.load_labels(path, ctx.catalog, test_frac=0.2, seed=0)
    assert np.array_equal(again.test, labels.test)
    other = TR.load_labels(path, ctx.catalog, test_frac=0.2, seed=1)
    assert not np.array_equal(other.test, labels.test)
    # a larger test fraction holds out more families
    big = TR.load_labels(path, ctx.catalog, test_frac=0.5, seed=0)
    assert len(np.unique(big.family[big.test])) > len(test_fams)


def test_load_labels_refuses_a_look_outside_the_catalog_or_a_bad_record(tmp_path):
    ctx = _ctx(tmp_path)
    rows = _critic_round(ctx.catalog, n_families=4)
    rows[1]["a"][3] = "Clothing99"
    path = _write_round(tmp_path, rows)
    with pytest.raises(ValueError, match="Clothing99"):
        TR.load_labels(path, ctx.catalog)
    rows = _critic_round(ctx.catalog, n_families=4)
    rows[2]["winner"] = "C"
    path = _write_round(tmp_path, rows)
    with pytest.raises(ValueError, match="winner"):
        TR.load_labels(path, ctx.catalog)
    rows = _critic_round(ctx.catalog, n_families=4)
    rows[2]["a"] = rows[2]["a"][:8]
    path = _write_round(tmp_path, rows)
    with pytest.raises(ValueError, match="9"):
        TR.load_labels(path, ctx.catalog)
    with pytest.raises(FileNotFoundError):
        TR.load_labels(tmp_path / "nope.jsonl", ctx.catalog)


def test_load_labels_with_nothing_left_raises(tmp_path):
    ctx = _ctx(tmp_path)
    rows = _critic_round(ctx.catalog, n_families=2, flip_every=1)  # every pair flipped
    path = _write_round(tmp_path, rows)
    with pytest.raises(ValueError, match="flipped"):
        TR.load_labels(path, ctx.catalog)


# ---------------------------------------------------------------- the gate and Phase 0 settings


def test_gate_is_the_ci_lower_bound_above_055():
    assert TR.GATE_CI_LO == 0.55
    assert TR.gate({"ci_lo": 0.551, "heldout": 0.6}) == {"ci_lo_min": 0.55, "ci_lo": 0.551,
                                                        "pass": True}
    assert TR.gate({"ci_lo": 0.55, "heldout": 0.6})["pass"] is False  # above, not at
    assert TR.gate({"ci_lo": 0.40, "heldout": 0.8})["pass"] is False  # the CI, not the point


def test_best_phase0_settings_picks_the_best_passed_row_per_code(tmp_path):
    def row(setting, heldout, passed):
        return {"key": setting.key(), "setting": setting.as_dict(), "passed": passed,
                "probe": {"heldout": heldout, "ci_lo": heldout - 0.03, "ci_hi": heldout + 0.03}}

    nose_bad = G.Setting(BrainParams(kind="lif", g_syn=8.0, bias=0.1, steps=60), "nose", 0.5)
    eyes_better_but_failed = G.Setting(BrainParams(kind="rate", g_syn=0.5, bias=0.0, steps=60),
                                       "eyes", 2.0)
    rows = [row(nose_bad, 0.47, True), row(NOSE, 0.65, True), row(EYES, 0.59, True),
            row(eyes_better_but_failed, 0.70, False),
            row(SETTING, 0.76, True)]
    path = tmp_path / "stage_b.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    best = TR.best_phase0_settings(path)
    assert set(best) == {"nose", "eyes", "all-sensory-brain"}
    assert C.setting_from_dict(best["nose"]["setting"]) == NOSE
    assert C.setting_from_dict(best["eyes"]["setting"]) == EYES  # the passed one, not the 0.70
    assert best["eyes"]["probe"]["heldout"] == 0.59
    assert TR.best_phase0_settings(tmp_path / "missing.jsonl") == {}
    # with no passer for a code, its best-probed row stands in and is flagged
    only_failed = [row(eyes_better_but_failed, 0.70, False)]
    path.write_text("".join(json.dumps(r) + "\n" for r in only_failed))
    assert TR.best_phase0_settings(path)["eyes"]["passed"] is False


def test_phase0_setting_reads_the_verdicts_best_or_falls_back(tmp_path):
    v = {"pass": True, "confirmed": True, "best": {"setting": SETTING.as_dict()}}
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(v))
    assert TR.phase0_setting(path) == SETTING
    assert TR.phase0_setting(tmp_path / "none.json") == TR.PHASE0_WINNER
    path.write_text(json.dumps({**v, "pass": False}))
    assert TR.phase0_setting(path) == TR.PHASE0_WINNER  # a failed Phase 0 pins nothing
    assert TR.PHASE0_WINNER.code == "all-sensory-brain" and TR.PHASE0_WINNER.g_in == 2.0
    assert TR.PHASE0_WINNER.brain.kind == "rate" and TR.PHASE0_WINNER.brain.steps == 60


# ---------------------------------------------------------------- training


def test_train_without_controls_writes_results_checkpoint_and_report(tmp_path, monkeypatch):
    _fast(monkeypatch)
    ctx = _ctx(tmp_path)
    catalog_dir = _install_graph(ctx, tmp_path)
    root = tmp_path / "repo"
    _write_round(root, _critic_round(ctx.catalog))
    results = TR.train_round(ctx, SETTING, round_name="r1", version="fly-test", root=root,
                             controls=False)

    # results.json: the fly's held-out number with its CI, the gate, the labels' counts
    out = root / "data" / "train" / "fly-test" / "results.json"
    on_disk = json.loads(out.read_text())
    assert on_disk == json.loads(json.dumps(results))
    fly = results["fly"]
    assert 0.0 <= fly["ci_lo"] <= fly["heldout"] <= fly["ci_hi"] <= 1.0
    assert fly["heldout"] > 0.6  # the learnable toy learns the planted critic
    assert fly["n_test"] == results["labels"]["n_test"] > 0
    assert results["labels"]["n_dropped"] == 36 and results["labels"]["n_pairs"] == 324
    assert results["labels"]["n_train"] + results["labels"]["n_test"] == 324
    assert results["gate"] == TR.gate(fly)
    assert results["controls_run"] is False and results["controls"] == {}
    assert results["paired"] == {} and results["setting"] == SETTING.as_dict()
    assert results["graph"]["hash"] == ctx.graph.graph_hash()
    assert set(fly["stats"]) >= {"active_frac", "readout_active_frac", "max_step_frac"}
    # the shipped head is the evaluated model: same held-out pairs, the same verdicts
    assert results["head"]["heldout"] == pytest.approx(fly["heldout"], abs=0.02)
    assert results["head"]["agreement_with_protocol"] >= 0.98
    assert results["head"]["taste_sd"] > 0

    # the checkpoint: a full manifest and the head
    ck = root / "checkpoints" / "fly-test"
    manifest, head = C.read_checkpoint(ck)
    assert results["checkpoint"]["dir"] == str(ck)
    assert manifest.version == "fly-test"
    assert manifest.graph_hash == ctx.graph.graph_hash()
    assert manifest.min_syn == ctx.graph.min_syn == 1
    assert manifest.sign_map == B.SIGN and manifest.nt_missing == ctx.graph.nt_missing
    assert manifest.setting == SETTING.as_dict() and manifest.c_ref == C_REF
    assert manifest.codes == {"odor": "v2", "allsens": "v1"}
    assert manifest.lfg_trait_config_commit == LFG_TRAIT_CONFIG_COMMIT
    assert manifest.catalog_hash == C.catalog_hash(ctx.catalog)
    assert manifest.columns_hash is not None  # the columns file beside the graph was hashed
    assert manifest.feathers == {}  # no raw/manifest.json in the tmp data dir: pinned nothing
    assert results["checkpoint"]["feathers_known"] is False
    for key in ("heldout", "ci_lo", "ci_hi", "lam", "n_train", "n_test", "gate_pass"):
        assert key in manifest.taste
    assert manifest.taste["gate_pass"] == results["gate"]["pass"]
    assert manifest.taste["heldout"] == fly["heldout"] and manifest.taste["lam"] == head.lam
    assert isinstance(head, R.TasteHead) and head.taste_sd == results["head"]["taste_sd"]

    # FlyBrain loads it and reproduces the head on the labelled looks
    brain = C.FlyBrain.load(ck, "cpu", catalog_dir)
    labels = TR.load_labels(root / "data" / "critic" / "r1.jsonl", ctx.catalog)
    X, _ = G.features_for(ctx, brain.sim, SETTING, labels.looks[:20], {})
    np.testing.assert_allclose(brain.taste(labels.looks[:20]), head.score(X), rtol=1e-5,
                               atol=1e-6)

    # REPORT.md: the gate line, plain either way, and the fact that no control was run
    text = (root / "REPORT.md").read_text()
    verdict = "PASS" if results["gate"]["pass"] else "FAIL"
    assert f"**Gate B: {verdict}**" in text
    assert f"{fly['heldout']:.3f}" in text and f"{fly['ci_lo']:.3f}" in text
    assert "0.55" in text
    assert ("goes live as a chooser" if verdict == "PASS"
            else "does not go live as a chooser") in text
    assert "not run" in text  # every control row says so
    assert "flip rate" in text.lower() and "0.840" in text and "0.912" in text  # the QC file
    assert manifest.graph_hash[:16] in text and "fly-test" in text
    assert "Best-of-6" in text  # the gate's second criterion, declared unmeasured


def test_train_with_controls_scores_every_control_on_the_same_test_pairs(tmp_path, monkeypatch):
    _fast(monkeypatch)
    ctx = _ctx(tmp_path)
    _install_graph(ctx, tmp_path)
    # a ">= 5-synapse" graph: the toy again under another threshold, with its columns
    g5 = dataclasses.replace(ctx.graph, min_syn=5)
    B.save_graph(g5, paths.graph_dir() / "graph-syn5.npz")
    save_columns(Columns(neuron=np.array([], np.int64), eye=np.array([], "<U1"),
                         q=np.array([], np.int32), r=np.array([], np.int32),
                         share=np.array([], np.float32), dropped=0, total=0),
                 paths.graph_dir() / "columns-syn5.npz")
    root = tmp_path / "repo"
    _write_round(root, _critic_round(ctx.catalog, n_families=80))
    stage_b = root / "data" / "probe" / "stage_b.jsonl"
    stage_b.parent.mkdir(parents=True)
    rows = [{"key": s.key(), "setting": s.as_dict(), "passed": True,
             "probe": {"heldout": h}} for s, h in ((NOSE, 0.65), (EYES, 0.59), (SETTING, 0.76))]
    stage_b.write_text("".join(json.dumps(r) + "\n" for r in rows))

    featured = []
    real_features = G.features_for

    def features(ctx_, sim, setting, looks, *a, **kw):
        featured.append((ctx_.graph.graph_hash(), setting))
        return real_features(ctx_, sim, setting, looks, *a, **kw)

    monkeypatch.setattr(TR.G, "features_for", features)
    results = TR.train_round(ctx, SETTING, round_name="r1", version="fly-test", root=root,
                             controls=True)
    ctl = results["controls"]
    assert results["controls_run"] is True
    assert set(ctl) == {"no_brain", "mlp", "rewired", "sign_shuffled", "eyes_only", "nose_only",
                        "syn5"}
    ran = {k for k, v in ctl.items() if "skipped" not in v}
    assert ran == {"no_brain", "mlp", "rewired", "sign_shuffled", "nose_only", "syn5"}
    assert "bank" in ctl["eyes_only"]["skipped"]  # no layer bank on the toy: eyes can't see
    assert ctl["no_brain"]["pixels"] is False  # one-hot only, for the same reason
    assert ctl["nose_only"]["setting"] == NOSE.as_dict()
    assert ctl["nose_only"]["phase0_heldout"] == 0.65
    assert ctl["syn5"]["graph_hash"] == g5.graph_hash() and ctl["syn5"]["min_syn"] == 5
    assert isinstance(results["rewire_repairs"], int)
    n_test = results["fly"]["n_test"]
    for name in ran:
        assert ctl[name]["n_test"] == n_test  # the same held-out pairs
        assert set(ctl[name]) >= {"heldout", "ci_lo", "ci_hi", "lam", "n_test"}
        d = results["paired"][name]
        assert set(d) == {"diff", "lo", "hi"} and d["lo"] <= d["diff"] <= d["hi"]
        assert d["diff"] == pytest.approx(results["fly"]["heldout"] - ctl[name]["heldout"],
                                          abs=1e-9)
        assert results["beats"][name] == (d["lo"] > 0)
    assert "eyes_only" not in results["paired"]
    # the real graph carried the fly and the nose-only run, then Phase 0's twins (seeds 11 and
    # 13; on this all-cholinergic toy the sign shuffle is the identity), then syn5
    real = ctx.graph.graph_hash()
    twin_r = RW.rewired_graph(ctx.graph, seed=ctx.seed + 11, device="cpu")[0].graph_hash()
    twin_s = RW.sign_shuffled_graph(ctx.graph, ctx.seed + 13).graph_hash()
    assert twin_r != real and twin_s == real
    assert [h for h, _ in featured] == [real, real, twin_r, twin_s, g5.graph_hash()]
    assert [s for _, s in featured] == [SETTING, NOSE, SETTING, SETTING, SETTING]
    assert ctl["rewired"]["graph_hash"] == twin_r and ctl["sign_shuffled"]["graph_hash"] == twin_s

    text = (root / "REPORT.md").read_text()
    for label in ("Random mover", "No brain", "MLP", "Rewired", "Sign-shuffled", "Eyes-only",
                  "Nose-only", "5-synapse", "Critic agreement"):
        assert label in text
    assert "not run" in text  # eyes-only
    for name in ran:
        assert f"{ctl[name]['heldout']:.3f}" in text
    assert f"{results['rewire_repairs']:,}" in text
    assert ("wiring matters" in text) or ("wiring does not matter" in text)


# ---------------------------------------------------------------- the report on a fixture


def _pr(h, lo=None, hi=None, **extra):
    lo = h - 0.03 if lo is None else lo
    hi = h + 0.03 if hi is None else hi
    return {"heldout": h, "ci_lo": lo, "ci_hi": hi, "train_acc": h + 0.1, "cv_acc": h - 0.01,
            "lam": 0.1, "n_test": 600, **extra}


def _diff(fly, other):
    d = fly - other
    return {"diff": d, "lo": d - 0.02, "hi": d + 0.02}


def _fixture(gate_pass=True):
    fly = 0.66 if gate_pass else 0.56
    ctl = {"no_brain": _pr(0.70, pixels=True), "mlp": _pr(0.69),
           "rewired": _pr(0.655), "sign_shuffled": _pr(0.60),
           "eyes_only": _pr(0.55, setting=EYES.as_dict(), phase0_heldout=0.59),
           "nose_only": _pr(0.63, setting=NOSE.as_dict(), phase0_heldout=0.65),
           "syn5": {"skipped": "no graph-syn5.npz in the data dir"}}
    paired = {k: _diff(fly, v["heldout"]) for k, v in ctl.items() if "skipped" not in v}
    paired["sign_shuffled"] = {"diff": 0.06, "lo": 0.02, "hi": 0.10}  # the one clear win
    return {
        "version": "fly-v1", "round": "r1", "device": "cuda", "seed": 0,
        "created_at": "2026-09-26T03:00:00+00:00",
        "labels": {"source": "data/critic/r1.jsonl", "n_pairs": 2700, "n_dropped": 300,
                   "n_train": 2100, "n_test": 600, "n_families": 1000, "n_test_families": 200,
                   "near_frac": 0.6},
        "setting": TR.PHASE0_WINNER.as_dict(),
        "graph": {"hash": "f2f62e0a" + "0" * 56, "min_syn": 3, "n": 165000, "edges": 12000000,
                  "nt_missing": 1234},
        "fly": _pr(fly, stats={"active_frac": 0.9, "readout_active_frac": 0.99,
                               "max_step_frac": 0.0}, seconds=120.0),
        "head": {"lam": 0.1, "taste_sd": 1.7, "heldout": fly, "agreement_with_protocol": 1.0,
                 "n_features": 2129},
        "controls": ctl, "paired": paired,
        "beats": {k: v["lo"] > 0 for k, v in paired.items()},
        "rewire_repairs": 22153, "controls_run": True,
        "gate": TR.gate({"ci_lo": fly - 0.03, "heldout": fly}),
        "checkpoint": {"dir": "checkpoints/fly-v1", "feathers_known": True,
                       "manifest": {"version": "fly-v1", "graph_hash": "f2f62e0a" + "0" * 56,
                                    "feathers": {"annotations": "2177e246" + "0" * 56,
                                                 "neurotransmitters": "95c92892" + "0" * 56,
                                                 "weights": "9b3beab1" + "0" * 56},
                                    "min_syn": 3, "nt_missing": 1234, "columns_hash": "c" * 64,
                                    "catalog_hash": "d" * 64, "c_ref": 0.7,
                                    "codes": {"odor": "v2", "allsens": "v1"},
                                    "lfg_trait_config_commit": LFG_TRAIT_CONFIG_COMMIT,
                                    "created_at": "2026-09-26T03:00:00+00:00"}},
    }


def test_write_report_renders_a_fixture_dict(tmp_path):
    results = _fixture(gate_pass=True)
    out = tmp_path / "REPORT.md"
    TR.write_report(results, QC, out)
    text = out.read_text()
    assert text.startswith("# ")
    assert "**Gate B: PASS**" in text and "goes live as a chooser" in text
    assert "0.660" in text and "0.630" in text and "0.690" in text  # the fly and its CI
    # every control row with its CI, the paired difference, and a plain yes/no
    for label, h in (("No brain", 0.70), ("MLP", 0.69), ("Rewired", 0.655),
                     ("Sign-shuffled", 0.60), ("Eyes-only", 0.55), ("Nose-only", 0.63)):
        line = next(ln for ln in text.splitlines() if ln.startswith("| ") and label in ln)
        assert f"{h:.3f}" in line and f"{h - 0.03:.3f}–{h + 0.03:.3f}" in line
    assert "| Random mover | 0.500 |" in text
    assert "5-synapse" in text and "not run: no graph-syn5.npz" in text
    assert "22,153" in text  # rewire repairs
    # the plain answers, unshaded
    assert "does not beat its own raw inputs" in text
    assert "wiring does not matter" in text
    assert "signs matter" in text
    assert "Critic agreement" in text and "0.840" in text and "0.912" in text
    assert "flip rate 0.500" in text.lower() or "flip rate: 0.500" in text.lower()
    # identity
    assert "f2f62e0a" in text and "2177e246" in text and LFG_TRAIT_CONFIG_COMMIT[:12] in text
    assert "checkpoints/fly-v1" in text
    # the rarity head's disclosure (spec §2 Readout)
    assert "rarity" in text.lower() and "additive" in text.lower()
    # the second gate criterion is declared unmeasured, not passed
    assert "Best-of-6" in text and "not" in text

    failing = _fixture(gate_pass=False)
    TR.write_report(failing, QC, out)
    text = out.read_text()
    assert "**Gate B: FAIL**" in text and "does not go live as a chooser" in text
    assert "beats its own raw inputs" not in text.replace("does not beat", "")
    # without a QC file the report says so rather than inventing numbers
    TR.write_report(failing, {}, out)
    assert "no qc" in out.read_text().lower() or "no quality-control" in out.read_text().lower()


def test_write_report_before_a_checkpoint_exists(tmp_path):
    results = _fixture()
    del results["checkpoint"]
    out = tmp_path / "REPORT.md"
    TR.write_report(results, QC, out)
    assert "not written" in out.read_text().lower()


# ---------------------------------------------------------------- the CLI


def test_register_adds_train_with_the_contracts_flags():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli_train.register(sub)
    args = parser.parse_args(["train"])
    assert (args.round, args.device, args.min_syn, args.version, args.no_controls) == (
        "r1", "cuda", 3, "fly-v1", False)
    assert args.func is cli_train._cmd_train
    args = parser.parse_args(["train", "--round", "r2", "--device", "cpu", "--min-syn", "5",
                              "--version", "fly-v2", "--no-controls"])
    assert (args.round, args.device, args.min_syn, args.version, args.no_controls) == (
        "r2", "cpu", 5, "fly-v2", True)


def test_cmd_train_runs_the_round_end_to_end(tmp_path, monkeypatch, capsys):
    _fast(monkeypatch)
    ctx = _ctx(tmp_path)
    _install_graph(ctx, tmp_path)
    root = tmp_path / "repo"
    _write_round(root, _critic_round(ctx.catalog, n_families=60))
    (root / "data" / "probe").mkdir(parents=True)
    (root / "data" / "probe" / "verdict.json").write_text(json.dumps(
        {"pass": True, "confirmed": True, "best": {"setting": SETTING.as_dict()}}))
    args = argparse.Namespace(round="r1", device="cpu", min_syn=1, version="fly-test",
                              no_controls=True)
    rc = cli_train._cmd_train(args, root=root, load_context=lambda device, min_syn: ctx)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Gate B:" in out and "REPORT.md" in out
    assert (root / "REPORT.md").exists()
    assert (root / "checkpoints" / "fly-test" / "manifest.json").exists()
    results = json.loads((root / "data" / "train" / "fly-test" / "results.json").read_text())
    assert results["controls_run"] is False and results["setting"] == SETTING.as_dict()
    assert results["round"] == "r1" and results["version"] == "fly-test"

    # no labels for that round: a clear refusal, nothing written
    args.round = "r9"
    rc = cli_train._cmd_train(args, root=root, load_context=lambda device, min_syn: ctx)
    assert rc == 2 and "r9" in capsys.readouterr().out
    assert not (root / "checkpoints" / "fly-test" / "r9").exists()
