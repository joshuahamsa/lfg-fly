import dataclasses
import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lfg_fly.brain.senses import CODE_NAMES, SLOTS, Retina
from lfg_fly.brain.sim import BrainParams
from lfg_fly.connectome import build as B
from lfg_fly.connectome.neurons import populations
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import probe as P
from lfg_fly.teacher import report as R
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.render import ZOrder


def _toy_context(tmp_path, feedforward: bool, confirm: bool = False):
    """ORNs -> readout. Feed-forward: a clean, learnable map. Otherwise: strong recurrence.

    `confirm` adds the confirmation set as the CLI builds it: a fresh instance of
    the task from seed 0 + CONFIRM_SEED_OFFSET.
    """
    rng = np.random.default_rng(0)
    n_orn, n_mid, n_ro = 60, 80, 60
    n = n_orn + n_mid + n_ro
    sc = ["cb_sensory"] * n_orn + ["cb_intrinsic"] * n_mid + ["descending_neuron"] * n_ro
    types = [f"ORN_G{i % 12}" for i in range(n_orn)] + ["x"] * (n_mid + n_ro)
    pre, post = [], []
    for o in range(n_orn):
        for m in rng.choice(n_mid, 6, replace=False):
            pre.append(o)
            post.append(n_orn + m)
    for m in range(n_mid):
        for r in rng.choice(n_ro, 5, replace=False):
            pre.append(n_orn + m)
            post.append(n_orn + n_mid + r)
    if not feedforward:
        for _ in range(900):
            a, b = rng.integers(n_orn, n, 2)
            if a != b:
                pre.append(int(a))
                post.append(int(b))
    edges = pd.DataFrame({"pre": pre, "post": post}).drop_duplicates()
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": sc, "type": types, "rootSide": [None] * n})
    nt = pd.DataFrame({"body": np.arange(1, n + 1), "consensus_nt": ["acetylcholine"] * n})
    g = B.build_graph(ann, nt, pd.DataFrame({"body_pre": edges.pre + 1,
                                             "body_post": edges.post + 1,
                                             "weight": 5}), min_syn=1)
    pops = populations(g)
    retina = Retina(neurons=np.array([], np.int64), px=np.array([], np.int64),
                    py=np.array([], np.int64), channel=np.array([], np.int64))
    values = {s: [f"{s}{i}" for i in range(6)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1"]
    cat = Catalog(body="male", values=values, odds={}, api="x")
    ps = T.make_probe_set(cat, n_pairs=900, seed=0)
    z = ZOrder(layer_z={s: float(i) for i, s in enumerate(SLOTS)}, overrides={},
               rank={s: i for i, s in enumerate(SLOTS)})
    extra = {"confirm_set": T.make_probe_set(cat, n_pairs=900, seed=1000)} if confirm else {}
    return G.Context(graph=g, pops=pops, retina=retina, catalog=cat, bank=None, zorder=z,
                     device="cpu", probe_set=ps, seed=0, **extra)


TOY_FINGERPRINT = "e14848875e27bbf2"  # the learnable toy's context at 7e50e0d, before Task 11b


def _pr(heldout: float) -> dict:
    return {"heldout": heldout, "ci_lo": round(heldout - 0.04, 3),
            "ci_hi": round(heldout + 0.04, 3)}


def _stage_c(best: dict, confirmation: float, seed: int = 0, rewired: float = 0.5,
             sign_shuffled: float = 0.49) -> dict:
    """A stage_c.json as Task 11b writes it: the best setting scored on the confirmation set."""
    return {"key": best["key"], "context": best["context"],
            "confirm_seed": seed + G.CONFIRM_SEED_OFFSET, "confirmation": _pr(confirmation),
            "one_hot_confirmation": _pr(0.76), "bayes_confirmation": 0.83,
            "rewired": _pr(rewired), "sign_shuffled": _pr(sign_shuffled),
            "rewire_repairs": 3, "decodability": _dec(0.9)}


def test_learnable_toy_passes_and_chaotic_toy_fails_smoothness(tmp_path):
    good = _toy_context(tmp_path, feedforward=True)
    # all-sensory codes give the toy ~60 identity channels
    # (its 12 glomeruli would bottleneck "nose")
    s_good = G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.0, steps=8, burn_in=5),
                       "all-sensory-brain", 1.0)
    a = G.stage_a(good, [s_good], tmp_path / "a.jsonl")
    assert a[0]["smooth"]["ok"] is True
    b = G.stage_b(good, a, tmp_path / "b.jsonl", cap=4, probe_all=True)
    assert b[0]["probe"]["heldout"] > 0.6
    # every probed setting carries per-slot decodability (spec §2), not just Stage C's best
    dec = b[0]["decodability"]
    assert list(dec) == list(SLOTS)
    assert all(d["acc"] > d["chance"] for d in dec.values())  # the learnable toy decodes
    assert a[0]["context"] == b[0]["context"] == G.context_fingerprint(good)

    bad = _toy_context(tmp_path, feedforward=False)
    s_bad = G.Setting(BrainParams(kind="lif", g_syn=12.0, bias=0.3, steps=40, burn_in=20),
                      "nose", 3.0)
    a_bad = G.stage_a(bad, [s_bad], tmp_path / "a2.jsonl")
    assert a_bad[0]["passed"] is False


def test_stage_a_is_resumable(tmp_path):
    ctx = _toy_context(tmp_path, feedforward=True)
    s = G.Setting(BrainParams(kind="rate", g_syn=1.0, steps=8, burn_in=5), "nose", 1.0)
    out = tmp_path / "a.jsonl"
    G.stage_a(ctx, [s], out)
    first = out.read_text()
    G.stage_a(ctx, [s], out)
    assert out.read_text() == first  # the second run found the key and skipped it


def test_stage_a_ignores_records_from_another_context(tmp_path, caplog):
    ctx = _toy_context(tmp_path, feedforward=True)
    s = G.Setting(BrainParams(kind="rate", g_syn=1.0, steps=8, burn_in=5), "nose", 1.0)
    out = tmp_path / "a.jsonl"
    stale = G.stage_a(ctx, [s], out)[0]
    # the same setting under another graph / probe set / seed must not be reused
    assert G.context_fingerprint(dataclasses.replace(ctx, seed=1)) != stale["context"]
    stale = {**stale, "context": "0123456789abcdef", "passed": "stale"}
    out.write_text(json.dumps(stale, sort_keys=True) + "\n")
    with caplog.at_level(logging.WARNING, logger="lfg_fly.teacher.grid"):
        again = G.stage_a(ctx, [s], out)
    assert again[0]["passed"] in (True, False)  # recomputed, not the stale record
    assert again[0]["context"] == G.context_fingerprint(ctx)
    assert len(G.read_jsonl(out)) == 2  # the stale record is kept on disk, never reused
    assert G.current_records(out, again[0]["context"]) == again
    assert "ignored 1 record(s) from a different context" in caplog.text


def test_read_jsonl_skips_a_truncated_last_line(tmp_path, caplog):
    path = tmp_path / "x.jsonl"
    path.write_text('{"key": "a"}\n{"key": "b"}\n{"key": "c", "sm')  # a killed append
    with caplog.at_level(logging.WARNING, logger="lfg_fly.teacher.grid"):
        assert G.read_jsonl(path) == [{"key": "a"}, {"key": "b"}]
    assert "truncated last line" in caplog.text
    # resuming appends after the fragment without gluing the new record onto it
    G._append(path, {"key": "c"})
    G._append(path, {"key": "d"})
    assert G.read_jsonl(path) == [{"key": "a"}, {"key": "b"}, {"key": "c"}, {"key": "d"}]
    # a whole record that only lost its newline is kept
    path.write_text('{"key": "a"}\n{"key": "b"}')
    G._append(path, {"key": "c"})
    assert G.read_jsonl(path) == [{"key": "a"}, {"key": "b"}, {"key": "c"}]
    # any other bad line still raises
    path.write_text('{"key": "a"}\n{"key": \n{"key": "c"}\n')
    with pytest.raises(json.JSONDecodeError):
        G.read_jsonl(path)


def test_verdict_drops_twins_of_another_setting_or_context():
    a = [{"key": "k1", "passed": True}]
    b = [{"key": "k1", "context": "ctx1", "passed": True,
          "probe": {"heldout": 0.61, "ci_lo": 0.57, "ci_hi": 0.65}}]
    twins = _stage_c(b[0], 0.62, rewired=0.55, sign_shuffled=0.52)
    ok = G.verdict(a, b, twins, {"heldout": 0.77}, 0.84)
    assert set(ok["controls"]) == {"one_hot", "bayes_ceiling", "rewired", "sign_shuffled"}
    old = {k: v for k, v in twins.items() if k != "confirm_seed"}  # written before Task 11b
    for stale in ({**twins, "key": "k0"},  # an earlier best's twins
                  {**twins, "context": "ctx0"},  # same setting, other graph
                  {**twins, "confirm_seed": 1},  # another confirmation set
                  old):  # twins scored on the selection set
        v = G.verdict(a, b, stale, {"heldout": 0.77}, 0.84)
        assert set(v["controls"]) == {"one_hot", "bayes_ceiling"}
    b[0]["passed"] = False  # no eligible setting: a leftover stage_c.json adds nothing
    v = G.verdict(a, b, twins, {"heldout": 0.77}, 0.84)
    assert v["best"] is None and set(v["controls"]) == {"one_hot", "bayes_ceiling"}


def _dec(acc: float) -> dict:
    return {slot: {"acc": acc, "chance": 0.25, "values": 7} for slot in SLOTS}


def _row(key, code, heldout, passed, context="ctx1"):
    return {"key": key, "context": context, "passed": passed, "dropped_by_cap": 0,
            "setting": {"brain": {"kind": "rate", "g_syn": 1.0, "bias": 0.0}, "code": code,
                        "g_in": 1.0},
            "probe": {"heldout": heldout, "ci_lo": heldout - 0.05, "ci_hi": heldout + 0.05},
            "decodability": _dec(heldout)}


def test_report_gives_decodability_per_code_and_only_the_best_settings_twins(tmp_path):
    rows = [_row("k1", "nose", 0.52, False), _row("k2", "nose", 0.55, False),
            _row("k3", "eyes", 0.51, False), _row("k9", "eyes", 0.99, True, context="old")]
    a = [{"key": r["key"], "context": r["context"], "passed": False,
          "setting": r["setting"]} for r in rows]
    (tmp_path / "stage_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a))
    (tmp_path / "stage_b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "stage_c.json").write_text(json.dumps(
        {"key": "k9", "context": "old", "rewire_repairs": 3, "decodability": _dec(0.9),
         "rewired": {"heldout": 0.5}, "sign_shuffled": {"heldout": 0.5}}))
    b = G.current_records(tmp_path / "stage_b.jsonl", "ctx1")
    v = {**G.verdict(a, b, {}, {"heldout": 0.77}, 0.84), "context": "ctx1"}
    (tmp_path / "verdict.json").write_text(json.dumps(v))
    md = tmp_path / "PHASE0.md"
    R.write_report(tmp_path, md)
    text = md.read_text()
    assert "**Verdict: FAIL**" in text
    # nothing passed the gate, yet every probed code's decodability is reported
    sec = text.split("## Per-slot decodability by input code")[1]
    assert "| nose | rate | 1.0 | 0.0 | 1.0 | no | 0.550 |" in sec  # nose's best-probed setting
    assert "| eyes | rate | 1.0 | 0.0 | 1.0 | no | 0.510 |" in sec  # not the other context's
    assert "| Slot | Chance | nose | eyes |" in sec
    slot_rows = [ln.split(" | ")[0][2:] for ln in sec.splitlines() if ln.startswith("| ")
                 and ln.split(" | ")[0][2:] in SLOTS]
    assert slot_rows == list(SLOTS)  # SLOTS order, not alphabetical
    assert "| Background | 25.0% | 55.0% | 51.0% |" in sec
    assert "No Stage B setting was probed for: eyes+nose, all-sensory-brain, all-sensory+vnc." \
        in sec
    assert "0.990" not in text  # the stale-context row is in no table
    # the leftover stage_c.json belongs to another setting and context: no Stage C, no twins
    assert "## Stage C" not in text and "twin of the best" not in text

    # once Stage C is the best setting's own run in this context, it is reported
    rows[1]["passed"] = True
    (tmp_path / "stage_b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = _stage_c(rows[1], 0.62)
    (tmp_path / "stage_c.json").write_text(json.dumps(c, sort_keys=True))
    b = G.current_records(tmp_path / "stage_b.jsonl", "ctx1")
    v = {**G.verdict(a, b, c, {"heldout": 0.77}, 0.84), "context": "ctx1"}
    (tmp_path / "verdict.json").write_text(json.dumps(v))
    R.write_report(tmp_path, md)
    text = md.read_text()
    assert "| rewired twin of the best | 0.500 |" in text
    assert "| sign-shuffled twin of the best | 0.490 |" in text
    sec_c = text.split("## Stage C: per-slot decodability of the best setting")[1]
    assert [ln.split(" | ")[0][2:] for ln in sec_c.splitlines()
            if ln.split(" | ")[0][2:] in SLOTS] == list(SLOTS)


def test_verdict_gates_on_the_confirmation_set_not_the_selection_score():
    a = [{"key": k, "passed": p} for k, p in (("k1", True), ("k2", True), ("k3", False))]
    b = [_row("k1", "nose", 0.70, True), _row("k2", "eyes", 0.64, True),
         _row("k3", "eyes", 0.90, False)]  # k3 failed Stage A: never the best, whatever its score

    # a selection-inflated 0.70 that the independent confirmation set scores at 0.58: FAIL
    v = G.verdict(a, b, _stage_c(b[0], 0.58), {"heldout": 0.77}, 0.84)
    assert v["best"]["key"] == "k1" and v["selection_heldout"] == 0.70
    assert v["confirmed"] is True and v["pass"] is False
    assert v["confirmation"]["heldout"] == 0.58
    assert G.verdict_line(v) == "VERDICT: FAIL (confirmation held-out 0.580; " \
                                "selected from 2 probed at 0.700)"

    # confirmation 0.61 on a matching Stage C: PASS, and every control is a confirmation number
    c = _stage_c(b[0], 0.61, rewired=0.55, sign_shuffled=0.52)
    v = G.verdict(a, b, c, {"heldout": 0.77}, 0.84)
    assert v["pass"] is True and v["confirmed"] is True
    assert v["controls_set"] == "confirmation"
    assert v["controls"] == {"one_hot": c["one_hot_confirmation"],
                             "bayes_ceiling": c["bayes_confirmation"],
                             "rewired": c["rewired"], "sign_shuffled": c["sign_shuffled"]}
    assert G.verdict_line(v) == "VERDICT: PASS (confirmation held-out 0.610; " \
                                "selected from 2 probed at 0.700)"

    # no Stage C: not a pass, whatever the selection score says; the controls are the selection's
    v = G.verdict(a, b, {}, {"heldout": 0.77}, 0.84)
    assert v["pass"] is False and v["confirmed"] is False and v["confirmation"] is None
    assert v["controls_set"] == "selection"
    assert v["controls"] == {"one_hot": {"heldout": 0.77}, "bayes_ceiling": 0.84}
    assert G.verdict_line(v) == ("VERDICT: PENDING (selected from 2 probed at 0.700; "
                                 "run --stage c to confirm)")

    # eligibility stays Stage A's: a selection score under the gate is still the candidate,
    # and only its confirmation decides
    b[0]["probe"]["heldout"] = 0.58
    b[1]["probe"]["heldout"] = 0.55
    v = G.verdict(a, b, {}, {"heldout": 0.77}, 0.84)
    assert v["best"]["key"] == "k1" and v["pass"] is False
    assert G.verdict_line(v).startswith("VERDICT: PENDING (")
    assert G.verdict(a, b, _stage_c(b[0], 0.61), {"heldout": 0.77}, 0.84)["pass"] is True
    assert G.verdict(a, b, _stage_c(b[0], 0.59), {"heldout": 0.77}, 0.84)["pass"] is False

    # an ineligible setting (failed Stage A) at selection 0.58 never passes, even with a
    # Stage C of its own that confirms at 0.61: there is no eligible best to confirm
    only = [_row("k4", "nose", 0.58, False)]
    v = G.verdict([{"key": "k4", "passed": False}], only, _stage_c(only[0], 0.61),
                  {"heldout": 0.77}, 0.84)
    assert v["best"] is None and v["pass"] is False and v["confirmed"] is False
    assert G.verdict_line(v) == "VERDICT: FAIL (no eligible setting)"


def test_pre_confirmation_verdict_json_reads_as_pending_not_pass(tmp_path):
    """A verdict.json written before Task 11b has `pass` computed from the selection score
    alone (`best["probe"]["heldout"] >= GATE`) and carries no `confirmed`/`confirmation` key.
    Both `verdict_status` and `fly report` (which trusts verdict.json as written, never
    recomputing it) must read that shape as PENDING, never as the selection-inflated PASS
    Task 11b exists to remove."""
    best = _row("k1", "nose", 0.63, True)
    pre_11b_verdict = {
        "gate": G.GATE,
        "pass": True,  # pre-11b: best["probe"]["heldout"] >= GATE, no confirmation involved
        "best": best,
        "stage_a": {"screened": 1, "passed": 1},
        "stage_b_probed": 1,
        "dropped_by_cap": 0,
        "controls": {"one_hot": {"heldout": 0.77}, "bayes_ceiling": 0.84},
        "context": "ctx1",
    }
    assert "confirmed" not in pre_11b_verdict and "confirmation" not in pre_11b_verdict
    assert G.verdict_status(pre_11b_verdict) == "PENDING"

    (tmp_path / "stage_a.jsonl").write_text(json.dumps(
        {"key": "k1", "context": "ctx1", "passed": True, "setting": best["setting"]}) + "\n")
    (tmp_path / "stage_b.jsonl").write_text(json.dumps(best) + "\n")
    (tmp_path / "verdict.json").write_text(json.dumps(pre_11b_verdict))
    md = tmp_path / "PHASE0.md"
    R.write_report(tmp_path, md)
    text = md.read_text()
    assert "**Verdict: PENDING**" in text
    assert "**Verdict: PASS**" not in text


def test_stage_c_matches_requires_this_runs_confirmation_seed():
    best = _row("k1", "nose", 0.7, True)
    c = _stage_c(best, 0.65)
    assert c["confirm_seed"] == 1000 and G.CONFIRM_SEED_OFFSET == 1000
    assert G.stage_c_matches(c, best) is True  # seed 0: the only seed the CLI runs
    assert G.stage_c_matches(c, best, seed=0) is True
    assert G.stage_c_matches(c, best, seed=5) is False  # seed 5 confirms on 1005
    assert G.stage_c_matches(_stage_c(best, 0.65, seed=5), best, seed=5) is True
    assert G.stage_c_matches({**c, "confirm_seed": 1001}, best) is False
    assert G.stage_c_matches({k: v for k, v in c.items() if k != "confirm_seed"}, best) is False
    # the verdict takes the run's seed and carries it for the report
    a = [{"key": "k1", "passed": True}]
    assert G.verdict(a, [best], c, {"heldout": 0.77}, 0.84, seed=5)["confirmed"] is False
    v = G.verdict(a, [best], _stage_c(best, 0.65, seed=5), {"heldout": 0.77}, 0.84, seed=5)
    assert v["confirmed"] is True and v["seed"] == 5 and v["confirm_seed"] == 1005


def test_context_fingerprint_ignores_the_confirmation_set(tmp_path):
    ctx = _toy_context(tmp_path, feedforward=True, confirm=True)
    assert ctx.confirm_set is not None
    bare = dataclasses.replace(ctx, confirm_set=None)
    other = dataclasses.replace(ctx, confirm_set=T.make_probe_set(ctx.catalog, n_pairs=300,
                                                                  seed=7))
    fps = {G.context_fingerprint(c) for c in (ctx, bare, other)}
    # unchanged from before the confirmation set existed: the running grid's records stay current
    assert fps == {TOY_FINGERPRINT}
    assert G.context_fingerprint(_toy_context(tmp_path, feedforward=True)) == TOY_FINGERPRINT


def test_confirmation_set_is_a_fresh_instance_of_the_task(tmp_path):
    ctx = _toy_context(tmp_path, feedforward=True, confirm=True)
    sel, conf = ctx.probe_set, ctx.confirm_set
    assert len(conf.y) == len(sel.y) == 900

    def pairs(ps):
        return {(ps.looks[a], ps.looks[b]) for a, b in zip(ps.a_idx, ps.b_idx, strict=True)}

    assert pairs(sel) != pairs(conf)  # fresh pairs and families
    # a fresh planted taste: make_probe_set seeds theta from seed + 1, so 1 vs 1001
    taste_sel, taste_conf = T.plant_taste(ctx.catalog, 1), T.plant_taste(ctx.catalog, 1001)
    for ps, taste in ((sel, taste_sel), (conf, taste_conf)):  # each set is labelled by its own
        du = np.array([taste.utility(ps.looks[a]) - taste.utility(ps.looks[b])
                       for a, b in zip(ps.a_idx, ps.b_idx, strict=True)])
        np.testing.assert_allclose(ps.p, 1.0 / (1.0 + np.exp(-taste.k * du)))
    shared = sel.looks[0]
    assert taste_sel.utility(shared) != taste_conf.utility(shared)


def test_stage_c_scores_the_best_setting_and_its_twins_on_the_confirmation_set(tmp_path,
                                                                              monkeypatch):
    ctx = _toy_context(tmp_path, feedforward=True, confirm=True)
    # the learnable toy's setting (its Stage B score is covered above); Stage C needs only this
    s = G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.0, steps=8, burn_in=5),
                  "all-sensory-brain", 1.0)
    best = {"key": s.key(), "context": G.context_fingerprint(ctx), "passed": True,
            "setting": s.as_dict(), "probe": _pr(0.75)}

    evaluated, featured = [], []
    real_evaluate, real_features, real_fit_bt = G.P.evaluate, G.features_for, G.P.fit_bt
    # one L2 strength instead of seven: the same fit and bootstrap, 7x cheaper on the CPU
    monkeypatch.setattr(G.P, "fit_bt", lambda D, y, groups, device: real_fit_bt(
        D, y, groups, device, lambdas=(1.0,)))

    def evaluate(X, ps, **kw):
        evaluated.append(ps)
        return real_evaluate(X, ps, **kw)

    def features(ctx_, sim, setting, looks, *args, **kw):
        featured.append(looks)
        return real_features(ctx_, sim, setting, looks, *args, **kw)

    monkeypatch.setattr(G.P, "evaluate", evaluate)
    monkeypatch.setattr(G, "features_for", features)
    out = tmp_path / "c.json"
    c = G.stage_c(ctx, best, out)
    assert json.loads(out.read_text()) == json.loads(json.dumps(c))
    assert c["key"] == best["key"] and c["context"] == best["context"]
    assert c["confirm_seed"] == 1000 == ctx.seed + G.CONFIRM_SEED_OFFSET
    assert G.stage_c_matches(c, best)
    conf = ctx.confirm_set
    # every probe score in Stage C is on the confirmation set: best, twins and one-hot alike
    assert len(evaluated) == 4 and all(ps is conf for ps in evaluated)
    # the real graph and both twins see the confirmation looks; decodability stays on selection
    assert sum(looks is conf.looks for looks in featured) == 3
    assert sum(looks is ctx.probe_set.looks for looks in featured) == 1
    for name in ("confirmation", "one_hot_confirmation", "rewired", "sign_shuffled"):
        assert set(c[name]) >= {"heldout", "ci_lo", "ci_hi", "train_acc", "cv_acc", "lam",
                                "n_test"}
        assert c[name]["n_test"] == int(conf.test.sum())
    assert c["bayes_confirmation"] == T.bayes_ceiling(conf.p[conf.test])
    assert c["confirmation"]["heldout"] > 0.6  # the learnable toy learns the fresh task too
    assert list(c["decodability"]) == list(SLOTS)
    assert isinstance(c["rewire_repairs"], int)

    with pytest.raises(ValueError, match="confirm"):
        G.stage_c(dataclasses.replace(ctx, confirm_set=None), best, tmp_path / "never.json")
    assert not (tmp_path / "never.json").exists()


def test_default_grid_covers_every_brain_and_code():
    grid = G.default_grid()
    assert {s.brain.kind for s in grid} == {"lif", "lif-avg", "lif-volley", "rate"}
    assert {s.code for s in grid} == {"nose", "eyes", "eyes+nose", "all-sensory-brain",
                                      "all-sensory+vnc"}
    assert len({s.key() for s in grid}) == len(grid)
    json.dumps([s.as_dict() for s in grid])


def _a_rec(kind, code, g_syn, d_min, context="ctx1", activity=True, senses=True):
    """A Stage A record with slot / full distances 5 / 9: smooth iff d_min <= 0.5."""
    extra = {"steps": 6} if kind == "lif-volley" else {}
    s = G.Setting(BrainParams(kind=kind, g_syn=float(g_syn), bias=0.0, **extra), code, 2.0)
    smooth = {"min": d_min, "slot": 5.0, "full": 9.0, "ok": P.smooth_ok(d_min, 5.0, 9.0)}
    return {"key": s.key(), "context": context, "setting": s.as_dict(), "smooth": smooth,
            "activity_ok": activity, "senses": {"nose": senses},
            "passed": bool(activity and smooth["ok"] and senses)}


def _strata(recs):
    return [(r["setting"]["brain"]["kind"], r["setting"]["code"]) for r in recs]


def _tied_and_others(context="ctx1"):
    """The real-graph shape: many deterministic spiking passers whose +0.01 flips no
    spike (d_min == 0, so d_min/d_slot ties at 0), listed first as grid order does."""
    tied = [_a_rec("lif-volley", "nose", g, 0.0, context) for g in range(1, 31)]
    others = ([_a_rec("lif-volley", "all-sensory+vnc", g, 0.0, context) for g in (1, 2)]
              + [_a_rec("rate", "nose", g, 0.2, context) for g in (1, 2)]
              + [_a_rec("rate", "all-sensory+vnc", g, 0.3, context) for g in (1, 2)])
    failed = [_a_rec("rate", "eyes", 1, 0.0, context, activity=False)]
    return tied, others, failed


def test_stage_b_probes_every_passer_and_zero_ties_never_crowd_out_a_cap():
    tied, others, failed = _tied_and_others()
    passers = {r["key"] for r in tied + others}
    chosen, dropped = G.stage_b_selection(tied + others + failed)
    assert {r["key"] for r in chosen} == passers and dropped == 0  # spec §3.0: every passer
    # the old smoothest-first cap of 4 kept four tied nose lif-volley settings: no rate, no
    # all-sensory. A stratified cap takes every (kind, code) once before any twice.
    chosen, dropped = G.stage_b_selection(tied + others + failed, cap=4)
    assert _strata(chosen) == [("lif-volley", "nose"), ("lif-volley", "all-sensory+vnc"),
                               ("rate", "nose"), ("rate", "all-sensory+vnc")]
    assert dropped == len(passers) - 4
    chosen, _ = G.stage_b_selection(tied + others + failed, cap=8)
    assert sorted(set(_strata(chosen))) == sorted(set(_strata(tied + others)))
    assert _strata(chosen).count(("lif-volley", "nose")) == 2  # round-robin, not 5 ties first
    with pytest.raises(ValueError):
        G.stage_b_selection(tied, cap=0)


def test_stage_b_fallback_prefers_fewest_failed_checks_one_per_stratum():
    # silent spiking settings: d_min == 0 is "smooth", but no activity and no sense change
    silent = [_a_rec("lif", "nose", g, 0.0, activity=False, senses=False) for g in range(1, 21)]
    rough = [_a_rec("rate", code, 1.0, 0.9) for code in CODE_NAMES]  # fail smoothness only
    records = silent + rough
    assert not any(r["passed"] for r in records)
    chosen, dropped = G.stage_b_selection(records)
    assert len(chosen) == G.FALLBACK_PROBES and dropped == 0
    assert _strata(chosen) == [("rate", c) for c in CODE_NAMES] + [("lif", "nose")]
    assert chosen[-1]["key"] == silent[0]["key"]


def _fake_probe(monkeypatch, good=("rate", "all-sensory+vnc")):
    """Stage B's GPU work replaced: `good` settings score 0.75 held-out, the rest 0.52."""
    seen = []

    def features(ctx, sim, setting, looks, images_cache=None, **kw):
        seen.append(setting)
        stats = {"active_frac": 0.2, "readout_active_frac": 0.2, "max_step_frac": 0.05}
        return np.zeros((len(looks), 1), np.float32), stats

    def evaluate(X, ps, device="cpu", seed=0):
        h = 0.75 if (seen[-1].brain.kind, seen[-1].code) == good else 0.52
        return SimpleNamespace(as_dict=lambda: {"heldout": h, "ci_lo": h - 0.03,
                                                "ci_hi": h + 0.03})

    monkeypatch.setattr(G, "features_for", features)
    monkeypatch.setattr(G.P, "evaluate", evaluate)
    monkeypatch.setattr(G.P, "per_slot_decodability", lambda *a, **k: _dec(0.5))


def _write_results(d, a, b, v, c=None):
    d.mkdir()
    (d / "stage_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a))
    (d / "stage_b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in b))
    (d / "verdict.json").write_text(json.dumps(v))
    if c is not None:
        (d / "stage_c.json").write_text(json.dumps(c))


def test_uncapped_stage_b_finds_the_passing_setting_and_a_cap_is_disclosed(tmp_path,
                                                                          monkeypatch):
    ctx = _toy_context(tmp_path, feedforward=True)
    context = G.context_fingerprint(ctx)
    tied, others, failed = _tied_and_others(context)
    a = tied + others + failed
    _fake_probe(monkeypatch)

    b = G.stage_b(ctx, a, tmp_path / "b.jsonl", sim=object())  # default: no cap
    assert {r["key"] for r in b} == {r["key"] for r in tied + others}
    v = {**G.verdict(a, b, {}, {"heldout": 0.77}, 0.84), "context": context}
    assert v["pass"] is False and v["dropped_by_cap"] == 0  # found, but not yet confirmed
    assert _strata([v["best"]]) == [("rate", "all-sensory+vnc")]
    assert G.verdict_line(v) == ("VERDICT: PENDING (selected from 36 probed at 0.750; "
                                 "run --stage c to confirm)")
    c = _stage_c(v["best"], 0.74)
    v = {**G.verdict(a, b, c, {"heldout": 0.77}, 0.84), "context": context}
    assert v["pass"] is True and v["dropped_by_cap"] == 0
    assert G.verdict_line(v) == ("VERDICT: PASS (confirmation held-out 0.740; "
                                 "selected from 36 probed at 0.750)")
    _write_results(tmp_path / "full", a, b, v, c)
    R.write_report(tmp_path / "full", tmp_path / "full.md")
    text = (tmp_path / "full.md").read_text()
    assert "**Verdict: PASS**" in text and "Incomplete" not in text
    assert "| all-sensory+vnc | rate |" in text  # the all-sensory code is in the decodability table

    # a budget cap: the gate never sees the passing setting, and everything says so
    capped = G.stage_b(ctx, a, tmp_path / "b_cap.jsonl", cap=2, sim=object())
    assert len(capped) == 2
    v = {**G.verdict(a, capped, {}, {"heldout": 0.77}, 0.84), "context": context}
    assert v["pass"] is False and v["dropped_by_cap"] == len(tied + others) - 2
    assert json.loads(json.dumps(v))["dropped_by_cap"] == 34  # lands in verdict.json
    assert G.verdict_line(v, cap=2) == (
        "VERDICT: PENDING (selected from 2 probed at 0.520; 34 Stage A passer(s) not probed in "
        "Stage B (--cap 2); run --stage c to confirm)")
    c = _stage_c(v["best"], 0.53)
    v = {**G.verdict(a, capped, c, {"heldout": 0.77}, 0.84), "context": context}
    line = G.verdict_line(v, cap=2)
    assert line.startswith("VERDICT: FAIL (confirmation held-out 0.530; selected from 2 probed "
                           "at 0.520; 34 Stage A passer(s) not probed in Stage B (--cap 2); "
                           "so this FAIL is not the protocol's")
    # a later `--stage c` run (no --cap given) still discloses the unprobed passers
    assert "34 Stage A passer(s) not probed" in G.verdict_line(v)
    _write_results(tmp_path / "capped", a, capped, v, c)
    R.write_report(tmp_path / "capped", tmp_path / "capped.md")
    text = (tmp_path / "capped.md").read_text()
    assert ("**Incomplete:** 34 setting(s) that passed Stage A were never probed in Stage B "
            "(a `--cap` was used, or Stage B did not finish), so this FAIL is not the protocol's "
            "verdict.") in text


def test_verdict_line_names_a_cap_even_when_it_dropped_nothing():
    v = {"pass": True, "confirmed": True, "best": {"probe": {"heldout": 0.7}},
         "selection_heldout": 0.7, "selection_pool": 5, "confirmation": {"heldout": 0.66},
         "dropped_by_cap": 0}
    assert G.verdict_line(v, cap=200) == ("VERDICT: PASS (confirmation held-out 0.660; "
                                          "selected from 5 probed at 0.700; "
                                          "0 Stage A passer(s) not probed in Stage B (--cap 200))")
    assert G.verdict_line({"pass": False, "best": None}) == "VERDICT: FAIL (no eligible setting)"


def test_grid_cli_probes_every_passer_by_default():
    from lfg_fly.cli import build_parser

    assert build_parser().parse_args(["grid"]).cap is None
    assert build_parser().parse_args(["grid", "--cap", "24"]).cap == 24


def _b_row(setting: G.Setting, heldout: float, context: str = "ctx1", dropped: int = 0) -> dict:
    """A Stage B row carrying the full BrainParams dict, as the real grid writes it."""
    return {"key": setting.key(), "context": context, "passed": True, "dropped_by_cap": dropped,
            "setting": setting.as_dict(), "probe": _pr(heldout), "decodability": _dec(heldout)}


def test_report_gives_the_confirmation_headline_and_the_full_setting_identity(tmp_path):
    volley6 = G.Setting(BrainParams(kind="lif-volley", g_syn=6.0, bias=0.05, steps=6),
                        "nose", 2.0)
    volley10 = G.Setting(BrainParams(kind="lif-volley", g_syn=6.0, bias=0.05, steps=10),
                         "nose", 2.0)
    avg = G.Setting(BrainParams(kind="lif-avg", g_syn=4.0, steps=60, trials=8, noise=0.05),
                    "eyes", 0.5)
    # a stale per-row dropped_by_cap: only verdict.json's count is the verdict's
    b = [_b_row(volley6, 0.702, dropped=99), _b_row(volley10, 0.753, dropped=99),
         _b_row(avg, 0.611, dropped=99)]
    a = [{"key": r["key"], "context": "ctx1", "passed": True, "setting": r["setting"]} for r in b]
    c = {**_stage_c(b[1], 0.741, rewired=0.5, sign_shuffled=0.49),
         "one_hot_confirmation": _pr(0.761), "bayes_confirmation": 0.829}
    v = {**G.verdict(a, b, c, {"heldout": 0.77}, 0.84), "context": "ctx1"}
    assert v["dropped_by_cap"] == 0
    d = tmp_path / "probe"
    _write_results(d, a, b, v, c)
    md = tmp_path / "PHASE0.md"
    R.write_report(d, md)
    text = md.read_text()
    assert "**Verdict: PASS**" in text and "Incomplete" not in text
    head = text.split("## Stage A")[0]
    assert "| Reference (confirmation set) | Held-out |" in head
    assert "| Bayes ceiling | 0.829 |" in head
    assert "| One-hot, no brain | 0.761 |" in head
    assert "| **Best fly setting (confirmation)** | **0.741** (0.701–0.781) |" in head
    assert "| rewired twin of the best | 0.500 |" in head
    assert "| sign-shuffled twin of the best | 0.490 |" in head
    assert "0.840" not in head and "0.770" not in head  # the selection-set controls are gone
    assert "selection (max over 3 probed; optimistic, not the gate)" in head
    assert "0.753" in head.split("selection (max over")[0].splitlines()[-1]
    # the best setting's identity, in full: T=6 and T=10 lif-volley are different settings
    assert "| kind | code | steps | trials | noise | g_syn | bias | g_in |" in head
    assert "| lif-volley | nose | 10 | 1 | 0.0 | 6.0 | 0.05 | 2.0 |" in head
    sec_b = text.split("## Stage B: planted-taste probe")[1].split("\n## ")[0]
    assert "| Brain | Code | steps | trials | g_syn | bias | g_in | eligible | held-out |" in sec_b
    assert "| lif-volley | nose | 10 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.753 |" in sec_b
    assert "| lif-volley | nose | 6 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.702 |" in sec_b
    assert "| lif-avg | eyes | 60 | 8 | 4.0 | 0.0 | 0.5 | yes | 0.611 |" in sec_b
    # how to read the checks: before the attribution, with each caveat
    assert text.index("## How to read the checks") < text.index(R.ATTRIBUTION)
    checks = text.split("## How to read the checks")[1].split("---")[0]
    for needle in ("distance 0", "× 1.01", "`eyes+nose` perturbs only the nose", "one `g_in`",
                   "any nonzero readout change", "batch position", "confirmation set",
                   "independent planted taste", "seed 1000"):
        assert needle in checks, needle

    # no confirmation yet: the report says PENDING and shows no confirmation-set number
    (d / "stage_c.json").unlink()
    v = {**G.verdict(a, b, {}, {"heldout": 0.77}, 0.84), "context": "ctx1"}
    (d / "verdict.json").write_text(json.dumps(v))
    R.write_report(d, md)
    text = md.read_text()
    head = text.split("## Stage A")[0]
    assert "**Verdict: PENDING**" in text
    assert "| Reference (selection set) | Held-out |" in head
    assert "| Bayes ceiling | 0.840 |" in head and "| One-hot, no brain | 0.770 |" in head
    assert "| **Best fly setting (confirmation)** | PENDING |" in head
    assert "twin of the best" not in head and "0.741" not in text
    assert "fly grid --stage c" in head
    assert "| lif-volley | nose | 10 | 1 | 0.0 | 6.0 | 0.05 | 2.0 |" in head
    assert "## Stage C" not in text


def test_cli_context_carries_the_confirmation_set(tmp_path, monkeypatch):
    from lfg_fly import cli, paths
    from lfg_fly.brain import senses
    from lfg_fly.connectome import columns
    from lfg_fly.teacher import render

    toy = _toy_context(tmp_path, feedforward=True)
    cache = paths.network_dir("mainnet") / "catalog"
    cache.mkdir(parents=True)
    (cache / "catalog-male.json").write_text(toy.catalog.to_json())
    monkeypatch.setattr(B, "load_graph", lambda path: toy.graph)
    monkeypatch.setattr(columns, "load_columns", lambda path: None)
    monkeypatch.setattr(senses, "build_retina", lambda cols, g: toy.retina)
    monkeypatch.setattr(render, "LayerBank", lambda *a, **k: None)
    monkeypatch.setattr(render, "load_zorder", lambda path: toy.zorder)
    ctx = cli._load_context(SimpleNamespace(min_syn=3, device="cpu", pairs=900))
    want = T.make_probe_set(toy.catalog, n_pairs=900, seed=1000)
    assert ctx.seed == 0 and len(ctx.confirm_set.y) == 900
    assert ctx.confirm_set.looks == want.looks
    np.testing.assert_array_equal(ctx.confirm_set.y, want.y)
    np.testing.assert_array_equal(ctx.confirm_set.p, want.p)
    assert ctx.probe_set.looks == toy.probe_set.looks  # the selection set is untouched
    # and so is the context the running grid's records are keyed by
    assert G.context_fingerprint(ctx) == G.context_fingerprint(toy) == TOY_FINGERPRINT


def test_grid_cli_reruns_stage_c_when_it_is_not_this_confirmation_set(tmp_path, monkeypatch,
                                                                      capsys):
    from lfg_fly import cli, env, paths

    ctx = _toy_context(tmp_path, feedforward=True, confirm=True)
    fp = G.context_fingerprint(ctx)
    out = tmp_path / "repo" / "data" / "probe"
    out.mkdir(parents=True)
    best = _row("k1", "nose", 0.70, True, context=fp)
    (out / "stage_a.jsonl").write_text(json.dumps({"key": "k1", "context": fp, "passed": True,
                                                   "setting": best["setting"]}) + "\n")
    (out / "stage_b.jsonl").write_text(json.dumps(best) + "\n")
    before_11b = {k: v for k, v in _stage_c(best, 0.66).items()
                  if k not in ("confirm_seed", "confirmation", "one_hot_confirmation",
                               "bayes_confirmation")}  # twins scored on the selection set
    (out / "stage_c.json").write_text(json.dumps(before_11b))
    calls = []

    def stage_c(ctx_, best_, path):
        calls.append(best_["key"])
        c = _stage_c(best_, 0.66)
        path.write_text(json.dumps(c))
        return c

    monkeypatch.setattr(cli, "_load_context", lambda args: ctx)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(env, "configure_libraries", lambda: None)
    monkeypatch.setattr(G, "stage_c", stage_c)

    assert cli.main(["grid", "--stage", "b", "--device", "cpu", "--pairs", "900"]) == 0
    assert calls == []  # --stage b never runs Stage C
    v = json.loads((out / "verdict.json").read_text())
    assert v["confirmed"] is False and v["pass"] is False and v["controls_set"] == "selection"
    assert "VERDICT: PENDING (selected from 1 probed at 0.700; run --stage c to confirm)" \
        in capsys.readouterr().out

    assert cli.main(["grid", "--stage", "c", "--device", "cpu", "--pairs", "900"]) == 0
    assert calls == ["k1"]  # the pre-11b stage_c.json was not reused
    printed = capsys.readouterr().out
    assert "confirmation held-out 0.660" in printed
    assert "VERDICT: PASS (confirmation held-out 0.660; selected from 1 probed at 0.700)" \
        in printed
    v = json.loads((out / "verdict.json").read_text())
    assert v["confirmed"] is True and v["pass"] is True and v["controls_set"] == "confirmation"
    assert v["confirm_seed"] == 1000 and v["context"] == fp

    assert cli.main(["grid", "--stage", "c", "--device", "cpu", "--pairs", "900"]) == 0
    assert calls == ["k1"]  # this time it matches: reused
    printed = capsys.readouterr().out
    assert "reused" in printed and "confirmation held-out 0.660" in printed
