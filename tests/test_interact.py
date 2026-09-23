import dataclasses
import json

import numpy as np
import pytest
from PIL import Image
from test_grid import _toy_context

from lfg_fly.brain.senses import SLOTS
from lfg_fly.brain.sim import BrainParams
from lfg_fly.teacher import catalog as K
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import interact as I
from lfg_fly.teacher import probe as P
from lfg_fly.teacher import render as R
from lfg_fly.teacher import tastes as S


def _toy_bank(tmp_path, cat, size=8):
    """Random-colour layers: an opaque Background, and a 6x6 patch for every other slot."""
    rng = np.random.default_rng(0)
    root = tmp_path / "layers"
    for slot in SLOTS:
        for value in cat.values[slot]:
            if value == "None":
                continue
            rgba = (*(int(c) for c in rng.integers(0, 256, 3)), 255)
            if slot == "Background":
                im = Image.new("RGBA", (16, 16), rgba)
            else:
                im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
                x, y = (int(v) for v in rng.integers(0, 10, 2))
                im.paste(Image.new("RGBA", (6, 6), rgba), (x, y))
            path = K.layer_cache_path(root, "male", slot, value)
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(path, format="PNG")
    return R.LayerBank(cat, root, size=size)


def _ctx(tmp_path):
    ctx = _toy_context(tmp_path, feedforward=True, confirm=True)
    return dataclasses.replace(ctx, bank=_toy_bank(tmp_path, ctx.catalog))


def _fast(monkeypatch):
    """One L2 strength for the linear fits and a small MLP: the same protocol, cheaper."""
    real_fit_bt, real_mlp = P.fit_bt, P.evaluate_mlp
    monkeypatch.setattr(P, "fit_bt", lambda D, y, groups, device: real_fit_bt(
        D, y, groups, device, lambdas=(1.0,)))
    monkeypatch.setattr(P, "evaluate_mlp", lambda X, ps, device="cpu", seed=0: real_mlp(
        X, ps, device=device, seed=seed, hidden=8, lambdas=(1.0,), folds=2))


SETTINGS = [G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.0, steps=8, burn_in=5),
                      "all-sensory-brain", 1.0),
            G.Setting(BrainParams(kind="rate", g_syn=0.5, bias=0.0, steps=8, burn_in=5),
                      "all-sensory-brain", 1.0)]


def _stage_a(ctx):
    return [{"key": s.key(), "context": G.context_fingerprint(ctx), "setting": s.as_dict(),
             "passed": True, "activity_ok": True, "smooth": {"ok": True}, "senses": {}}
            for s in SETTINGS]


def test_taste_sets_relabel_phase0s_selection_and_confirmation_sets(tmp_path):
    ctx = _ctx(tmp_path)
    sets = I.taste_sets(ctx)
    assert set(sets.selection) == set(sets.confirmation) == set(S.TASTES)
    assert np.array_equal(sets.selection["additive"].ps.y, ctx.probe_set.y)
    assert np.array_equal(sets.confirmation["additive"].ps.y, ctx.confirm_set.y)
    for taste in S.TASTES:
        assert sets.selection[taste].ps.looks is ctx.probe_set.looks
        assert sets.confirmation[taste].ps.looks is ctx.confirm_set.looks
    assert sets.pixels["selection"].shape == (len(ctx.probe_set.looks), 11)
    with pytest.raises(ValueError, match="bank"):
        I.taste_sets(dataclasses.replace(ctx, bank=None))


def test_calibration_gives_ceiling_oracle_gap_and_no_brain_controls(tmp_path, monkeypatch):
    _fast(monkeypatch)
    ctx = _ctx(tmp_path)
    out = tmp_path / "calibration.json"
    cal = I.calibrate(ctx, I.taste_sets(ctx), out)
    assert json.loads(out.read_text()) == json.loads(json.dumps(cal))
    assert cal["context"] == I.interact_fingerprint(ctx)
    for set_name in ("selection", "confirmation"):
        for taste in S.TASTES:
            row = cal["sets"][set_name][taste]
            assert row["gap"] == pytest.approx(row["bayes"] - row["additive_oracle"])
            for control in ("one_hot", "one_hot_pixels", "mlp"):
                assert 0.0 <= row[control]["heldout"] <= 1.0
        assert cal["sets"][set_name]["additive"]["gap"] == pytest.approx(0.0, abs=1e-9)
        assert cal["sets"][set_name]["additive"]["too_weak"] is False


def test_stage_b_is_blocked_by_a_missing_stale_or_weak_calibration():
    ok = {"context": "c1", "sets": {s: {t: {"too_weak": False} for t in S.TASTES}
                                    for s in ("selection", "confirmation")}}
    assert I.calibration_blocks(ok, "c1") is None
    assert "calib" in I.calibration_blocks({}, "c1")
    assert "another context" in I.calibration_blocks(ok, "c2")
    weak = json.loads(json.dumps(ok))
    weak["sets"]["confirmation"]["visual"]["too_weak"] = True
    assert "visual" in I.calibration_blocks(weak, "c1")


def test_interact_fingerprint_extends_phase0s_context(tmp_path):
    ctx = _ctx(tmp_path)
    fp = I.interact_fingerprint(ctx)
    assert fp != G.context_fingerprint(ctx)
    assert fp != I.interact_fingerprint(dataclasses.replace(ctx, seed=1))


def test_stage_b_reproduces_phase0_and_stops_when_it_does_not(tmp_path, monkeypatch):
    _fast(monkeypatch)  # Phase 0's Stage B below runs under the same patch
    ctx = _ctx(tmp_path)
    a = _stage_a(ctx)
    phase0 = G.stage_b(ctx, a, tmp_path / "p0" / "stage_b.jsonl")
    sets = I.taste_sets(ctx)
    out = tmp_path / "stage_b.jsonl"
    rows = I.stage_b(ctx, sets, a, phase0, out)
    assert len(rows) == 2
    p0 = {r["key"]: r["probe"]["heldout"] for r in phase0}
    for row in rows:
        assert set(row["probe"]) == set(S.TASTES)
        assert row["probe"]["additive"]["heldout"] == p0[row["key"]]
        assert row["phase0_additive"] == p0[row["key"]]
        assert row["context"] == I.interact_fingerprint(ctx)
    # the best Phase 0 setting goes first
    assert rows[0]["key"] == max(phase0, key=lambda r: r["probe"]["heldout"])["key"]
    # resumable: nothing is simulated again
    again = I.stage_b(ctx, sets, a, phase0, out, sim=object())
    assert [r["key"] for r in again] == [r["key"] for r in rows]

    off = [json.loads(json.dumps(r)) for r in phase0]
    off[0]["probe"]["heldout"] += 2 * I.REPRO_TOL
    bad = tmp_path / "bad.jsonl"
    with pytest.raises(I.ReproductionError, match="Phase 0"):
        I.stage_b(ctx, sets, a, off, bad)
    first = max(off, key=lambda r: r["probe"]["heldout"])["key"]
    assert first not in {r["key"] for r in G.read_jsonl(bad)}


def test_stage_b_limit_probes_the_best_phase0_setting_only(tmp_path, monkeypatch):
    _fast(monkeypatch)
    ctx = _ctx(tmp_path)
    a = _stage_a(ctx)
    phase0 = [{"key": SETTINGS[0].key(), "probe": {"heldout": 0.1}},
              {"key": SETTINGS[1].key(), "probe": {"heldout": 0.9}}]
    sets = I.taste_sets(ctx)
    out = tmp_path / "stage_b.jsonl"
    with pytest.raises(I.ReproductionError):  # the toy doesn't score 0.9: proof it ran first
        I.stage_b(ctx, sets, a, phase0, out, limit=1)
    rows = I.stage_b(ctx, sets, a, [], out, limit=1)
    assert [r["key"] for r in rows] == [SETTINGS[0].key()]  # no Phase 0 scores: grid order
    assert I.unprobed(a, rows) == 1 and I.unprobed(a, I.stage_b(ctx, sets, a, [], out)) == 0


def test_readings_follow_the_preregistered_rules():
    pos, neg = {"diff": 0.05, "lo": 0.01, "hi": 0.09}, {"diff": 0.02, "lo": -0.01, "hi": 0.05}
    r = I.readings({"ci_lo": 0.56}, {"one_hot": pos, "rewired": neg, "mlp": neg})
    assert r == {"learns": True, "uses_interactions": True, "wiring_matters": False,
                 "beats_no_brain": False}
    r = I.readings({"ci_lo": 0.55}, {"one_hot": neg, "rewired": pos, "mlp": pos})
    assert r == {"learns": False, "uses_interactions": False, "wiring_matters": True,
                 "beats_no_brain": True}


def test_stage_c_confirms_each_tastes_best_with_twins_controls_and_readings(tmp_path,
                                                                           monkeypatch):
    _fast(monkeypatch)
    ctx = _ctx(tmp_path)
    sets = I.taste_sets(ctx)
    a = _stage_a(ctx)
    rows = I.stage_b(ctx, sets, a, [], tmp_path / "stage_b.jsonl")
    # make the tastes disagree on the best setting
    rows[0]["probe"]["latent"]["heldout"], rows[1]["probe"]["latent"]["heldout"] = 0.9, 0.1
    rows[0]["probe"]["visual"]["heldout"], rows[1]["probe"]["visual"]["heldout"] = 0.1, 0.9

    featured = []
    real_features = G.features_for

    def features(ctx_, sim, setting, looks, *args, **kw):
        featured.append((setting.key(), looks is ctx.confirm_set.looks))
        return real_features(ctx_, sim, setting, looks, *args, **kw)

    monkeypatch.setattr(G, "features_for", features)
    out = tmp_path / "stage_c.json"
    c = I.stage_c(ctx, sets, rows, out)
    assert json.loads(out.read_text()) == json.loads(json.dumps(c))
    # each distinct best setting is simulated once per graph, on the confirmation looks only
    assert len(featured) == 3 * 2 and all(on_conf for _, on_conf in featured)
    assert c["tastes"]["latent"]["key"] == rows[0]["key"]
    assert c["tastes"]["visual"]["key"] == rows[1]["key"]
    n_test = int(ctx.confirm_set.test.sum())
    for taste in S.TASTES:
        t = c["tastes"][taste]
        for name in ("fly", "rewired", "sign_shuffled"):
            assert t[name]["n_test"] == n_test
        assert set(t["controls"]) == {"one_hot", "one_hot_pixels", "mlp"}
        assert set(t["paired"]) == {"one_hot", "one_hot_pixels", "mlp", "rewired",
                                    "sign_shuffled"}
        assert t["paired"]["rewired"]["diff"] == pytest.approx(
            t["fly"]["heldout"] - t["rewired"]["heldout"])
        assert t["readings"] == I.readings(t["fly"], t["paired"])
        assert t["gap"] == pytest.approx(t["bayes"] - t["additive_oracle"])
    assert isinstance(c["rewire_repairs"], int)


def _cli(tmp_path, monkeypatch, weak=False):
    """`fly interact` on the toy, with the three stages faked and recorded."""
    from lfg_fly import cli, env, paths
    from lfg_fly.teacher import interact_report as IR

    ctx = _ctx(tmp_path)
    fp = I.interact_fingerprint(ctx)
    repo = tmp_path / "repo"
    p0 = repo / "data" / "probe"
    p0.mkdir(parents=True)
    a = _stage_a(ctx)
    (p0 / "stage_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a))
    calls = []

    def calibrate(ctx_, sets, path):
        calls.append("calib")
        cal = {"context": fp, "sets": {s: {t: {"too_weak": weak and t == "visual"}
                                           for t in S.TASTES} for s in I.SETS}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cal))
        return cal

    def stage_b(ctx_, sets, a_, b0, path, limit=None):
        calls.append(("b", limit, len(a_)))
        rows = [{"key": r["key"], "context": fp, "passed": True} for r in a_][:limit]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rows

    def stage_c(ctx_, sets, rows, path):
        calls.append(("c", len(rows)))
        return {}

    monkeypatch.setattr(cli, "_load_context", lambda args: ctx)
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    monkeypatch.setattr(env, "configure_libraries", lambda: None)
    monkeypatch.setattr(I, "taste_sets", lambda ctx_: None)
    monkeypatch.setattr(I, "calibrate", calibrate)
    monkeypatch.setattr(I, "stage_b", stage_b)
    monkeypatch.setattr(I, "stage_c", stage_c)
    monkeypatch.setattr(IR, "calibration_lines", lambda cal: ["CALIBRATED"])
    monkeypatch.setattr(IR, "reading_lines", lambda c: ["READINGS"])
    return lambda *argv: cli.main(["interact", *argv, "--device", "cpu"]), calls


def test_interact_cli_runs_calibration_then_stage_b_then_stage_c(tmp_path, monkeypatch, capsys):
    run, calls = _cli(tmp_path, monkeypatch)
    assert run("--stage", "b") == 2 and calls == []
    assert "stage B refused: no calibration.json" in capsys.readouterr().out
    assert run("--stage", "calib") == 0 and calls == ["calib"]
    assert "CALIBRATED" in capsys.readouterr().out
    assert run("--stage", "b", "--limit", "1") == 0 and calls[-1] == ("b", 1, 2)
    assert "1 Stage A passer(s) not probed yet" in capsys.readouterr().out
    assert run("--stage", "c") == 2 and calls[-1] == ("b", 1, 2)  # C never sees a partial B
    assert "stage C refused" in capsys.readouterr().out
    assert run("--stage", "b") == 0 and calls[-1] == ("b", None, 2)
    assert run("--stage", "c") == 0 and calls[-1] == ("c", 2)
    assert "READINGS" in capsys.readouterr().out


def test_interact_cli_refuses_stage_b_on_a_weak_taste(tmp_path, monkeypatch, capsys):
    run, calls = _cli(tmp_path, monkeypatch, weak=True)
    assert run("--stage", "all") == 2 and calls == ["calib"]
    assert "too weak" in capsys.readouterr().out


def test_report_cli_writes_phase0b_once_its_results_exist(tmp_path, monkeypatch):
    from lfg_fly import cli, paths
    from lfg_fly.teacher import interact_report as IR
    from lfg_fly.teacher import report

    written = []
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(report, "write_report", lambda d, md: written.append(md.name))
    monkeypatch.setattr(IR, "write_report", lambda d, md: written.append((d.name, md.name)))
    assert cli.main(["report"]) == 0 and written == ["PHASE0.md"]
    (tmp_path / "data" / "probe-interact").mkdir(parents=True)
    (tmp_path / "data" / "probe-interact" / "stage_c.json").write_text("{}")
    assert cli.main(["report"]) == 0
    assert written[1:] == ["PHASE0.md", ("probe-interact", "PHASE0B.md")]
