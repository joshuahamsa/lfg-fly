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


def _toy_context(tmp_path, feedforward: bool):
    """ORNs -> readout. Feed-forward: a clean, learnable map. Otherwise: strong recurrence."""
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
    return G.Context(graph=g, pops=pops, retina=retina, catalog=cat, bank=None, zorder=z,
                     device="cpu", probe_set=ps, seed=0)


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
    twins = {"rewired": {"heldout": 0.55}, "sign_shuffled": {"heldout": 0.52}}
    ok = G.verdict(a, b, {"key": "k1", "context": "ctx1", **twins}, {"heldout": 0.77}, 0.84)
    assert set(ok["controls"]) == {"one_hot", "bayes_ceiling", "rewired", "sign_shuffled"}
    for stale in ({"key": "k0", "context": "ctx1", **twins},  # an earlier best's twins
                  {"key": "k1", "context": "ctx0", **twins}):  # same setting, other graph
        v = G.verdict(a, b, stale, {"heldout": 0.77}, 0.84)
        assert set(v["controls"]) == {"one_hot", "bayes_ceiling"}
    b[0]["passed"] = False  # no eligible setting: a leftover stage_c.json adds nothing
    v = G.verdict(a, b, {"key": "k1", "context": "ctx1", **twins}, {"heldout": 0.77}, 0.84)
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
    c = {"key": "k2", "context": "ctx1", "rewire_repairs": 3, "decodability": _dec(0.9),
         "rewired": {"heldout": 0.5}, "sign_shuffled": {"heldout": 0.49}}
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


def test_verdict_applies_gate():
    a = [{"key": "k1", "passed": True}]
    b = [{"key": "k1", "passed": True, "probe": {"heldout": 0.61, "ci_lo": 0.57, "ci_hi": 0.65}}]
    v = G.verdict(a, b, {"rewired": {"heldout": 0.55}}, {"heldout": 0.77}, 0.84)
    assert v["pass"] is True and v["best"]["key"] == "k1"
    b[0]["probe"]["heldout"] = 0.58
    assert G.verdict(a, b, {}, {"heldout": 0.77}, 0.84)["pass"] is False


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


def _write_results(d, a, b, v):
    d.mkdir()
    (d / "stage_a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in a))
    (d / "stage_b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in b))
    (d / "verdict.json").write_text(json.dumps(v))


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
    assert v["pass"] is True and v["dropped_by_cap"] == 0
    assert _strata([v["best"]]) == [("rate", "all-sensory+vnc")]
    assert G.verdict_line(v) == "VERDICT: PASS (best held-out 0.750)"
    _write_results(tmp_path / "full", a, b, v)
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
    line = G.verdict_line(v, cap=2)
    assert line.startswith("VERDICT: FAIL (best held-out 0.520; 34 Stage A passer(s) not "
                           "probed in Stage B (--cap 2); so this FAIL is not the protocol's")
    # a later `--stage c` run (no --cap given) still discloses the unprobed passers
    assert "34 Stage A passer(s) not probed" in G.verdict_line(v)
    _write_results(tmp_path / "capped", a, capped, v)
    R.write_report(tmp_path / "capped", tmp_path / "capped.md")
    text = (tmp_path / "capped.md").read_text()
    assert ("**Incomplete:** 34 setting(s) that passed Stage A were never probed in Stage B "
            "(a `--cap` was used, or Stage B did not finish), so this FAIL is not the protocol's "
            "verdict.") in text


def test_verdict_line_names_a_cap_even_when_it_dropped_nothing():
    v = {"pass": True, "best": {"probe": {"heldout": 0.7}}, "dropped_by_cap": 0}
    assert G.verdict_line(v, cap=200) == ("VERDICT: PASS (best held-out 0.700; "
                                          "0 Stage A passer(s) not probed in Stage B (--cap 200))")
    assert G.verdict_line({"pass": False, "best": None}) == "VERDICT: FAIL (no eligible setting)"


def test_grid_cli_probes_every_passer_by_default():
    from lfg_fly.cli import build_parser

    assert build_parser().parse_args(["grid"]).cap is None
    assert build_parser().parse_args(["grid", "--cap", "24"]).cap == 24
