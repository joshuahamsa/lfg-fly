import json

from lfg_fly.teacher import interact_report as IR
from lfg_fly.teacher.tastes import TASTES


def _res(h):
    return {"heldout": h, "ci_lo": h - 0.03, "ci_hi": h + 0.03, "n_test": 600}


def _diff(d):
    return {"diff": d, "lo": d - 0.02, "hi": d + 0.02}


SETTING = {"brain": {"kind": "rate", "g_syn": 1.0, "bias": 0.1, "steps": 60, "trials": 1,
                     "noise": 0.0}, "code": "all-sensory-brain", "g_in": 2.0}
EYES = {"brain": {"kind": "lif-volley", "g_syn": 4.0, "bias": 0.05, "steps": 6, "trials": 1,
                  "noise": 0.0}, "code": "eyes", "g_in": 0.5}


def _taste(fly, wiring):
    paired = {"one_hot": _diff(0.04), "one_hot_pixels": _diff(0.0), "mlp": _diff(-0.05),
              "rewired": _diff(wiring), "sign_shuffled": _diff(0.03)}
    return {"key": "k", "setting": SETTING, "selection_heldout": fly + 0.01,
            "bayes": 0.82, "additive_oracle": 0.74, "gap": 0.08,
            "fly": _res(fly), "rewired": _res(fly - wiring), "sign_shuffled": _res(fly - 0.03),
            "controls": {"one_hot": _res(fly - 0.04), "one_hot_pixels": _res(fly),
                         "mlp": _res(fly + 0.05)},
            "paired": paired,
            "readings": {"learns": True, "uses_interactions": True,
                         "wiring_matters": wiring - 0.02 > 0, "beats_no_brain": False}}


def _write(d):
    c = {"context": "ctx", "confirm_seed": 1000, "rewire_repairs": 22153,
         "tastes": {"additive": _taste(0.758, 0.0), "latent": _taste(0.701, 0.05),
                    "visual": _taste(0.644, 0.0)}}
    cal = {"context": "ctx", "sets": {s: {t: {"bayes": 0.82, "additive_oracle": 0.74,
                                              "gap": 0.08, "too_weak": t == "visual",
                                              "one_hot": _res(0.7), "one_hot_pixels": _res(0.71),
                                              "mlp": _res(0.72)} for t in TASTES}
                                      for s in ("selection", "confirmation")}}
    rows = [{"key": "a", "context": "ctx", "passed": True, "setting": SETTING,
             "phase0_additive": 0.760, "phase0_lam": 1.0,
             "probe": {"additive": {**_res(0.761), "lam": 1.0}, "latent": _res(0.70),
                       "visual": _res(0.60)}},
            {"key": "b", "context": "ctx", "passed": True, "setting": EYES,
             "phase0_additive": 0.600, "phase0_lam": 0.01,
             "probe": {"additive": {**_res(0.598), "lam": 1.0}, "latent": _res(0.55),
                       "visual": _res(0.66)}},
            {"key": "old", "context": "stale", "passed": True, "setting": EYES,
             "phase0_additive": 0.1,
             "probe": {"additive": _res(0.9), "latent": _res(0.9), "visual": _res(0.9)}}]
    (d / "stage_c.json").write_text(json.dumps(c))
    (d / "calibration.json").write_text(json.dumps(cal))
    (d / "stage_b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return c, cal


def test_report_states_every_reading_with_its_paired_difference(tmp_path):
    _write(tmp_path)
    IR.write_report(tmp_path, tmp_path / "PHASE0B.md")
    md = (tmp_path / "PHASE0B.md").read_text()
    assert md.startswith("# Phase 0b")
    readings = md.split("## Readings")[1].split("##")[0]
    latent = next(line for line in readings.splitlines() if line.startswith("| latent"))
    assert "**yes**" in latent and "+0.050 (+0.030 to +0.070)" in latent  # wiring matters
    visual = next(line for line in readings.splitlines() if line.startswith("| visual"))
    assert "no: +0.000 (-0.020 to +0.020)" in visual  # rewired: no difference
    assert "no: -0.050" in visual  # the MLP beats the fly
    assert "0.701 (0.671–0.731)" in md  # the fly's confirmation score with its CI
    assert "| latent | rate | all-sensory-brain | 60 | 1 | 0.0 | 1.0 | 0.1 | 2.0 | 0.711 |" in md


def test_report_uses_only_this_contexts_stage_b_rows(tmp_path):
    _write(tmp_path)
    IR.write_report(tmp_path, tmp_path / "PHASE0B.md")
    md = (tmp_path / "PHASE0B.md").read_text()
    assert "2 settings probed" in md
    assert "| lif-volley | eyes | 0.598 | 0.550 | 0.660 |" in md  # not the stale 0.9s
    assert ("Additive-taste score minus Phase 0's, over 2 settings: mean -0.0005, "
            "sd 0.0015, largest |difference| 0.0020; 1 chose a different λ") in md
    assert "22153 repair swaps" in md
    assert "Janelia FlyEM MaleCNS" in md


def test_console_lines_flag_weak_tastes_and_name_every_reading(tmp_path):
    c, cal = _write(tmp_path)
    cal_lines = IR.calibration_lines(cal)
    assert len(cal_lines) == 2 * len(TASTES)
    assert sum("TOO WEAK" in line for line in cal_lines) == 2  # visual, both sets
    lines = IR.reading_lines(c)
    assert len(lines) == len(TASTES)
    assert "wiring_matters YES" in lines[1] and "wiring_matters no" in lines[2]
