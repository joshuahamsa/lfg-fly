"""Render data/probe-interact/*.json(l) into docs/PHASE0B.md (spec §3.0b)."""

from __future__ import annotations

import json
from pathlib import Path

from lfg_fly.teacher.grid import read_jsonl
from lfg_fly.teacher.interact import LEARNS_LO, REPRO_MEAN_TOL, SETS, reproduction_shift
from lfg_fly.teacher.report import ATTRIBUTION
from lfg_fly.teacher.tastes import TASTES

# (reading, column label, the model the fly is compared with; None for its own CI)
READINGS = (("learns", f"Learns it (CI lower bound > {LEARNS_LO})", None),
            ("uses_interactions", "Uses interactions (fly − one-hot)", "one_hot"),
            ("wiring_matters", "Wiring matters (fly − rewired)", "rewired"),
            ("beats_no_brain", "Beats no-brain (fly − MLP)", "mlp"))
TWINS = (("rewired", "Rewired twin"), ("sign_shuffled", "Sign-shuffled twin"))
CONTROLS = (("one_hot", "One-hot logistic regression"),
            ("one_hot_pixels", "One-hot + pixel stats logistic regression"),
            ("mlp", "MLP (256 hidden) on one-hot + pixel stats"))


def _acc(r: dict) -> str:
    return f"{r['heldout']:.3f} ({r['ci_lo']:.3f}–{r['ci_hi']:.3f})"


def _diff(d: dict) -> str:
    return f"{d['diff']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})"


def calibration_lines(cal: dict) -> list[str]:
    out = []
    for set_name in SETS:
        for taste in TASTES:
            r = cal["sets"][set_name][taste]
            out.append(
                f"calib {set_name:12s} {taste:8s} Bayes {r['bayes']:.3f}  additive oracle "
                f"{r['additive_oracle']:.3f}  gap {r['gap']:+.3f}"
                f"{'  TOO WEAK' if r['too_weak'] else ''}  |  one-hot "
                f"{r['one_hot']['heldout']:.3f}  +pixels {r['one_hot_pixels']['heldout']:.3f}  "
                f"MLP {r['mlp']['heldout']:.3f}")
    return out


def reading_lines(c: dict) -> list[str]:
    out = []
    for taste in TASTES:
        t = c["tastes"][taste]
        verdicts = ", ".join(f"{key} {'YES' if t['readings'][key] else 'no'}"
                             for key, _, _ in READINGS)
        out.append(f"{taste:8s} fly {_acc(t['fly'])}  rewired {t['rewired']['heldout']:.3f}  "
                   f"one-hot {t['controls']['one_hot']['heldout']:.3f}  "
                   f"MLP {t['controls']['mlp']['heldout']:.3f}  |  {verdicts}")
    return out


def _readings_table(T: dict) -> list[str]:
    lines = ["| Taste | " + " | ".join(label for _, label, _ in READINGS) + " |",
             "|---|" + "---|" * len(READINGS)]
    for taste in TASTES:
        t, cells = T[taste], []
        for key, _, vs in READINGS:
            detail = f"CI lo {t['fly']['ci_lo']:.3f}" if vs is None else _diff(t["paired"][vs])
            cells.append(f"{'**yes**' if t['readings'][key] else 'no'}: {detail}")
        note = " (Phase 0's taste, for reference)" if taste == "additive" else ""
        lines.append(f"| {taste}{note} | " + " | ".join(cells) + " |")
    return lines


def _accuracy_table(T: dict) -> list[str]:
    def row(label: str, cells) -> str:
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [f"| Model | {' | '.join(TASTES)} |", "|---|" + "---|" * len(TASTES),
             row("Bayes ceiling", (f"{T[t]['bayes']:.3f}" for t in TASTES)),
             row("Additive oracle", (f"{T[t]['additive_oracle']:.3f}" for t in TASTES)),
             row("**The fly**", (f"**{_acc(T[t]['fly'])}**" for t in TASTES))]
    lines += [row(label, (_acc(T[t][key]) for t in TASTES)) for key, label in TWINS]
    lines += [row(label, (_acc(T[t]["controls"][key]) for t in TASTES))
              for key, label in CONTROLS]
    lines += ["", "The additive oracle is the best any additive model could do with unlimited "
              "data. The gap between it and the Bayes ceiling is the accuracy that only the "
              "interactions carry.", "", "## Paired differences (fly − model)", "",
              f"| Model | {' | '.join(TASTES)} |", "|---|" + "---|" * len(TASTES)]
    lines += [row(label, (_diff(T[t]["paired"][key]) for t in TASTES))
              for key, label in (*TWINS, *CONTROLS)]
    return lines


def _best_settings(T: dict) -> list[str]:
    lines = ["| Taste | kind | code | steps | trials | noise | g_syn | bias | g_in | selection |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for taste in TASTES:
        s, b = T[taste]["setting"], T[taste]["setting"]["brain"]
        lines.append(f"| {taste} | {b['kind']} | {s['code']} | {b.get('steps', '—')} | "
                     f"{b.get('trials', '—')} | {b.get('noise', '—')} | {b['g_syn']} | "
                     f"{b['bias']} | {s['g_in']} | {T[taste]['selection_heldout']:.3f} |")
    return lines


def _strata(rows: list[dict]) -> list[str]:
    """The best selection score of each (brain kind, input code), per taste."""
    best: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        if r.get("passed"):
            k = (r["setting"]["brain"]["kind"], r["setting"]["code"])
            cell = best.setdefault(k, {t: 0.0 for t in TASTES})
            for t in TASTES:
                cell[t] = max(cell[t], r["probe"][t]["heldout"])
    lines = [f"| Brain | Code | {' | '.join(TASTES)} |", "|---|---|" + "---|" * len(TASTES)]
    for (kind, code), cell in sorted(best.items()):
        lines.append(f"| {kind} | {code} | " + " | ".join(f"{cell[t]:.3f}" for t in TASTES) + " |")
    return lines


def _calibration_table(cal: dict) -> list[str]:
    lines = ["| Taste | Bayes | Additive oracle | Gap | One-hot | + pixels | MLP |",
             "|---|---|---|---|---|---|---|"]
    for taste in TASTES:
        r = cal["sets"]["selection"][taste]
        lines.append(f"| {taste} | {r['bayes']:.3f} | {r['additive_oracle']:.3f} | "
                     f"{r['gap']:+.3f} | {r['one_hot']['heldout']:.3f} | "
                     f"{r['one_hot_pixels']['heldout']:.3f} | {r['mlp']['heldout']:.3f} |")
    return lines


def _reproduction(rows: list[dict]) -> str:
    s = reproduction_shift(rows)
    if not s["n"]:
        return "No Stage B row had a Phase 0 score to compare with."
    return (f"Additive-taste score minus Phase 0's, over {s['n']} settings: mean {s['mean']:+.4f}, "
            f"sd {s['sd']:.4f}, largest |difference| {s['max_abs']:.4f}; {s['lam_changed']} chose "
            "a different λ than in Phase 0. The labels are Phase 0's exactly; single scores move "
            "by up to ~0.01 between identical runs because GPU rounding can flip the readout's "
            f"CV-chosen λ (spec §3.0b amendment 1). Stage C requires |mean| ≤ {REPRO_MEAN_TOL}.")


def write_report(results_dir: Path, out_md: Path) -> None:
    c = json.loads((results_dir / "stage_c.json").read_text())
    cal = json.loads((results_dir / "calibration.json").read_text())
    rows = [r for r in read_jsonl(results_dir / "stage_b.jsonl") if r["context"] == c["context"]]
    T = c["tastes"]
    lines = [
        "# Phase 0b: does the fly learn trait interactions?", "",
        "Pre-registered in spec §3.0b before any run. This probe is a diagnostic, not a gate: "
        "Phase 0's verdict stands. Every number below is on the **confirmation set**, an "
        "independent instance of each taste that played no part in choosing the setting. "
        "Differences are paired (both models on the same test pairs), with 95% family-"
        "bootstrap CIs, and a reading holds iff its CI lies above 0.", "",
        "## Readings", "", *_readings_table(T), "",
        "## Held-out accuracy", "", *_accuracy_table(T), "",
        "## The best setting per taste", "",
        "Chosen on each taste's selection set, from every Stage A passer.", "",
        *_best_settings(T), "",
        "## Stage B: the best selection score per brain and code", "",
        f"{len(rows)} settings probed on every taste's selection set.", "", *_strata(rows), "",
        "## Calibration (selection set)", "", *_calibration_table(cal), "",
        "## Reproduction", "", _reproduction(rows),
        f"Rewired twin: {c['rewire_repairs']} repair swaps.", "",
        "## Attribution", "", ATTRIBUTION, "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
