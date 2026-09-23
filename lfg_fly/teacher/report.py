"""Render data/probe/*.json(l) into docs/PHASE0.md: the committed, honest answer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lfg_fly.brain.senses import CODE_NAMES, SLOTS
from lfg_fly.teacher.grid import current_records, stage_c_matches

ATTRIBUTION = (
    "Connectome: Janelia FlyEM MaleCNS v1.0 (CC BY 4.0), Berg et al., Cell 189(18):5504–5526.e15 "
    "(2026), doi:10.1016/j.cell.2026.08.015. Changes: thresholded at ≥3 synapses, signed by "
    "predicted transmitter, photoreceptor columns derived from synaptic partners, simulated. "
    "No endorsement by HHMI/Janelia, Cambridge, MRC-LMB or Google is implied."
)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _setting_cells(r: dict) -> str:
    s = r["setting"]
    return f"{s['brain']['kind']} | {s['brain']['g_syn']} | {s['brain']['bias']} | {s['g_in']}"


def _decodability_by_code(b: list[dict]) -> list[str]:
    """Spec §2: which sense carries identity, measured for every input code, whether or
    not anything passed the gate. Per code: its best-probed Stage B setting."""
    best: dict[str, dict] = {}
    for r in b:
        code = r["setting"]["code"]
        if "decodability" in r and (code not in best
                                    or r["probe"]["heldout"] > best[code]["probe"]["heldout"]):
            best[code] = r
    codes = [c for c in CODE_NAMES if c in best] + sorted(set(best) - set(CODE_NAMES))
    lines = ["", "## Per-slot decodability by input code", "",
             "For each input code, the Stage B setting with the highest held-out taste accuracy "
             "(eligible or not). A linear readout is trained to name each slot's value on the "
             "training looks and scored on the held-out looks; chance is the most common value's "
             "share of the training looks.", ""]
    if not codes:
        return lines + ["No Stage B setting was probed, so there is no decodability to report."]
    lines += ["| Code | Brain | g_syn | bias | g_in | eligible | taste held-out |",
              "|---|---|---|---|---|---|---|"]
    for code in codes:
        r = best[code]
        lines.append(f"| {code} | {_setting_cells(r)} | {'yes' if r['passed'] else 'no'} | "
                     f"{r['probe']['heldout']:.3f} |")
    lines += ["", "| Slot | Chance | " + " | ".join(codes) + " |", "|---" * (len(codes) + 2) + "|"]
    for slot in SLOTS:
        decs = [best[code]["decodability"].get(slot) for code in codes]
        chance = next((d["chance"] for d in decs if d), None)
        cells = []
        for d in decs:
            if d is None:
                cells.append("—")
            elif abs(d["chance"] - chance) > 1e-9:  # only if the probe sets differed
                cells.append(f"{_pct(d['acc'])} (chance {_pct(d['chance'])})")
            else:
                cells.append(_pct(d["acc"]))
        lines.append(f"| {slot} | {'—' if chance is None else _pct(chance)} | "
                     + " | ".join(cells) + " |")
    missing = [c for c in CODE_NAMES if c not in best]
    if missing:
        lines += ["", f"No Stage B setting was probed for: {', '.join(missing)}."]
    return lines


def write_report(results_dir: Path, out_md: Path) -> None:
    v = json.loads((results_dir / "verdict.json").read_text())
    # only the verdict's own context: stale records from another graph or probe set never count
    a = current_records(results_dir / "stage_a.jsonl", v.get("context"))
    b = current_records(results_dir / "stage_b.jsonl", v.get("context"))
    lines = ["# Phase 0: can the fly learn a taste at all?", ""]
    lines += [f"**Verdict: {'PASS' if v['pass'] else 'FAIL'}** (gate: held-out ≥ {v['gate']:.2f} "
              "on a planted additive taste, by a setting that passes every constraint).", ""]
    dropped = v.get("dropped_by_cap", 0)
    if dropped:  # spec §3.0 ranks every passer by held-out: a capped Stage B is partial
        consequence = ("a better passer may exist" if v["pass"]
                       else "this FAIL is not the protocol's verdict")
        lines += [f"**Incomplete:** {dropped} setting(s) that passed Stage A were never probed "
                  f"in Stage B (a `--cap` was used, or Stage B did not finish), so {consequence}. "
                  "Rerun Stage B without `--cap` to probe every passer.", ""]
    ctl = v["controls"]
    lines += ["| Reference | Held-out |", "|---|---|",
              f"| Bayes ceiling | {ctl['bayes_ceiling']:.3f} |",
              f"| One-hot, no brain | {ctl['one_hot']['heldout']:.3f} |"]
    if v["best"]:
        pb = v["best"]["probe"]
        lines.append(f"| **Best fly setting** | **{pb['heldout']:.3f}** "
                     f"({pb['ci_lo']:.3f}–{pb['ci_hi']:.3f}) |")
    for name in ("rewired", "sign_shuffled"):
        if name in ctl:
            lines.append(f"| {name.replace('_', '-')} twin of the best | "
                         f"{ctl[name]['heldout']:.3f} |")
    sa = v["stage_a"]
    lines += ["", "## Stage A: screening", "",
              f"{sa['passed']} of {sa['screened']} settings passed every check.", "",
              "| Brain | Code | Screened | Passed |", "|---|---|---|---|"]
    tally = Counter((r["setting"]["brain"]["kind"], r["setting"]["code"]) for r in a)
    ok = Counter((r["setting"]["brain"]["kind"], r["setting"]["code"]) for r in a if r["passed"])
    for (kind, code), count in sorted(tally.items()):
        lines.append(f"| {kind} | {code} | {count} | {ok[(kind, code)]} |")
    lines += ["", "## Stage B: planted-taste probe", "",
              "| Brain | Code | g_syn | bias | g_in | eligible | held-out | 95% CI |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sorted(b, key=lambda r: -r["probe"]["heldout"]):
        s, pb = r["setting"], r["probe"]
        lines.append(f"| {s['brain']['kind']} | {s['code']} | {s['brain']['g_syn']} | "
                     f"{s['brain']['bias']} | {s['g_in']} | {'yes' if r['passed'] else 'no'} | "
                     f"{pb['heldout']:.3f} | {pb['ci_lo']:.3f}–{pb['ci_hi']:.3f} |")
    lines += _decodability_by_code(b)
    c_path = results_dir / "stage_c.json"
    c = json.loads(c_path.read_text()) if c_path.exists() else {}
    if stage_c_matches(c, v["best"]):  # never another setting's (or context's) twins
        lines += ["", "## Stage C: per-slot decodability of the best setting", "",
                  "| Slot | Held-out | Chance |", "|---|---|---|"]
        for slot in SLOTS:
            if slot in c["decodability"]:
                d = c["decodability"][slot]
                lines.append(f"| {slot} | {_pct(d['acc'])} | {_pct(d['chance'])} |")
        lines += ["", f"Rewire repair swaps: {c['rewire_repairs']:,}."]
    lines += ["", "---", "", ATTRIBUTION, ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
