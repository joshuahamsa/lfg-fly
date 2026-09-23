"""Render data/probe/*.json(l) into docs/PHASE0.md: the committed, honest answer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lfg_fly.teacher.grid import read_jsonl

ATTRIBUTION = (
    "Connectome: Janelia FlyEM MaleCNS v1.0 (CC BY 4.0), Berg et al., Cell 189(18):5504–5526.e15 "
    "(2026), doi:10.1016/j.cell.2026.08.015. Changes: thresholded at ≥3 synapses, signed by "
    "predicted transmitter, photoreceptor columns derived from synaptic partners, simulated. "
    "No endorsement by HHMI/Janelia, Cambridge, MRC-LMB or Google is implied."
)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(results_dir: Path, out_md: Path) -> None:
    v = json.loads((results_dir / "verdict.json").read_text())
    a = read_jsonl(results_dir / "stage_a.jsonl")
    b = read_jsonl(results_dir / "stage_b.jsonl")
    lines = ["# Phase 0: can the fly learn a taste at all?", ""]
    lines += [f"**Verdict: {'PASS' if v['pass'] else 'FAIL'}** (gate: held-out ≥ {v['gate']:.2f} "
              "on a planted additive taste, by a setting that passes every constraint).", ""]
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
    if b and b[0].get("dropped_by_cap"):
        lines += ["", f"Stage B cap dropped {b[0]['dropped_by_cap']} passing settings "
                  "(smoothest kept)."]
    c_path = results_dir / "stage_c.json"
    if c_path.exists():
        c = json.loads(c_path.read_text())
        lines += ["", "## Stage C: per-slot decodability of the best setting", "",
                  "| Slot | Held-out | Chance |", "|---|---|---|"]
        for slot, d in c["decodability"].items():
            lines.append(f"| {slot} | {_pct(d['acc'])} | {_pct(d['chance'])} |")
        lines += ["", f"Rewire repair swaps: {c['rewire_repairs']:,}."]
    lines += ["", "---", "", ATTRIBUTION, ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
