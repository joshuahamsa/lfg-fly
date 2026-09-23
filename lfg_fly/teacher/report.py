"""Render data/probe/*.json(l) into docs/PHASE0.md: the committed, honest answer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from lfg_fly.brain.senses import CODE_NAMES, SLOTS
from lfg_fly.teacher.grid import (
    CONFIRM_SEED_OFFSET,
    current_records,
    stage_c_matches,
    verdict_status,
)

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


def _brain(s: dict, field: str) -> str:
    """One BrainParams field of a setting; "—" for a hand-written record that lacks it."""
    return str(s["brain"].get(field, "—"))


def _identity(best: dict | None) -> list[str]:
    """The best setting in full: lif-volley T=6 and T=10 (or any two settings that differ
    only in steps, trials or noise) must never read as the same brain."""
    lines = ["", "## The best setting", ""]
    if best is None:
        return lines + ["No setting passed Stage A and was probed, so there is no best setting."]
    s = best["setting"]
    return lines + ["| kind | code | steps | trials | noise | g_syn | bias | g_in |",
                    "|---|---|---|---|---|---|---|---|",
                    f"| {s['brain']['kind']} | {s['code']} | {_brain(s, 'steps')} | "
                    f"{_brain(s, 'trials')} | {_brain(s, 'noise')} | {s['brain']['g_syn']} | "
                    f"{s['brain']['bias']} | {s['g_in']} |"]


def _how_to_read(seed: int, confirm_seed: int) -> list[str]:
    return [
        "", "## How to read the checks", "",
        "1. **Smoothness can pass trivially.** Deterministic spiking brains often pass the "
        "smoothness check without being smooth: a 0.01 concentration change flips no spike, so "
        "the readout moves not at all (distance 0), and 0 is at most 10% of any one-slot swap's "
        "distance.",
        "2. **Each code's minimal perturbation.** The `eyes` code's minimal perturbation is image "
        "brightness × 1.01, not a concentration change. `eyes+nose` perturbs only the nose, and "
        "both senses share one `g_in`, so the grid never drives the eyes at another gain than "
        "the nose.",
        "3. **The sense check is weak.** A sense check passes on any nonzero readout change "
        "between two sets of looks, however small.",
        "4. **`lif-avg` is not reproducible across batch layouts.** Its trial noise stream is per "
        "batch, so a look's features depend on its batch position: the same look in another "
        "batch layout gets different features.",
        "5. **The gate uses the confirmation set.** Stage B scores every passer on the same "
        "selection probe set, and the best setting's selection score is the maximum of those "
        "scores, so it is optimistic (the winner's curse). The gate is that one setting's score "
        "on the confirmation set: an independent planted taste and independent pairs, drawn "
        f"from seed {confirm_seed} (the run's seed {seed} + {CONFIRM_SEED_OFFSET}), which no "
        "selection ever saw. The twins and the one-hot control are scored on it too.",
    ]


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
    status, best, confirmation = verdict_status(v), v["best"], v.get("confirmation")
    seed = v.get("seed", 0)
    confirm_seed = v.get("confirm_seed", seed + CONFIRM_SEED_OFFSET)
    lines = ["# Phase 0: can the fly learn a taste at all?", ""]
    lines += [f"**Verdict: {status}** (gate: held-out ≥ {v['gate']:.2f} on the confirmation set, "
              "a planted additive taste that no selection saw, by the setting that passes every "
              "constraint and scores best on the selection set).", ""]
    if status == "PENDING":
        lines += ["**PENDING:** the best setting has no confirmation-set score yet, so there is "
                  "no verdict. Run `fly grid --stage c` to confirm it.", ""]
    # verdict.json's own count: a Stage B row's dropped_by_cap is only that run's cap
    dropped = v.get("dropped_by_cap", 0)
    if dropped:  # spec §3.0 ranks every passer by held-out: a capped Stage B is partial
        consequence = ("this FAIL is not the protocol's verdict" if status == "FAIL"
                       else "a better passer may exist")
        lines += [f"**Incomplete:** {dropped} setting(s) that passed Stage A were never probed "
                  f"in Stage B (a `--cap` was used, or Stage B did not finish), so {consequence}. "
                  "Rerun Stage B without `--cap` to probe every passer.", ""]
    ctl = v["controls"]
    controls_set = v.get("controls_set", "confirmation" if confirmation else "selection")
    lines += [f"| Reference ({controls_set} set) | Held-out |", "|---|---|",
              f"| Bayes ceiling | {ctl['bayes_ceiling']:.3f} |",
              f"| One-hot, no brain | {ctl['one_hot']['heldout']:.3f} |"]
    if best and confirmation:
        lines.append(f"| **Best fly setting (confirmation)** | **{confirmation['heldout']:.3f}** "
                     f"({confirmation['ci_lo']:.3f}–{confirmation['ci_hi']:.3f}) |")
    elif best:
        lines.append("| **Best fly setting (confirmation)** | PENDING |")
    for name in ("rewired", "sign_shuffled"):
        if name in ctl:
            lines.append(f"| {name.replace('_', '-')} twin of the best | "
                         f"{ctl[name]['heldout']:.3f} |")
    if best:
        pb = best["probe"]
        selection = v.get("selection_heldout", pb["heldout"])
        pool = v.get("selection_pool", v.get("stage_b_probed"))
        lines += ["", f"Best setting's selection score: {selection:.3f} "
                  f"({pb['ci_lo']:.3f}–{pb['ci_hi']:.3f}), selection (max over {pool} probed; "
                  "optimistic, not the gate)."]
    lines += _identity(best)
    sa = v["stage_a"]
    lines += ["", "## Stage A: screening", "",
              f"{sa['passed']} of {sa['screened']} settings passed every check.", "",
              "| Brain | Code | Screened | Passed |", "|---|---|---|---|"]
    tally = Counter((r["setting"]["brain"]["kind"], r["setting"]["code"]) for r in a)
    ok = Counter((r["setting"]["brain"]["kind"], r["setting"]["code"]) for r in a if r["passed"])
    for (kind, code), count in sorted(tally.items()):
        lines.append(f"| {kind} | {code} | {count} | {ok[(kind, code)]} |")
    lines += ["", "## Stage B: planted-taste probe", "",
              "Every setting scored on the selection probe set, the set the best is picked on.",
              "",
              "| Brain | Code | steps | trials | g_syn | bias | g_in | eligible | held-out "
              "| 95% CI |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(b, key=lambda r: -r["probe"]["heldout"]):
        s, pb = r["setting"], r["probe"]
        lines.append(f"| {s['brain']['kind']} | {s['code']} | {_brain(s, 'steps')} | "
                     f"{_brain(s, 'trials')} | {s['brain']['g_syn']} | {s['brain']['bias']} | "
                     f"{s['g_in']} | {'yes' if r['passed'] else 'no'} | "
                     f"{pb['heldout']:.3f} | {pb['ci_lo']:.3f}–{pb['ci_hi']:.3f} |")
    lines += _decodability_by_code(b)
    c_path = results_dir / "stage_c.json"
    c = json.loads(c_path.read_text()) if c_path.exists() else {}
    if stage_c_matches(c, best, seed):  # never another setting's (or context's) Stage C
        lines += ["", "## Stage C: per-slot decodability of the best setting", "",
                  "On the selection probe set's held-out looks, like Stage B's. The confirmation "
                  "scores are in the table at the top.", "",
                  "| Slot | Held-out | Chance |", "|---|---|---|"]
        for slot in SLOTS:
            if slot in c["decodability"]:
                d = c["decodability"][slot]
                lines.append(f"| {slot} | {_pct(d['acc'])} | {_pct(d['chance'])} |")
        lines += ["", f"Rewire repair swaps: {c['rewire_repairs']:,}. "
                  f"Confirmation set: seed {c['confirm_seed']}."]
    lines += _how_to_read(seed, confirm_seed)
    lines += ["", "---", "", ATTRIBUTION, ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
