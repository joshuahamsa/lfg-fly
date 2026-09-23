"""Phase 0b (spec §3.0b): does the fly learn a taste with trait interactions, and does
the real wiring help there?

- calibrate: every taste's Bayes ceiling, additive oracle and no-brain controls, on
  the selection and the confirmation set. An interaction taste whose ceiling-minus-
  oracle gap is below MIN_GAP is too weak to test anything, and Stage B won't start.
- Stage B: every Phase 0 Stage A passer, simulated once on the selection looks and
  scored on every taste. The additive taste is Phase 0's, so a score off Phase 0's
  Stage B by more than REPRO_TOL stops the run.
- Stage C: per taste, the best setting on the confirmation set with its rewired and
  sign-shuffled twins (Phase 0's seeds) and the no-brain controls; the paired
  differences and the four pre-registered readings.

Records are stamped with `interact_fingerprint`: Phase 0's context plus the taste
definitions, so resume never reuses a number from another graph, catalog or taste.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lfg_fly.brain.sim import Simulator
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import probe as P
from lfg_fly.teacher.pixels import character_mask, pixel_stats, visual_terms
from lfg_fly.teacher.render import composite
from lfg_fly.teacher.sample import bayes_ceiling
from lfg_fly.teacher.tastes import TASTE_OFFSETS, TASTES, Tasted, additive_oracle, taste_set

MIN_GAP = 0.03  # spec §3.0b: a weaker interaction term is retuned before Stage B
REPRO_TOL = 0.005  # spec §3.0b: the additive taste must reproduce Phase 0's Stage B
LEARNS_LO = 0.55  # Gate B's bar on the CI lower bound (spec §3.4)
TASTE_VERSION = 1  # bump whenever a taste's definition changes
RENDER_BATCH = 512
SETS = ("selection", "confirmation")


class ReproductionError(RuntimeError):
    """The additive taste's Stage B score is not Phase 0's for the same setting."""


@dataclass(frozen=True)
class TasteSets:
    selection: dict[str, Tasted]
    confirmation: dict[str, Tasted]
    pixels: dict[str, np.ndarray]  # set name -> pixel stats per look, in looks order


def render_looks(ctx: G.Context, looks) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pixel stats and the visual taste's terms for every look, rendered in batches."""
    stats, terms = [], {"contrast": [], "harmony": []}
    for i in range(0, len(looks), RENDER_BATCH):
        chunk = looks[i: i + RENDER_BATCH]
        imgs = composite(chunk, ctx.bank, ctx.zorder)
        stats.append(pixel_stats(imgs))
        for k, v in visual_terms(imgs, character_mask(chunk, ctx.bank)).items():
            terms[k].append(v)
    return np.concatenate(stats), {k: np.concatenate(v) for k, v in terms.items()}


def taste_sets(ctx: G.Context) -> TasteSets:
    """Phase 0's selection and confirmation sets, relabelled with every taste."""
    if ctx.bank is None:
        raise ValueError("Phase 0b renders every look: the context needs its layer bank")
    if ctx.confirm_set is None:
        raise ValueError("Phase 0b needs the confirmation set (seed + CONFIRM_SEED_OFFSET)")
    out, pixels = {}, {}
    for name, ps, seed in (("selection", ctx.probe_set, ctx.seed),
                           ("confirmation", ctx.confirm_set, ctx.seed + G.CONFIRM_SEED_OFFSET)):
        pixels[name], terms = render_looks(ctx, ps.looks)
        out[name] = {t: taste_set(ps, ctx.catalog, t, seed, visual_terms=terms) for t in TASTES}
    return TasteSets(out["selection"], out["confirmation"], pixels)


def interact_fingerprint(ctx: G.Context) -> str:
    payload = {"phase0": G.context_fingerprint(ctx), "tastes": list(TASTES),
               "offsets": TASTE_OFFSETS, "version": TASTE_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def no_brain(t: Tasted, cat, pixels: np.ndarray, device: str, seed: int
             ) -> dict[str, P.ProbeResult]:
    """Spec §3.4 controls 2 and 3, plus Phase 0's one-hot model, on one taste's set."""
    oh = P.one_hot(t.ps.looks, cat)
    ohp = np.hstack([oh, pixels])
    return {"one_hot": P.evaluate(oh, t.ps, device=device, seed=seed),
            "one_hot_pixels": P.evaluate(ohp, t.ps, device=device, seed=seed),
            "mlp": P.evaluate_mlp(ohp, t.ps, device=device, seed=seed)}


def _ceilings(t: Tasted, cat) -> dict[str, float]:
    bayes = bayes_ceiling(t.ps.p[t.ps.test])
    oracle = additive_oracle(t, cat)
    return {"bayes": bayes, "additive_oracle": oracle, "gap": bayes - oracle}


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def calibrate(ctx: G.Context, sets: TasteSets, out_path: Path) -> dict:
    out = {"context": interact_fingerprint(ctx), "min_gap": MIN_GAP, "sets": {}}
    for set_name in SETS:
        rows = {}
        for taste, t in getattr(sets, set_name).items():
            ceil = _ceilings(t, ctx.catalog)
            ctrl = no_brain(t, ctx.catalog, sets.pixels[set_name], ctx.device, ctx.seed)
            rows[taste] = {**ceil, **{k: v.as_dict() for k, v in ctrl.items()},
                           "too_weak": bool(taste != "additive" and ceil["gap"] < MIN_GAP)}
        out["sets"][set_name] = rows
    _write_json(out_path, out)
    return out


def calibration_blocks(calib: dict, context: str) -> str | None:
    """Why Stage B may not start yet, or None."""
    if not calib:
        return "no calibration.json: run `fly interact --stage calib` first"
    if calib.get("context") != context:
        return "calibration.json is from another context: rerun `fly interact --stage calib`"
    weak = sorted({f"{taste} ({set_name})" for set_name in SETS
                   for taste, row in calib["sets"][set_name].items() if row["too_weak"]})
    if weak:
        return (f"too weak to test (ceiling - additive oracle < {MIN_GAP}): {', '.join(weak)}; "
                "retune it and record the retune in spec §3.0b before Stage B")
    return None


def _passers(stage_a_records: list[dict], phase0_b: list[dict]) -> list[dict]:
    """Every Stage A passer, best Phase 0 score first, so the reproduction check meets
    the setting that matters most before anything else."""
    p0 = {r["key"]: r["probe"]["heldout"] for r in phase0_b}
    passed = [r for r in stage_a_records if r["passed"]]
    return sorted(passed, key=lambda r: -p0.get(r["key"], -1.0))


def stage_b(ctx: G.Context, sets: TasteSets, stage_a_records: list[dict],
            phase0_b: list[dict], out_path: Path, sim: Simulator | None = None,
            limit: int | None = None) -> list[dict]:
    """`stage_a_records` and `phase0_b` must be Phase 0's, in this context. `limit` probes
    only the first N passers (best Phase 0 score first): the reproduction pre-flight."""
    context = interact_fingerprint(ctx)
    done = {r["key"]: r for r in G.current_records(out_path, context)}
    phase0 = {r["key"]: r["probe"]["heldout"] for r in phase0_b}
    looks, cache, out = ctx.probe_set.looks, {}, []
    for rec in _passers(stage_a_records, phase0_b)[:limit]:
        if rec["key"] in done:
            out.append(done[rec["key"]])
            continue
        if sim is None:
            sim = Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
        t0 = time.time()
        X, stats = G.features_for(ctx, sim, G._setting_from(rec), looks, cache)
        probe = {taste: P.evaluate(X, t.ps, device=ctx.device, seed=ctx.seed).as_dict()
                 for taste, t in sets.selection.items()}
        p0 = phase0.get(rec["key"])
        if p0 is not None and abs(probe["additive"]["heldout"] - p0) > REPRO_TOL:
            raise ReproductionError(
                f"{rec['key']}: additive taste scores {probe['additive']['heldout']:.4f}, "
                f"Phase 0's Stage B scored {p0:.4f} (tolerance {REPRO_TOL}); stopping")
        row = {"key": rec["key"], "context": context, "setting": rec["setting"],
               "passed": rec["passed"], "probe": probe, "phase0_additive": p0,
               "stats": stats, "seconds": round(time.time() - t0, 2)}
        G._append(out_path, row)
        out.append(row)
    return out


def unprobed(stage_a_records: list[dict], rows: list[dict]) -> int:
    """Stage A passers with no Stage B row: Stage C must not pick a best without them."""
    probed = {r["key"] for r in rows}
    return sum(1 for r in stage_a_records if r["passed"] and r["key"] not in probed)


def best_for(rows: list[dict], taste: str) -> dict | None:
    eligible = [r for r in rows if r.get("passed")]
    return max(eligible, key=lambda r: r["probe"][taste]["heldout"], default=None)


def readings(fly: dict, paired: dict) -> dict[str, bool]:
    """Spec §3.0b's four pre-registered readings."""
    return {"learns": bool(fly["ci_lo"] > LEARNS_LO),
            "uses_interactions": bool(paired["one_hot"]["lo"] > 0),
            "wiring_matters": bool(paired["rewired"]["lo"] > 0),
            "beats_no_brain": bool(paired["mlp"]["lo"] > 0)}


def _twin_graphs(ctx: G.Context):
    """The real graph, then Phase 0's twins (same seeds), built one at a time."""
    from lfg_fly.teacher.rewire import rewired_graph, sign_shuffled_graph

    yield "fly", ctx.graph, None
    rewired, repairs = rewired_graph(ctx.graph, seed=ctx.seed + 11, device=ctx.device)
    yield "rewired", rewired, repairs
    del rewired
    yield "sign_shuffled", sign_shuffled_graph(ctx.graph, ctx.seed + 13), None


def stage_c(ctx: G.Context, sets: TasteSets, rows: list[dict], out_path: Path) -> dict:
    best = {taste: best_for(rows, taste) for taste in TASTES}
    missing = [t for t, b in best.items() if b is None]
    if missing:
        raise ValueError(f"stage C: no eligible Stage B row for {missing}")
    settings = {b["key"]: G._setting_from(b) for b in best.values()}
    looks, images = ctx.confirm_set.looks, {}
    scored: dict[str, dict[str, tuple[P.ProbeResult, dict]]] = {t: {} for t in TASTES}
    repairs = None
    for name, graph, rep in _twin_graphs(ctx):
        repairs = rep if rep is not None else repairs
        sim = Simulator(graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
        for key, setting in settings.items():
            X, stats = G.features_for(ctx, sim, setting, looks, images)
            for taste in TASTES:
                if best[taste]["key"] == key:
                    res = P.evaluate(X, sets.confirmation[taste].ps, device=ctx.device,
                                     seed=ctx.seed)
                    scored[taste][name] = (res, stats)
        del sim, graph  # one simulator on the GPU at a time
    out = {"context": interact_fingerprint(ctx), "confirm_seed": ctx.seed + G.CONFIRM_SEED_OFFSET,
           "rewire_repairs": repairs, "tastes": {}}
    for taste in TASTES:
        t = sets.confirmation[taste]
        ctrl = no_brain(t, ctx.catalog, sets.pixels["confirmation"], ctx.device, ctx.seed)
        fly = scored[taste]["fly"][0]
        others = {**ctrl, "rewired": scored[taste]["rewired"][0],
                  "sign_shuffled": scored[taste]["sign_shuffled"][0]}
        family = t.ps.family[t.ps.test]
        paired = {k: P.paired_diff(fly.correct, v.correct, family, seed=ctx.seed)
                  for k, v in others.items()}
        entry = {"key": best[taste]["key"], "setting": best[taste]["setting"],
                 "selection_heldout": best[taste]["probe"][taste]["heldout"],
                 **_ceilings(t, ctx.catalog),
                 "controls": {k: v.as_dict() for k, v in ctrl.items()}, "paired": paired}
        for name, (res, stats) in scored[taste].items():
            entry[name] = {**res.as_dict(), "stats": stats}
        entry["readings"] = readings(entry["fly"], paired)
        out["tastes"][taste] = entry
    _write_json(out_path, out)
    return out
