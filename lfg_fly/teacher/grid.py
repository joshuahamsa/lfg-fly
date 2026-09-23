"""The Phase 0 grid (spec §3.0): screen (A), probe (B), twin controls (C), verdict.

- Stage A: every setting simulates 32 base looks plus three perturbations, and
  must pass the activity, sense-change and smoothness checks.
- Stage B: the full planted-taste probe for every Stage A passer (spec §3.0
  ranks the passers by held-out accuracy, so none may be skipped). An optional
  `cap` limits the count for a budget run. It interleaves (brain kind, input
  code) strata, is disclosed as `dropped_by_cap` in verdict.json and on the
  VERDICT line, and makes a FAIL inconclusive. When nothing passes, a few
  settings (fewest failed checks, one per stratum) are probed anyway, flagged
  `passed: False` and ineligible for the gate.
  Every probed setting also gets per-slot decodability, so the report can say
  which input code carries identity even when nothing passes the gate.
- Stage C: the best eligible setting's rewired and sign-shuffled twins, plus
  per-slot decodability.

Every stage appends JSON lines, keyed by `Setting.key()`, so a killed run
resumes where it stopped. Every record is stamped with the run's context
fingerprint (graph, threshold, probe-set size, seed, catalog), and resume
reuses only records from the same context: a rebuilt graph or a different
`--min-syn` / `--pairs` never inherits stale numbers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lfg_fly.brain.senses import C_REF, CODE_NAMES, NON_BODY_SLOTS, SLOTS, InputBuilder
from lfg_fly.brain.sim import BrainParams, Simulator
from lfg_fly.teacher import probe as P
from lfg_fly.teacher.render import composite
from lfg_fly.teacher.sample import base_look, near_variant

GATE = 0.60
BATCH_COLUMNS = 512  # looks x trials per GPU batch (8 GB card, N = 165k)
FALLBACK_PROBES = 6  # ineligible settings probed for the report when nothing passes Stage A

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Setting:
    brain: BrainParams
    code: str
    g_in: float

    def as_dict(self) -> dict:
        return {"brain": self.brain.as_dict(), "code": self.code, "g_in": self.g_in}

    def key(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


def default_grid() -> list[Setting]:
    out = []
    lif_like = [
        ("lif", {"steps": 60}),
        ("lif-avg", {"steps": 60, "trials": 8, "noise": 0.05}),
        ("lif-volley", {"steps": 6}),
        ("lif-volley", {"steps": 10}),
    ]
    for code in CODE_NAMES:
        for kind, extra in lif_like:
            for g in (4.0, 6.0, 8.0):
                for bias in (0.0, 0.05, 0.1):
                    for g_in in (0.5, 2.0):
                        brain = BrainParams(kind=kind, g_syn=g, bias=bias, **extra)
                        out.append(Setting(brain, code, g_in))
        for g in (0.5, 1.0, 2.0):
            for bias in (0.0, 0.1):
                for g_in in (0.5, 2.0):
                    brain = BrainParams(kind="rate", g_syn=g, bias=bias, steps=60)
                    out.append(Setting(brain, code, g_in))
    return out


@dataclass
class Context:
    graph: Any
    pops: Any
    retina: Any
    catalog: Any
    bank: Any
    zorder: Any
    device: str
    probe_set: Any
    seed: int = 0


def context_fingerprint(ctx: Context) -> str:
    """What every Stage A/B/C number depends on besides the setting itself.

    A short hex digest over the graph hash, its synapse threshold, the probe-set
    size, the seed and the catalog's canonical JSON. Resume reuses only records
    stamped with the current fingerprint.
    """
    payload = {
        "graph_hash": ctx.graph.graph_hash(),
        "min_syn": int(ctx.graph.min_syn),
        "pairs": len(ctx.probe_set.y),
        "seed": int(ctx.seed),
        "catalog": hashlib.sha256(ctx.catalog.to_json().encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    """Read JSON lines. A killed append leaves a truncated last line: that one is
    skipped with a warning. Any other unparsable line raises."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            log.warning("%s: skipping a truncated last line (%d chars)", path, len(line))
    return out


def current_records(path: Path, context: str | None) -> list[dict]:
    """The records in `path` stamped with `context`; the rest are counted and ignored."""
    records = read_jsonl(path)
    kept = [r for r in records if r.get("context") == context]
    if len(kept) != len(records):
        log.warning("%s: ignored %d record(s) from a different context (current %s)",
                    path, len(records) - len(kept), context)
    return kept


def _repair_tail(path: Path) -> None:
    """A killed append can leave a last line with no newline. Appending after it
    would glue the next record onto it, so drop it if it's a fragment (or end it
    if it's a whole record) first."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
    data = path.read_bytes()
    head, sep, tail = data.rpartition(b"\n")
    try:
        json.loads(tail)
        data += b"\n"
    except ValueError:
        log.warning("%s: dropping a truncated last line (%d bytes) before appending",
                    path, len(tail))
        data = head + sep
    path.write_bytes(data)


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(path)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _images(ctx: Context, looks, cache: dict | None) -> torch.Tensor | None:
    if ctx.bank is None:
        return None
    missing = [lk for lk in looks if cache is None or lk not in cache]
    if missing:
        rendered = composite(missing, ctx.bank, ctx.zorder)
        if cache is not None:
            for lk, img in zip(missing, rendered, strict=True):
                cache[lk] = img
        else:
            return rendered
    return torch.stack([cache[lk] for lk in looks])


def features_for(ctx: Context, sim: Simulator, setting: Setting, looks, images_cache=None,
                 conc: np.ndarray | None = None, image_scale: float = 1.0,
                 builder: InputBuilder | None = None) -> tuple[np.ndarray, dict]:
    builder = builder or InputBuilder(setting.code, ctx.graph.n, ctx.pops, ctx.retina,
                                      setting.g_in, ctx.device)
    batch = max(1, BATCH_COLUMNS // setting.brain.trials)
    feats, act, ro_act, max_frac = [], [], [], 0.0
    for i in range(0, len(looks), batch):
        chunk = looks[i: i + batch]
        imgs = _images(ctx, chunk, images_cache) if builder.uses_eyes else None
        if imgs is not None and image_scale != 1.0:
            imgs = (imgs * image_scale).clamp(0.0, 1.0)
        c = None if conc is None else conc[i: i + batch]
        res = sim.run(builder.drive(chunk, imgs, c), setting.brain, builder.rest_drive(),
                      builder.rest_key())
        feats.append(res.features.float().cpu().numpy())
        act.append(res.active_frac.cpu().numpy())
        ro_act.append(res.readout_active_frac.cpu().numpy())
        max_frac = max(max_frac, res.max_step_frac)
    stats = {"active_frac": float(np.concatenate(act).mean()),
             "readout_active_frac": float(np.concatenate(ro_act).mean()),
             "max_step_frac": float(max_frac)}
    return np.concatenate(feats), stats


def _sense_checks(ctx, sim, setting, rng, images_cache) -> dict:
    base = InputBuilder(setting.code, ctx.graph.n, ctx.pops, ctx.retina, setting.g_in, ctx.device)
    one = [base_look(ctx.catalog, rng) for _ in range(16)]
    two = [base_look(ctx.catalog, rng) for _ in range(16)]
    out = {}
    for comp in base.components:
        b = base.only(comp) if len(base.components) > 1 else base
        f1, _ = features_for(ctx, sim, setting, one, images_cache, builder=b)
        f2, _ = features_for(ctx, sim, setting, two, images_cache, builder=b)
        out[comp] = P.readout_distance(f1, f2) > 0
    return out


def stage_a(ctx: Context, settings: list[Setting], out_path: Path, n: int = 32,
            sim: Simulator | None = None) -> list[dict]:
    context = context_fingerprint(ctx)
    done = {r["key"]: r for r in current_records(out_path, context)}
    sim = sim or Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
    records = []
    for setting in settings:
        key = setting.key()
        if key in done:
            records.append(done[key])
            continue
        t0 = time.time()
        rng = np.random.default_rng(ctx.seed + 7)
        cache: dict = {}
        bases = [base_look(ctx.catalog, rng) for _ in range(n)]
        slot1 = [near_variant(b, ctx.catalog, rng, n_slots=1) for b in bases]
        full = [base_look(ctx.catalog, rng) for _ in range(n)]
        f_base, stats = features_for(ctx, sim, setting, bases, cache)
        f_slot, _ = features_for(ctx, sim, setting, slot1, cache)
        f_full, _ = features_for(ctx, sim, setting, full, cache)
        if setting.code == "eyes":
            f_min, _ = features_for(ctx, sim, setting, bases, cache, image_scale=1.01)
        else:
            conc = np.full((n, len(SLOTS)), C_REF, np.float32)
            conc[:, SLOTS.index(str(rng.choice(NON_BODY_SLOTS)))] += 0.01
            f_min, _ = features_for(ctx, sim, setting, bases, cache, conc=conc)
        d = {"min": P.readout_distance(f_base, f_min), "slot": P.readout_distance(f_base, f_slot),
             "full": P.readout_distance(f_base, f_full)}
        smooth = {**d, "ok": P.smooth_ok(d["min"], d["slot"], d["full"])}
        senses = _sense_checks(ctx, sim, setting, rng, cache)
        activity = P.activity_ok(stats["max_step_frac"], stats["readout_active_frac"])
        rec = {"key": key, "context": context, "setting": setting.as_dict(), "stats": stats,
               "smooth": smooth, "senses": senses, "activity_ok": activity,
               "passed": bool(activity and smooth["ok"] and all(senses.values())),
               "seconds": round(time.time() - t0, 2)}
        _append(out_path, rec)
        records.append(rec)
    return records


def _stratified(pool: list[dict]) -> list[dict]:
    """`pool` interleaved round-robin over its (brain kind, input code) strata.

    Each stratum keeps `pool`'s order, and the strata take turns in order of first
    appearance, so any prefix covers every kind and code before any of them
    repeats. Nothing here ranks by the smoothness ratio d_min/d_slot. A
    deterministic spiking setting whose +0.01 perturbation flips no spike has
    d_min == 0 exactly, and on the real graph dozens of those ties (grid order,
    nose first) filled a smoothest-24 cap with no rate or all-sensory setting.
    """
    strata: dict[tuple[str, str], list[dict]] = {}
    for rec in pool:
        s = rec["setting"]
        strata.setdefault((s["brain"]["kind"], s["code"]), []).append(rec)
    out: list[dict] = []
    for i in range(max((len(v) for v in strata.values()), default=0)):
        out += [v[i] for v in strata.values() if i < len(v)]
    return out


def _checks_failed(rec: dict) -> int:
    """How many of the three Stage A checks (activity, smoothness, senses) a record failed."""
    return (int(not rec["activity_ok"]) + int(not rec["smooth"]["ok"])
            + int(not all(rec["senses"].values())))


def stage_b_selection(records: list[dict], cap: int | None = None,
                      probe_all: bool = False) -> tuple[list[dict], int]:
    """The Stage A records Stage B probes, and how many of the pool a cap left out.

    - By default, every passer (spec §3.0: passers are ranked by held-out, so
      the gate has to see all of them).
    - `probe_all`: every record, eligible or not.
    - Nothing passed: FALLBACK_PROBES ineligible settings for the report. They
      are the fewest-failed-checks record of each stratum in turn.

    `cap` truncates the stratified order: every (kind, code) stratum is taken
    once before any is taken twice.
    """
    if cap is not None and cap < 1:
        raise ValueError(f"cap must be None (probe every passer) or >= 1, got {cap}")
    passed = [r for r in records if r["passed"]]
    if probe_all:
        pool = _stratified(records)
    elif passed:
        pool = _stratified(passed)
    else:
        pool = _stratified(sorted(records, key=_checks_failed))[:FALLBACK_PROBES]
    chosen = pool if cap is None else pool[:cap]
    return chosen, len(pool) - len(chosen)


def _setting_from(rec: dict) -> Setting:
    d = rec["setting"]
    return Setting(BrainParams(**d["brain"]), d["code"], d["g_in"])


def _look_train_mask(ps) -> np.ndarray:
    """Looks that appear in a training pair; the rest are the held-out looks."""
    look_train = np.zeros(len(ps.looks), bool)
    look_train[np.unique(np.concatenate([ps.a_idx[~ps.test], ps.b_idx[~ps.test]]))] = True
    return look_train


def stage_b(ctx: Context, records: list[dict], out_path: Path, cap: int | None = None,
            probe_all: bool = False, sim: Simulator | None = None) -> list[dict]:
    context = context_fingerprint(ctx)
    stale = [r for r in records if r.get("context") != context]
    if stale:
        log.warning("stage B: ignored %d Stage A record(s) from a different context (current %s)",
                    len(stale), context)
        records = [r for r in records if r.get("context") == context]
    chosen, dropped = stage_b_selection(records, cap, probe_all)
    if dropped:
        log.warning("stage B: cap %d left %d setting(s) unprobed; the verdict discloses it",
                    cap, dropped)
    done = {r["key"]: r for r in current_records(out_path, context)}
    sim = sim or Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
    ps, cache, out = ctx.probe_set, {}, []
    look_train = _look_train_mask(ps)
    for rec in chosen:
        if rec["key"] in done:
            out.append(done[rec["key"]])
            continue
        t0 = time.time()
        X, stats = features_for(ctx, sim, _setting_from(rec), ps.looks, cache)
        result = P.evaluate(X, ps, device=ctx.device, seed=ctx.seed)
        decodability = P.per_slot_decodability(X, ps.looks, ctx.catalog, look_train, ctx.device)
        row = {"key": rec["key"], "context": context, "setting": rec["setting"],
               "passed": rec["passed"], "probe": result.as_dict(), "decodability": decodability,
               "stats": stats, "dropped_by_cap": dropped, "seconds": round(time.time() - t0, 2)}
        _append(out_path, row)
        out.append(row)
    return out


def stage_c(ctx: Context, best: dict, out_path: Path) -> dict:
    from lfg_fly.teacher.rewire import rewired_graph, sign_shuffled_graph

    setting = _setting_from(best)
    ps, out = ctx.probe_set, {"key": best["key"], "context": context_fingerprint(ctx)}
    rewired, repairs = rewired_graph(ctx.graph, seed=ctx.seed + 11, device=ctx.device)
    twins = (("rewired", rewired),
             ("sign_shuffled", sign_shuffled_graph(ctx.graph, ctx.seed + 13)))
    for name, graph in twins:
        sim = Simulator(graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
        X, stats = features_for(ctx, sim, setting, ps.looks, {})
        out[name] = {**P.evaluate(X, ps, device=ctx.device, seed=ctx.seed).as_dict(),
                     "stats": stats}
    out["rewire_repairs"] = repairs
    sim = Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
    X, _ = features_for(ctx, sim, setting, ps.looks, {})
    out["decodability"] = P.per_slot_decodability(X, ps.looks, ctx.catalog,
                                                  _look_train_mask(ps), ctx.device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def best_eligible(stage_b_records: list[dict]) -> dict | None:
    """The gate's candidate: the eligible Stage B row with the highest held-out."""
    eligible = [r for r in stage_b_records if r.get("passed")]
    return max(eligible, key=lambda r: r["probe"]["heldout"], default=None)


def stage_c_matches(stage_c_result: dict, best: dict | None) -> bool:
    """True iff `stage_c_result` is Stage C of exactly this best setting, in its context.

    Anything else (a leftover stage_c.json from an earlier best, or from another
    graph or probe set) must not be reported as the best setting's twins.
    """
    return (best is not None and bool(stage_c_result)
            and stage_c_result.get("key") == best["key"]
            and stage_c_result.get("context") == best.get("context"))


def unprobed_passers(stage_a_records: list[dict], stage_b_records: list[dict]) -> int:
    """Stage A passers with no Stage B row: settings the gate never saw.

    Counted from the records rather than taken from a run's cap, so the number
    stays right across resumes, a changed `--cap` and `--stage c` reruns. After a
    full Stage B it is nonzero only when a cap was used.
    """
    probed = {r["key"] for r in stage_b_records}
    return sum(1 for r in stage_a_records if r.get("passed") and r["key"] not in probed)


def verdict(stage_a_records: list[dict], stage_b_records: list[dict], stage_c_result: dict,
            one_hot: dict, bayes: float) -> dict:
    best = best_eligible(stage_b_records)
    twins = stage_c_result if stage_c_matches(stage_c_result, best) else {}
    return {
        "gate": GATE,
        "pass": bool(best is not None and best["probe"]["heldout"] >= GATE),
        "best": best,
        "stage_a": {"screened": len(stage_a_records),
                    "passed": sum(r["passed"] for r in stage_a_records)},
        "stage_b_probed": len(stage_b_records),
        # nonzero: a FAIL is inconclusive, because spec §3.0 ranks every passer
        "dropped_by_cap": unprobed_passers(stage_a_records, stage_b_records),
        "controls": {"one_hot": one_hot, "bayes_ceiling": bayes, **{
            k: v for k, v in twins.items() if k in ("rewired", "sign_shuffled")}},
    }


def verdict_line(v: dict, cap: int | None = None) -> str:
    """The run's last line, e.g. `VERDICT: PASS (best held-out 0.753)`.

    Whenever a cap is in force, or the verdict left a passer unprobed, the line
    says how many passers the gate never saw. A FAIL in that state is flagged
    as not the protocol's verdict.
    """
    parts = [f"best held-out {v['best']['probe']['heldout']:.3f}" if v["best"]
             else "no eligible setting"]
    dropped = v.get("dropped_by_cap", 0)
    if cap is not None or dropped:
        parts.append(f"{dropped} Stage A passer(s) not probed in Stage B"
                     + (f" (--cap {cap})" if cap is not None else ""))
        if dropped and not v["pass"]:
            parts.append("so this FAIL is not the protocol's verdict: "
                         "rerun Stage B without --cap to probe every passer")
    return f"VERDICT: {'PASS' if v['pass'] else 'FAIL'} ({'; '.join(parts)})"
