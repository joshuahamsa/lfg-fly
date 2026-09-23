"""The Phase 0 grid (spec §3.0): screen (A), probe (B), twin controls (C), verdict.

- Stage A: every setting simulates 32 base looks plus three perturbations, and
  must pass the activity, sense-change and smoothness checks.
- Stage B: the full planted-taste probe for Stage A passers, capped at the
  `cap` smoothest. When nothing passes, the smoothest settings are probed
  anyway, flagged `passed: False` and ineligible for the gate.
- Stage C: the best eligible setting's rewired and sign-shuffled twins, plus
  per-slot decodability.

Every stage appends JSON lines, keyed by `Setting.key()`, so a killed run
resumes where it stopped.
"""

from __future__ import annotations

import json
import math
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


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    done = {r["key"]: r for r in read_jsonl(out_path)}
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
        rec = {"key": key, "setting": setting.as_dict(), "stats": stats, "smooth": smooth,
               "senses": senses, "activity_ok": activity,
               "passed": bool(activity and smooth["ok"] and all(senses.values())),
               "seconds": round(time.time() - t0, 2)}
        _append(out_path, rec)
        records.append(rec)
    return records


def _smooth_ratio(rec: dict) -> float:
    s = rec["smooth"]
    return s["min"] / s["slot"] if s["slot"] > 0 else math.inf


def _setting_from(rec: dict) -> Setting:
    d = rec["setting"]
    return Setting(BrainParams(**d["brain"]), d["code"], d["g_in"])


def stage_b(ctx: Context, records: list[dict], out_path: Path, cap: int = 24,
            probe_all: bool = False, sim: Simulator | None = None) -> list[dict]:
    passed = [r for r in records if r["passed"]]
    if probe_all:
        pool = list(records)
    elif passed:
        pool = passed
    else:  # nothing eligible: still probe the 6 smoothest, flagged ineligible, for the report
        pool = sorted(records, key=_smooth_ratio)[:6]
    chosen = sorted(pool, key=_smooth_ratio)[:cap]
    dropped = len(pool) - len(chosen)
    done = {r["key"]: r for r in read_jsonl(out_path)}
    sim = sim or Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)
    ps, cache, out = ctx.probe_set, {}, []
    for rec in chosen:
        if rec["key"] in done:
            out.append(done[rec["key"]])
            continue
        t0 = time.time()
        X, stats = features_for(ctx, sim, _setting_from(rec), ps.looks, cache)
        result = P.evaluate(X, ps, device=ctx.device, seed=ctx.seed)
        row = {"key": rec["key"], "setting": rec["setting"], "passed": rec["passed"],
               "probe": result.as_dict(), "stats": stats, "dropped_by_cap": dropped,
               "seconds": round(time.time() - t0, 2)}
        _append(out_path, row)
        out.append(row)
    return out


def stage_c(ctx: Context, best: dict, out_path: Path) -> dict:
    from lfg_fly.teacher.rewire import rewired_graph, sign_shuffled_graph

    setting = _setting_from(best)
    ps, out = ctx.probe_set, {"key": best["key"]}
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
    look_train = np.zeros(len(ps.looks), bool)
    look_train[np.unique(np.concatenate([ps.a_idx[~ps.test], ps.b_idx[~ps.test]]))] = True
    out["decodability"] = P.per_slot_decodability(X, ps.looks, ctx.catalog, look_train,
                                                  ctx.device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def verdict(stage_a_records: list[dict], stage_b_records: list[dict], stage_c_result: dict,
            one_hot: dict, bayes: float) -> dict:
    eligible = [r for r in stage_b_records if r.get("passed")]
    best = max(eligible, key=lambda r: r["probe"]["heldout"], default=None)
    return {
        "gate": GATE,
        "pass": bool(best is not None and best["probe"]["heldout"] >= GATE),
        "best": best,
        "stage_a": {"screened": len(stage_a_records),
                    "passed": sum(r["passed"] for r in stage_a_records)},
        "stage_b_probed": len(stage_b_records),
        "controls": {"one_hot": one_hot, "bayes_ceiling": bayes, **{
            k: v for k, v in stage_c_result.items() if k in ("rewired", "sign_shuffled")}},
    }
