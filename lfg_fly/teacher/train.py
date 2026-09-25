"""Training on the critic's labels (spec §3.3), the controls and REPORT.md (spec §3.4).

- `load_labels`: `data/critic/<round>.jsonl` (spec §3.2, as `critic.assemble` writes it)
  becomes a `LabelSet`: pairs the critic flipped on are dropped, the winner is the
  label, the strength (1–3) is the sample weight, and >= 20% of look families are
  held out. It has the `ProbeSet` attributes `probe.evaluate` reads, so the fly and
  every control are scored by the Phase 0 protocol on the same held-out pairs.
- `train`: the looks are simulated with the Phase 0 brain at `c_ref` (spec §3.3), the
  taste head is a weighted Bradley–Terry regression (spec §2 Readout), and the
  controls of spec §3.4 rows 2–7 are scored on the same test pairs with paired
  family-bootstrap differences. The go-live gate is the fly's family-bootstrap 95% CI
  lower bound above `GATE_CI_LO`. Beating a control is not required; whether the fly
  does is stated.
- `write_checkpoint`: `checkpoints/<version>/` with a full manifest (spec §2 Checkpoint
  and versioning) and the head.
- `write_report`: REPORT.md, every number with its CI, the controls beside the fly,
  the gate verdict and the critic's QC, in plain statements either way (spec §3.4:
  the report "won't be shaded").
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lfg_fly import paths
from lfg_fly.brain import checkpoint as C
from lfg_fly.brain import readout as R
from lfg_fly.brain.senses import C_REF, SLOTS, build_retina
from lfg_fly.brain.sim import BrainParams, Simulator
from lfg_fly.connectome import build as B
from lfg_fly.connectome.columns import load_columns
from lfg_fly.connectome.neurons import populations
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import probe as P
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.interact import render_looks
from lfg_fly.teacher.render import LFG_TRAIT_CONFIG_COMMIT
from lfg_fly.teacher.report import ATTRIBUTION
from lfg_fly.teacher.rewire import rewired_graph, sign_shuffled_graph
from lfg_fly.teacher.sample import Look

log = logging.getLogger(__name__)

GATE_CI_LO = 0.55  # spec §3.4 go-live gate: family-bootstrap 95% CI lower bound above this
WINNERS = ("A", "B")
# Phase 0's winner (data/probe/verdict.json at d8e7279): the fall-back when no verdict is found
PHASE0_WINNER = G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.1, steps=60),
                          "all-sensory-brain", 2.0)
SENSE_CODES = ("eyes", "nose")  # spec §3.4 row 6
SYN5 = 5  # spec §3.4 row 7
CONTROL_NAMES = ("no_brain", "mlp", "rewired", "sign_shuffled", "eyes_only", "nose_only", "syn5")
RESULTS = "results.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- labels (spec §3.2 → §3.3)


@dataclass
class LabelSet:
    """The critic's judgments as a probe set. Pair i says look `a_idx[i]` beats `b_idx[i]`
    iff `y[i] == 1`; `w[i]` is the critic's strength; `test[i]` marks a held-out family."""

    looks: list[Look]
    a_idx: np.ndarray
    b_idx: np.ndarray
    family: np.ndarray
    near: np.ndarray
    y: np.ndarray  # float32: 1.0 iff the critic chose a
    w: np.ndarray  # float32: strength 1-3
    test: np.ndarray  # bool: the held-out families (>= test_frac of them)
    n_dropped: int  # pairs the critic flipped on (position bias), never trained or scored
    pair_id: np.ndarray | None = None
    source: str = ""

    @property
    def n_pairs(self) -> int:
        return int(len(self.y))

    @property
    def n_train(self) -> int:
        return int((~self.test).sum())

    @property
    def n_test(self) -> int:
        return int(self.test.sum())


def _check_look(values: list, cat: Catalog, where: str) -> Look:
    if not isinstance(values, list) or len(values) != len(SLOTS):
        raise ValueError(f"{where}: a look has {len(SLOTS)} values in SLOTS order, "
                         f"got {values!r}")
    look = tuple(str(v) for v in values)
    for slot, value in zip(SLOTS, look, strict=True):
        if value not in cat.values.get(slot, ()):
            raise ValueError(f"{where}: {slot}={value!r} is not in the catalog")
    return look


def load_labels(jsonl: Path, cat: Catalog, test_frac: float = 0.2, seed: int = 0) -> LabelSet:
    """`data/critic/<round>.jsonl` → `LabelSet` (spec §3.2 quality control, §3.4 metrics).

    Pairs with `flipped: true` are dropped (the critic chose the left side both times).
    Every look must be in the catalog: the one-hot control and the brain's codes need
    that, and a look outside it means the wrong catalog. The test split holds
    `ceil(test_frac × families)` families, drawn from `seed`; no family straddles it.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    jsonl = Path(jsonl)
    lines = [ln for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    index: dict[Look, int] = {}
    a_idx, b_idx, family, near, y, w, pair_ids = [], [], [], [], [], [], []
    dropped = 0
    for n, line in enumerate(lines, 1):
        where = f"{jsonl.name}:{n}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}: not JSON ({e.msg})") from None
        if r.get("flipped") is True:
            dropped += 1
            continue
        if r.get("winner") not in WINNERS:
            raise ValueError(f"{where}: winner must be A or B, got {r.get('winner')!r}")
        try:
            strength = int(r["strength"])
            fam = int(r["family"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{where}: missing or bad strength/family ({e})") from None
        if strength < 1:
            raise ValueError(f"{where}: strength must be >= 1, got {strength}")
        a = _check_look(r.get("a"), cat, where)
        b = _check_look(r.get("b"), cat, where)
        a_idx.append(index.setdefault(a, len(index)))
        b_idx.append(index.setdefault(b, len(index)))
        family.append(fam)
        near.append(bool(r.get("near", False)))
        y.append(1.0 if r["winner"] == "A" else 0.0)
        w.append(float(strength))
        pair_ids.append(int(r.get("pair_id", n - 1)))
    if not y:
        raise ValueError(f"{jsonl}: no usable pair ({dropped} flipped pair(s) dropped, "
                         f"{len(lines)} line(s))")
    family_arr = np.array(family, dtype=np.int64)
    fams = np.unique(family_arr)
    rng = np.random.default_rng(seed)
    order = rng.permutation(fams)
    test_fams = set(order[: math.ceil(test_frac * len(fams))].tolist())
    return LabelSet(
        looks=list(index),
        a_idx=np.array(a_idx, dtype=np.int64),
        b_idx=np.array(b_idx, dtype=np.int64),
        family=family_arr,
        near=np.array(near, dtype=bool),
        y=np.array(y, dtype=np.float32),
        w=np.array(w, dtype=np.float32),
        test=np.isin(family_arr, list(test_fams)),
        n_dropped=dropped,
        pair_id=np.array(pair_ids, dtype=np.int64),
        source=str(jsonl),
    )


# ---------------------------------------------------------------- the gate and Phase 0's settings


def gate(fly: dict) -> dict:
    """Spec §3.4 go-live gate: the taste head's family-bootstrap 95% CI lower bound above
    GATE_CI_LO. (Its best-of-6 criterion is round 2's measurement; see the report.)"""
    ci_lo = float(fly["ci_lo"])
    return {"ci_lo_min": GATE_CI_LO, "ci_lo": ci_lo, "pass": bool(ci_lo > GATE_CI_LO)}


def best_phase0_settings(stage_b_jsonl: Path) -> dict[str, dict]:
    """Per input code, Phase 0's best Stage B row (highest planted-taste held-out), passers
    preferred: spec §3.4 row 6 takes the eyes-only and nose-only brains from there. A
    code with no passer gets its best-probed row, flagged `passed: False`. Missing file
    → {}."""
    best: dict[str, dict] = {}
    for r in G.read_jsonl(Path(stage_b_jsonl)):
        code = r["setting"]["code"]
        rank = (bool(r.get("passed")), float(r["probe"]["heldout"]))
        cur = best.get(code)
        if cur is None or rank > (bool(cur.get("passed")), float(cur["probe"]["heldout"])):
            best[code] = r
    return best


def phase0_setting(verdict_json: Path) -> G.Setting:
    """The Phase 0 winner from `data/probe/verdict.json` (its confirmed best setting), or
    `PHASE0_WINNER` when there is no passing verdict to read."""
    verdict_json = Path(verdict_json)
    if not verdict_json.exists():
        log.warning("%s: no Phase 0 verdict; training the pinned winner %s", verdict_json,
                    PHASE0_WINNER.key())
        return PHASE0_WINNER
    v = json.loads(verdict_json.read_text(encoding="utf-8"))
    if G.verdict_status(v) != "PASS" or not v.get("best"):
        log.warning("%s: Phase 0 did not PASS there; training the pinned winner %s",
                    verdict_json, PHASE0_WINNER.key())
        return PHASE0_WINNER
    return C.setting_from_dict(v["best"]["setting"])


# ---------------------------------------------------------------- training (spec §3.3)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


def _evaluate(X: np.ndarray, labels: LabelSet, device: str, seed: int) -> P.ProbeResult:
    """The protocol: weighted BT, family-grouped CV, family-bootstrap CI (spec §2, §3.4)."""
    return P.evaluate(X, labels, device=device, seed=seed, w=labels.w)


def _timed(fn):
    t0 = time.time()
    out = fn()
    return out, round(time.time() - t0, 2)


def _head_check(head: R.TasteHead, X: np.ndarray, labels: LabelSet, fly: P.ProbeResult,
                seed: int) -> dict:
    """The shipped head on the same held-out pairs: it must be the model the protocol scored."""
    s = head.score(X)
    te = labels.test
    pred = s[labels.a_idx[te]] - s[labels.b_idx[te]] > 0
    correct = pred == (labels.y[te] > 0.5)
    lo, hi = P.family_bootstrap(correct, labels.family[te], seed=seed)
    return {"lam": float(head.lam), "taste_sd": float(head.taste_sd),
            "n_features": int(X.shape[1]), "heldout": float(correct.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "agreement_with_protocol": float((correct == fly.correct).mean())}


def _sim(ctx: G.Context) -> Simulator:
    return Simulator(ctx.graph, ctx.pops.sensory_any, ctx.pops.readout, ctx.device)


def _brain_control(ctx: G.Context, sim: Simulator, setting: G.Setting, labels: LabelSet,
                   images: dict, seed: int, extra: dict | None = None
                   ) -> tuple[P.ProbeResult, dict]:
    """One brain-based control: simulate, score by the protocol, one results row."""
    (X, stats), secs = _timed(lambda: G.features_for(ctx, sim, setting, labels.looks, images))
    res = _evaluate(X, labels, ctx.device, seed)
    return res, {**res.as_dict(), "stats": stats, "seconds": secs, **(extra or {})}


def _no_brain(ctx: G.Context, labels: LabelSet, seed: int) -> tuple[dict, dict, dict]:
    """Spec §3.4 rows 2 and 3 on one-hot traits + pixel stats (mean RGB, 8-bin luminance
    histogram, via `interact.render_looks`). Without a layer bank the pixels are left out
    and the row says so."""
    oh = P.one_hot(labels.looks, ctx.catalog)
    pixels = ctx.bank is not None and ctx.zorder is not None
    if pixels:
        pix, _ = render_looks(ctx, labels.looks)
        X = np.hstack([oh, pix])
    else:
        X = oh
    lr, s1 = _timed(lambda: _evaluate(X, labels, ctx.device, seed))
    mlp, s2 = _timed(lambda: P.evaluate_mlp(X, labels, device=ctx.device, seed=seed, w=labels.w))
    correct = {"no_brain": lr.correct, "mlp": mlp.correct}
    rows = {"no_brain": {**lr.as_dict(), "pixels": pixels, "n_features": int(X.shape[1]),
                         "seconds": s1},
            "mlp": {**mlp.as_dict(), "pixels": pixels, "hidden": P.MLP_HIDDEN,
                    "n_features": int(X.shape[1]), "seconds": s2}}
    return correct, rows, {}


def _syn5_context(ctx: G.Context) -> tuple[G.Context | None, str]:
    """Spec §3.4 row 7: the same brain on the ≥5-synapse graph, if `fly build --min-syn 5`
    left one in the data dir (with its columns file, else the context's retina)."""
    gdir = paths.graph_dir()
    gpath = gdir / f"graph-syn{SYN5}.npz"
    if not gpath.exists():
        return None, f"no graph-syn{SYN5}.npz in the data dir"
    if ctx.graph.min_syn == SYN5:
        return None, f"the fly's own graph is already the ≥{SYN5}-synapse graph"
    g5 = B.load_graph(gpath)
    pops5 = populations(g5)
    cpath = gdir / f"columns-syn{SYN5}.npz"
    retina = build_retina(load_columns(cpath), g5) if cpath.exists() else ctx.retina
    return dataclasses.replace(ctx, graph=g5, pops=pops5, retina=retina), ""


def train(ctx: G.Context, labels: LabelSet, setting: G.Setting, out_dir: Path,
          controls: bool = True, *, phase0_stage_b: Path | None = None) -> dict:
    """Spec §3.3 training and §3.4 controls. Writes `out_dir/results.json` and the head
    (`out_dir/head.npz` + `head.json`, via `readout.save_head`); returns the results.

    The fly: every labelled look simulated once under `c_ref` (spec §2: taste is
    rarity-invariant by construction), scored by `probe.evaluate` with the critic's
    strength as the weight, and the head fitted on the training pairs by
    `readout.fit_taste` (the same weighted BT), then checked on the held-out pairs.
    Controls (rows 2–7), each on the same held-out pairs with a paired family-bootstrap
    difference: one-hot + pixel LR, MLP, rewired and sign-shuffled twins (Phase 0's
    seeds), eyes-only and nose-only (Phase 0's best Stage B setting per code, from
    `phase0_stage_b`), and the ≥5-synapse graph when it exists. A control that cannot
    run is recorded as `{"skipped": reason}`, never silently left out.
    """
    out_dir = Path(out_dir)
    seed = int(ctx.seed)
    t_all = time.time()
    tr = ~labels.test
    images: dict = {}
    sim = _sim(ctx)
    (X, stats), fly_secs = _timed(lambda: G.features_for(ctx, sim, setting, labels.looks, images))
    fly = _evaluate(X, labels, ctx.device, seed)
    head = R.fit_taste(X, labels.a_idx[tr], labels.b_idx[tr], labels.y[tr], labels.w[tr],
                       labels.family[tr], ctx.device)
    head_row = _head_check(head, X, labels, fly, seed)
    R.save_head(out_dir, head)

    ctl: dict[str, dict] = {}
    correct: dict[str, np.ndarray] = {}
    repairs = None
    if controls:
        c_ok, rows, _ = _no_brain(ctx, labels, seed)
        correct.update(c_ok)
        ctl.update(rows)
        # row 6: the senses one at a time, Phase 0's best setting for each, on the real graph
        stage_b = (Path(phase0_stage_b) if phase0_stage_b is not None
                   else paths.repo_root() / "data" / "probe" / "stage_b.jsonl")
        best = best_phase0_settings(stage_b)
        for code in SENSE_CODES:
            name = f"{code}_only"
            row = best.get(code)
            if row is None:
                ctl[name] = {"skipped": f"no Phase 0 Stage B row for code {code!r} in {stage_b}"}
                continue
            if code == "eyes" and ctx.bank is None:
                ctl[name] = {"skipped": "no layer bank in the context: the eyes cannot see the "
                                        "looks"}
                continue
            s = C.setting_from_dict(row["setting"])
            res, ctl[name] = _brain_control(ctx, sim, s, labels, images, seed, {
                "setting": s.as_dict(), "phase0_heldout": float(row["probe"]["heldout"]),
                "phase0_passed": bool(row.get("passed"))})
            correct[name] = res.correct
        del sim  # one simulator on the GPU at a time
        # rows 4 and 5: Phase 0's twins (same seeds), the fly's own setting
        rewired, repairs = rewired_graph(ctx.graph, seed=seed + 11, device=ctx.device)
        twins = (("rewired", rewired), ("sign_shuffled", sign_shuffled_graph(ctx.graph, seed + 13)))
        for name, graph in twins:
            tctx = dataclasses.replace(ctx, graph=graph)
            tsim = _sim(tctx)
            res, ctl[name] = _brain_control(tctx, tsim, setting, labels, images, seed,
                                            {"graph_hash": graph.graph_hash()})
            correct[name] = res.correct
            del tsim
        del rewired, twins
        # row 7: the threshold
        ctx5, why = _syn5_context(ctx)
        if ctx5 is None:
            ctl["syn5"] = {"skipped": why}
        else:
            sim5 = _sim(ctx5)
            res, ctl["syn5"] = _brain_control(ctx5, sim5, setting, labels, images, seed, {
                "graph_hash": ctx5.graph.graph_hash(), "min_syn": int(ctx5.graph.min_syn),
                "edges": int(len(ctx5.graph.col))})
            correct["syn5"] = res.correct
            del sim5, ctx5
        missing = [n for n in CONTROL_NAMES if n not in ctl]
        for n in missing:  # never silently absent
            ctl[n] = {"skipped": "not attempted"}
    fam_te = labels.family[labels.test]
    paired = {n: P.paired_diff(fly.correct, c, fam_te, seed=seed) for n, c in correct.items()}
    fams = np.unique(labels.family)
    results = {
        "created_at": _now(), "device": ctx.device, "seed": seed,
        "labels": {"source": labels.source, "n_pairs": labels.n_pairs,
                   "n_dropped": int(labels.n_dropped), "n_train": labels.n_train,
                   "n_test": labels.n_test, "n_families": int(len(fams)),
                   "n_test_families": int(len(np.unique(fam_te))),
                   "near_frac": float(labels.near.mean()),
                   "n_looks": len(labels.looks)},
        "setting": setting.as_dict(), "c_ref": C_REF,
        "graph": {"hash": ctx.graph.graph_hash(), "min_syn": int(ctx.graph.min_syn),
                  "n": int(ctx.graph.n), "edges": int(len(ctx.graph.col)),
                  "nt_missing": int(ctx.graph.nt_missing)},
        "fly": {**fly.as_dict(), "stats": stats, "seconds": fly_secs},
        "head": {**head_row, "path": str(out_dir)},
        "controls": ctl, "paired": paired,
        "beats": {n: bool(d["lo"] > 0) for n, d in paired.items()},
        "rewire_repairs": None if repairs is None else int(repairs),
        "controls_run": bool(controls),
        "gate": gate(fly.as_dict()),
        "seconds": round(time.time() - t_all, 2),
    }
    _write_json(out_dir / RESULTS, results)
    return _jsonable(results)


# ---------------------------------------------------------------- the checkpoint (spec §2)


def _feathers() -> tuple[dict[str, str], bool]:
    """The feather sha256s from `fly fetch`'s manifest; ({}, False) when it is absent."""
    path = paths.raw_dir() / "manifest.json"
    if not path.exists():
        log.warning("%s: no fetch manifest; the checkpoint pins no feather sha256", path)
        return {}, False
    m = json.loads(path.read_text(encoding="utf-8"))
    pins = {k: str(v["sha256"]) for k, v in m.items() if isinstance(v, dict) and "sha256" in v}
    return pins, True


def _columns_hash(min_syn: int) -> str | None:
    path = paths.graph_dir() / f"columns-syn{min_syn}.npz"
    return C.columns_hash(load_columns(path)) if path.exists() else None


def write_checkpoint(ctx: G.Context, setting: G.Setting, head: R.TasteHead, results: dict,
                     version: str, *, directory: Path | None = None,
                     feathers: dict[str, str] | None = None) -> tuple[Path, C.Manifest]:
    """`checkpoints/<version>/` (spec §2 Checkpoint and versioning): the manifest pins the
    graph hash, the feather sha256s (`FLY_DATA_DIR/raw/manifest.json`), the threshold,
    the sign map, the NT-missing count, the column assignment, the setting, `c_ref`, the
    sense-code versions, the LFG trait_config commit, the catalog and the taste numbers
    with the gate verdict. Returns the directory and the manifest; `results` gains a
    `checkpoint` entry."""
    default_dir = paths.checkpoint_dir(version)  # validates the version component
    directory = default_dir if directory is None else Path(directory)
    known = True
    if feathers is None:
        feathers, known = _feathers()
    fly = results["fly"]
    taste = {"heldout": fly["heldout"], "ci_lo": fly["ci_lo"], "ci_hi": fly["ci_hi"],
             "lam": float(head.lam), "n_train": results["labels"]["n_train"],
             "n_test": results["labels"]["n_test"], "gate_pass": bool(results["gate"]["pass"]),
             "taste_sd": float(head.taste_sd), "head_heldout": results["head"]["heldout"],
             "labels": results["labels"]["source"], "round": results.get("round")}
    manifest = C.Manifest(
        version=version, graph_hash=ctx.graph.graph_hash(), feathers=dict(feathers),
        min_syn=int(ctx.graph.min_syn), sign_map=dict(B.SIGN),
        nt_missing=int(ctx.graph.nt_missing), columns_hash=_columns_hash(ctx.graph.min_syn),
        setting=setting.as_dict(), c_ref=C_REF, codes=dict(C.DEFAULT_CODES),
        lfg_trait_config_commit=LFG_TRAIT_CONFIG_COMMIT, catalog_hash=C.catalog_hash(ctx.catalog),
        taste=taste)
    C.write_checkpoint(directory, manifest, head)
    results["checkpoint"] = {"dir": str(directory), "feathers_known": bool(known),
                             "manifest": manifest.to_dict()}
    return directory, manifest


# ---------------------------------------------------------------- REPORT.md (spec §3.4)


def _f(x, digits: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _ci(r: dict) -> str:
    return f"{_f(r.get('ci_lo'))}–{_f(r.get('ci_hi'))}"


def _diff(d: dict | None) -> tuple[str, str, str]:
    if d is None:
        return "—", "—", "—"
    return _f(d["diff"], sign=True), f"{_f(d['lo'], sign=True)} to {_f(d['hi'], sign=True)}", (
        "yes" if d["lo"] > 0 else "no")


def _setting_words(s: dict) -> str:
    b = s["brain"]
    return (f"{b['kind']}, g_syn {b['g_syn']}, bias {b['bias']}, {b.get('steps', '—')} steps, "
            f"code `{s['code']}`, g_in {s['g_in']}")


def _control_label(name: str, row: dict, results: dict) -> str:
    if name == "no_brain":
        if row.get("pixels") is False:
            return ("No brain: logistic regression on one-hot traits (one-hot only: no layer "
                    "bank, so no mean RGB or luminance histogram)")
        return ("No brain: logistic regression on one-hot traits + mean RGB + 8-bin luminance "
                "histogram")
    if name == "mlp":
        return f"MLP ({row.get('hidden', P.MLP_HIDDEN)} hidden) on the same raw features"
    if name == "rewired":
        rep = results.get("rewire_repairs")
        return "Rewired fly (degree-preserving" + (
            f"; {rep:,} repair swaps)" if rep is not None else ")")
    if name == "sign_shuffled":
        return "Sign-shuffled fly (NT labels permuted across neurons)"
    if name in ("eyes_only", "nose_only"):
        label = "Eyes-only (on the tonic baseline)" if name == "eyes_only" else "Nose-only"
        if "setting" in row:
            label += (f": Phase 0's best `{name[:4]}` setting, {_setting_words(row['setting'])}"
                      f", planted-taste held-out {_f(row.get('phase0_heldout'))}")
        return label
    if name == "syn5":
        label = f"≥{SYN5}-synapse graph"
        if "graph_hash" in row:
            label += f" (hash {row['graph_hash'][:16]}…, {row.get('edges', 0):,} edges)"
        return label
    return name


def _answers(results: dict) -> list[str]:
    """Spec §3.4's questions, answered plainly whichever way they fall."""
    fly, ctl, paired = results["fly"], results["controls"], results["paired"]

    def have(name: str) -> bool:
        return name in ctl and "skipped" not in ctl[name]

    def vs(name: str) -> str:
        d = paired[name]
        return (f"{_f(ctl[name]['heldout'])} ({_ci(ctl[name])}) against the fly's "
                f"{_f(fly['heldout'])} ({_ci(fly)}); paired difference fly − model "
                f"{_f(d['diff'], sign=True)} (95% CI {_f(d['lo'], sign=True)} to "
                f"{_f(d['hi'], sign=True)})")

    def not_run(name: str) -> str:
        return f"Not answered: {ctl.get(name, {}).get('skipped', 'the control was not run')}."

    out = ["", "## What the controls say", "",
           "A control is beaten when the paired difference's 95% CI lower bound is above 0. "
           "Beating the controls is not required for the gate (spec §3.4); whether the fly "
           "does is stated here either way.", ""]
    # raw inputs
    if have("no_brain"):
        beats = paired["no_brain"]["lo"] > 0
        out.append(f"- **Does the fly beat its own raw inputs?** "
                   f"{'Yes' if beats else 'No'}: the fly "
                   f"{'beats' if beats else 'does not beat'} its own raw inputs. The no-brain "
                   f"regression scores {vs('no_brain')}.")
    else:
        out.append(f"- **Does the fly beat its own raw inputs?** {not_run('no_brain')}")
    if have("mlp"):
        beats = paired["mlp"]["lo"] > 0
        out.append(f"- **Against a conventional net:** the MLP scores {vs('mlp')}. The fly "
                   f"{'beats' if beats else 'does not beat'} it.")
    else:
        out.append(f"- **Against a conventional net:** {not_run('mlp')}")
    # wiring
    if have("rewired"):
        matters = paired["rewired"]["lo"] > 0
        out.append(f"- **Does the real wiring matter?** "
                   f"{'Yes: the wiring matters' if matters else 'No: the wiring does not matter'}"
                   f" for this taste. The rewired twin (every neuron's in- and out-degree "
                   f"preserved) scores {vs('rewired')}."
                   + ("" if matters else " Every published fly-game control so far found no "
                                         "advantage for real wiring, and neither does this one."))
    else:
        out.append(f"- **Does the real wiring matter?** {not_run('rewired')}")
    if have("sign_shuffled"):
        matters = paired["sign_shuffled"]["lo"] > 0
        out.append(f"- **Do the signs matter?** "
                   f"{'Yes: the signs matter' if matters else 'No: the signs do not matter'}. "
                   f"The sign-shuffled twin scores {vs('sign_shuffled')}.")
    else:
        out.append(f"- **Do the signs matter?** {not_run('sign_shuffled')}")
    # senses
    senses = {n: ctl[n]["heldout"] for n in ("eyes_only", "nose_only") if have(n)}
    if len(senses) == 2:
        e, n_ = senses["eyes_only"], senses["nose_only"]
        lead = ("the nose carries more of it than the eyes" if n_ > e
                else "the eyes carry more of it than the nose" if e > n_
                else "eyes and nose carry it equally")
        out.append(f"- **Which sense carries taste?** Eyes-only {_f(e)} ({_ci(ctl['eyes_only'])}), "
                   f"nose-only {_f(n_)} ({_ci(ctl['nose_only'])}), the fly (all senses) "
                   f"{_f(fly['heldout'])}: {lead}. Paired differences fly − eyes "
                   f"{_f(paired['eyes_only']['diff'], sign=True)}, fly − nose "
                   f"{_f(paired['nose_only']['diff'], sign=True)}.")
    elif senses:
        (name, h), = senses.items()
        other = "nose_only" if name == "eyes_only" else "eyes_only"
        out.append(f"- **Which sense carries taste?** Only {name.replace('_', '-')} was run: "
                   f"{_f(h)} ({_ci(ctl[name])}) against the fly's {_f(fly['heldout'])}; "
                   f"{other.replace('_', '-')} {not_run(other).lower()}")
    else:
        out.append(f"- **Which sense carries taste?** {not_run('eyes_only')} "
                   f"{not_run('nose_only')}")
    # threshold
    if have("syn5"):
        d = paired["syn5"]["diff"]
        out.append(f"- **Threshold sensitivity:** the ≥{SYN5}-synapse graph scores {vs('syn5')}. "
                   f"The threshold moves the number by {_f(-d, sign=True)}.")
    else:
        out.append(f"- **Threshold sensitivity:** {not_run('syn5')}")
    beaten = sum(1 for n in paired if paired[n]["lo"] > 0)
    out += ["", f"The fly beats {beaten} of the {len(paired)} control(s) that ran."]
    return out


def _qc_lines(qc: dict, round_name: str, fly: dict) -> list[str]:
    out = ["", "## The critic's labels (spec §3.2 quality control)", ""]
    if not qc:
        return out + [f"No QC file for round {round_name} (`data/critic/{round_name}-qc.json`): "
                      "the flip rate, agreement and implied ceiling are not reported here."]
    n_flip, n_dup = qc.get("n_flip_checked", 0), qc.get("n_dup_checked", 0)
    out += [f"- Pairs judged: {qc.get('n_pairs', '—')}; A chosen in "
            f"{_f(qc.get('a_wins'))} of them (position bias shows as a share far from 0.5).",
            f"- Flip rate {_f(qc.get('flip_rate'))}: {qc.get('n_dropped', '—')} of {n_flip} "
            "flip-checked pairs got the opposite verdict with the sides swapped; those pairs "
            "were dropped from training and scoring.",
            f"- Agreement a = {_f(qc.get('agreement'))} over {n_dup} pairs judged twice; "
            f"implied ceiling (1+√(2a−1))/2 = {_f(qc.get('implied_ceiling'))}. Agreement is a "
            "lower bound on what a model can reach against one critic; the ceiling is exact "
            "under uniform label noise and an upper estimate when pair difficulty varies."]
    ceiling = qc.get("implied_ceiling")
    if ceiling is not None and not (isinstance(ceiling, float) and math.isnan(ceiling)):
        rel = ("below" if fly["heldout"] < ceiling else "at or above")
        out.append(f"- The fly's held-out accuracy {_f(fly['heldout'])} is {rel} the implied "
                   f"ceiling {_f(ceiling)}.")
    hist = qc.get("strength_hist")
    if hist:
        out.append("- Strength histogram (1–3): "
                   + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items())) + ".")
    return out


def _identity(results: dict) -> list[str]:
    out = ["", "## Checkpoint identity (spec §2 Checkpoint and versioning)", ""]
    ck = results.get("checkpoint")
    if not ck:
        return out + ["The checkpoint is not written yet."]
    m = ck["manifest"]
    out += [f"- Checkpoint: `{ck['dir']}` (version `{m['version']}`), created {m['created_at']}.",
            f"- Graph hash `{m['graph_hash']}`; threshold ≥{m['min_syn']} synapses; "
            f"{m['nt_missing']:,} neurons without an NT prediction (treated as `unclear`).",
            "- Feathers (sha256): " + (", ".join(f"{k} `{v}`" for k, v in
                                                sorted(m["feathers"].items()))
                                       if m["feathers"] else
                                       "not pinned (no `raw/manifest.json` in the data dir)")
            + ".",
            f"- Column assignment hash: `{m['columns_hash'] or 'not pinned'}`; catalog hash "
            f"`{m['catalog_hash'] or 'not pinned'}`.",
            f"- Brain setting: {_setting_words(results['setting'])}; c_ref {m['c_ref']}; sense "
            f"codes {m['codes']}; LFG trait_config commit `{m['lfg_trait_config_commit']}`.",
            "- Sign map and the head's standardization stats and weights are in the checkpoint "
            "(`manifest.json`, `head.npz`, `head.json`)."]
    return out


def write_report(results: dict, qc: dict, out_md: Path) -> None:
    """REPORT.md (spec §3.4): the fly's held-out accuracy and CI, every control with its CI
    and paired difference, the gate verdict, the critic's QC and the checkpoint identity.
    Plain statements either way."""
    fly, ctl, paired, g = results["fly"], results["controls"], results["paired"], results["gate"]
    labels = results["labels"]
    version = results.get("version", "fly-v1")
    round_name = results.get("round", "r1")
    verdict = "PASS" if g["pass"] else "FAIL"
    lines = [f"# REPORT: the fly's taste, `{version}`", "",
             f"Written {results.get('created_at', '—')} by `fly train --round {round_name}` on "
             f"`{results.get('device', '—')}`. Every accuracy is pairwise, on held-out look "
             "families the head never saw; every interval is a 95% family-bootstrap CI (2,000 "
             "resamples) and every fly − model difference is a paired bootstrap over the same "
             "families (spec §3.4).", "",
             "## Gate B", "",
             f"**Gate B: {verdict}** — the taste head's held-out accuracy is {_f(fly['heldout'])} "
             f"(95% CI {_ci(fly)}, n = {fly['n_test']:,} pairs over "
             f"{labels.get('n_test_families', '—')} families). The go-live gate needs the CI "
             f"lower bound above {GATE_CI_LO:.2f}; {_f(g['ci_lo'])} is "
             f"{'above it' if g['pass'] else 'not above it'}. "
             + ("The fly goes live as a chooser (spec §3.4)."
                if g["pass"] else
                "The fly does not go live as a chooser (spec §3.4); the operator chooses as in "
                "spec §3.0."), "",
             "Best-of-6 agreement against the 1/6 chance rate, the gate's second criterion, is "
             "round 2's measurement and has not been made; this verdict rests on the CI "
             "criterion alone.", "",
             "## The fly", "",
             f"- Brain: {_setting_words(results['setting'])}, simulated at c_ref "
             f"{results.get('c_ref', C_REF)} (taste is rarity-invariant by construction, "
             "spec §2).",
             f"- Graph hash `{results['graph']['hash'][:16]}…`, ≥{results['graph']['min_syn']} "
             f"synapses, {results['graph']['n']:,} neurons, {results['graph']['edges']:,} edges.",
             f"- Held-out {_f(fly['heldout'])} ({_ci(fly)}); training accuracy "
             f"{_f(fly.get('train_acc'))}; CV accuracy {_f(fly.get('cv_acc'))}; L2 λ = "
             f"{fly.get('lam')}; readout features {results['head'].get('n_features', '—')}.",
             f"- The shipped head (fitted on the training pairs, the model the protocol scored) "
             f"gets {_f(results['head']['heldout'])} ({_f(results['head'].get('ci_lo'))}–"
             f"{_f(results['head'].get('ci_hi'))}) on the same held-out pairs and agrees with "
             f"the protocol's per-pair verdicts on "
             f"{100 * results['head'].get('agreement_with_protocol', float('nan')):.1f}% of "
             f"them; taste sd {_f(results['head']['taste_sd'])} (β's unit in the decision "
             "score).",
             f"- Activity: mean active fraction {_f(fly.get('stats', {}).get('active_frac'))}, "
             f"readout active {_f(fly.get('stats', {}).get('readout_active_frac'))}; "
             f"{fly.get('seconds', '—')} s of simulation.",
             "", "## Labels", "",
             f"- Source `{labels.get('source', '—')}` (round {round_name}): "
             f"{labels['n_pairs'] + labels['n_dropped']:,} pairs judged, "
             f"{labels['n_dropped']:,} dropped as position-bias flips, {labels['n_pairs']:,} "
             f"kept over {labels.get('n_families', '—')} look families "
             f"({labels.get('n_looks', '—')} distinct looks); near pairs "
             f"{_f(labels.get('near_frac'))}.",
             f"- Split by family: {labels['n_train']:,} training pairs, {labels['n_test']:,} "
             f"held-out pairs in {labels.get('n_test_families', '—')} families (≥ 20% of "
             "families).",
             "- The critic's strength (1–3) is the sample weight; the L2 strength is chosen by "
             "5-fold cross-validation grouped by family.",
             "", "## Controls (spec §3.4)", ""]
    if not results.get("controls_run", bool(ctl)):
        lines += ["**No control was run** (`--no-controls`): every row below says so.", ""]
    lines += ["| # | Model | Held-out | 95% CI | Fly − model | 95% CI | Fly beats it |",
              "|---|---|---|---|---|---|---|"]
    beats_random = fly["ci_lo"] > 0.5
    lines.append(f"| 1 | Random mover | 0.500 | — | {_f(fly['heldout'] - 0.5, sign=True)} | "
                 f"fly's own CI {_ci(fly)} | {'yes' if beats_random else 'no'} |")
    numbers = {"no_brain": 2, "mlp": 3, "rewired": 4, "sign_shuffled": 5, "eyes_only": 6,
               "nose_only": 6, "syn5": 7}
    for name in CONTROL_NAMES:
        row = ctl.get(name)
        if row is None or "skipped" in row:
            why = row["skipped"] if row else "not run"
            lines.append(f"| {numbers[name]} | {_control_label(name, row or {}, results)} | "
                         f"not run: {why} | — | — | — | — |")
            continue
        d, dci, yes = _diff(paired.get(name))
        lines.append(f"| {numbers[name]} | {_control_label(name, row, results)} | "
                     f"{_f(row['heldout'])} | {_ci(row)} | {d} | {dci} | {yes} |")
    if qc:
        lines.append(f"| 8 | Critic agreement a / implied ceiling | a = {_f(qc.get('agreement'))} "
                     f"→ ceiling {_f(qc.get('implied_ceiling'))} | — | — | — | — |")
    else:
        lines.append("| 8 | Critic agreement a / implied ceiling | no QC file | — | — | — | — |")
    lines += _answers(results) if results.get("controls_run", bool(ctl)) else [
        "", "## What the controls say", "", "Nothing: no control was run."]
    lines += _qc_lines(qc, round_name, fly)
    lines += ["", "## The rarity head", "",
              "Not trained here. The rarity head (spec §2) is fitted nightly by `fly retrain` "
              "from a supply snapshot and learns an additive, already-solved function "
              "(Σ n_live/freq over the nine traits). Rarity is the fly's incentive; taste is "
              "where the fly is tested, and this report scores taste only."]
    lines += _identity(results)
    lines += ["", "## Reproduce", "",
              f"`fly train --round {round_name} --version {version} --device "
              f"{results.get('device', 'cuda')} --min-syn {results['graph']['min_syn']}` "
              f"(seed {results.get('seed', 0)}; results in `data/train/{version}/results.json`).",
              "", "---", "", ATTRIBUTION, ""]
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_md.with_name(out_md.name + ".tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(out_md)


# ---------------------------------------------------------------- the round, end to end


def round_paths(root: Path, round_name: str, version: str) -> dict[str, Path]:
    """Where a round's inputs and outputs live under a repo root."""
    ck_name = paths.checkpoint_dir(version).name  # validates the version component
    return {"labels": root / "data" / "critic" / f"{round_name}.jsonl",
            "qc": root / "data" / "critic" / f"{round_name}-qc.json",
            "phase0_stage_b": root / "data" / "probe" / "stage_b.jsonl",
            "phase0_verdict": root / "data" / "probe" / "verdict.json",
            "out_dir": root / "data" / "train" / ck_name,
            "checkpoint": root / "checkpoints" / ck_name,
            "report": root / "REPORT.md"}


def train_round(ctx: G.Context, setting: G.Setting, *, round_name: str = "r1",
                version: str = "fly-v1", root: Path | None = None,
                controls: bool = True) -> dict:
    """Labels → train → checkpoint → REPORT.md, under `root` (default: the repo)."""
    p = round_paths(Path(root) if root is not None else paths.repo_root(), round_name, version)
    labels = load_labels(p["labels"], ctx.catalog)
    results = train(ctx, labels, setting, p["out_dir"], controls=controls,
                    phase0_stage_b=p["phase0_stage_b"])
    results["round"], results["version"] = round_name, version
    head = R.load_head(p["out_dir"])
    write_checkpoint(ctx, setting, head, results, version, directory=p["checkpoint"])
    qc = json.loads(p["qc"].read_text(encoding="utf-8")) if p["qc"].exists() else {}
    write_report(results, qc, p["report"])
    _write_json(p["out_dir"] / RESULTS, results)
    return _jsonable(results)
