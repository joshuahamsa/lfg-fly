"""The heads: the only trained part of the fly (spec §2 Readout, Decision score).

- `TasteHead`: Bradley–Terry logistic regression on standardized readout features,
  L2 by family-grouped cross-validation, the critic's strength as the sample
  weight (`probe.fit_bt(..., w=)`). Taste simulations pin every concentration to
  `c_ref`, so taste cannot depend on rarity; that pin lives in the caller
  (`checkpoint.FlyBrain.taste`). `taste_sd`, the score's standard deviation over
  the training looks, is stored so the decision score can express greed β in
  units of taste's standard deviation.
- `RarityHead`: ridge regression from standardized features (a second, live-
  concentration simulation) to the look's standardized `nft_rarity`, retrained
  nightly and named by its supply-snapshot hash.
- `concentrations`: the nose's concentration-as-rarity curve `c = 0.4 + 0.6·p`.
- `save_head` / `load_head`: arrays in an .npz, scalars in a .json beside it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import probe as P

SD_FLOOR = 1e-3  # probe._standardize's floor: a constant feature standardizes to 0, not inf
C_MIN, C_SPAN = 0.4, 0.6  # c = C_MIN + C_SPAN * p
HEAD_NPZ, HEAD_JSON = "head.npz", "head.json"


def _stats(X: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and (floored) standard deviation over `rows`, in float64."""
    sub = np.asarray(X, dtype=np.float64)[rows]
    mu = sub.mean(0)
    sd = sub.std(0)
    sd[sd < SD_FLOOR] = SD_FLOOR
    return mu, sd


def _standardized(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(mu):
        raise ValueError(f"features have shape {X.shape}, the head expects (n, {len(mu)})")
    return (X - mu) / sd


@dataclass
class TasteHead:
    """`score(X) = ((X - mu) / sd) @ w`: the fly's taste for each look (spec §2 Readout)."""

    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    lam: float
    taste_sd: float

    def score(self, X: np.ndarray) -> np.ndarray:
        return _standardized(X, self.mu, self.sd) @ self.w


def fit_taste(X: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray, y: np.ndarray,
              weight: np.ndarray | None, family: np.ndarray, device: str) -> TasteHead:
    """Weighted Bradley–Terry (spec §2 Readout) via `probe.fit_bt`.

    `X` [n_looks, d] readout features under c_ref; pair i says look `a_idx[i]` beats
    `b_idx[i]` iff `y[i] == 1`; `weight[i]` is the critic's strength (None = flat);
    `family[i]` groups the CV folds. Features are standardized over the looks that
    appear in a pair, and `taste_sd` is the score's standard deviation over them.
    """
    X = np.asarray(X, dtype=np.float32)
    a = np.asarray(a_idx, dtype=np.int64)
    b = np.asarray(b_idx, dtype=np.int64)
    if not (len(a) == len(b) == len(y) == len(family)):
        raise ValueError("a_idx, b_idx, y and family must have one entry per pair")
    rows = np.unique(np.concatenate([a, b]))
    mu, sd = _stats(X, rows)
    Xs = _standardized(X, mu, sd).astype(np.float32)
    D = torch.as_tensor(Xs[a] - Xs[b], device=device)
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    w_t, lam, _ = P.fit_bt(D, y_t, np.asarray(family), device, w=weight)
    head = TasteHead(mu=mu, sd=sd, w=w_t.cpu().numpy().astype(np.float64), lam=float(lam),
                     taste_sd=0.0)
    head.taste_sd = float(head.score(X[rows]).std())
    return head


@dataclass
class RarityHead:
    """`score(X) = ((X - mu) / sd) @ w + b`: the look's standardized `nft_rarity`, read from a
    live-concentration simulation (spec §2 Readout, rarity head)."""

    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    snapshot_hash: str
    target_mu: float
    target_sd: float

    def score(self, X: np.ndarray) -> np.ndarray:
        return _standardized(X, self.mu, self.sd) @ self.w + self.b


def fit_rarity(X: np.ndarray, target: np.ndarray, snapshot_hash: str,
               lam: float = 1.0) -> RarityHead:
    """Ridge regression (spec §2 Readout, rarity head): minimize
    `||Xs w + b - z||² + lam ||w||²`, with `Xs` the standardized features and `z` the
    standardized target (`nft_rarity` per look). Closed form, float64."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if X.ndim != 2 or len(t) != len(X):
        raise ValueError("X must be [n_looks, d] with one target per look")
    if lam <= 0:
        raise ValueError("lam must be positive")
    rows = np.arange(len(X))
    mu, sd = _stats(X, rows)
    Xs = (X - mu) / sd
    t_mu = float(t.mean())
    t_sd = float(t.std())
    if t_sd < 1e-12:
        t_sd = 1.0
    z = (t - t_mu) / t_sd
    # centered design: the intercept is the residual mean
    xm, zm = Xs.mean(0), z.mean()
    Xc, zc = Xs - xm, z - zm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xc.T @ zc)
    b = float(zm - xm @ w)
    return RarityHead(mu=mu, sd=sd, w=w, b=b, snapshot_hash=str(snapshot_hash),
                      target_mu=t_mu, target_sd=t_sd)


def _trait_rarity(n_live: float, freq: float) -> float:
    """n_live / freq; a value nobody live wears is infinitely rare."""
    return math.inf if freq <= 0 else n_live / freq


def concentrations(looks, supply: dict, catalog) -> np.ndarray:
    """Per-trait odour concentrations `c = 0.4 + 0.6·p` [n, 9] (spec §2, Nose).

    `p` is the trait's percentile of `n_live / freq` within its slot: the fraction of
    the slot's values whose rarity is at most this value's, so the rarest value (and
    every value with zero live supply, whose rarity is infinite) gets p = 1 and the
    most common gets 1/K. The slot's values are the catalog's plus every value the
    supply snapshot counts for that slot, so the curve depends only on those two
    stored inputs, never on which looks are being scored. `supply` is
    `GET /api/rarity/supply`: `{"n_live": int, "counts": {slot: {value: int}}}`.
    """
    n_live = float(supply.get("n_live", 0) or 0)
    counts = supply.get("counts") or {}
    out = np.empty((len(looks), len(SLOTS)), dtype=np.float32)
    for s, slot in enumerate(SLOTS):
        slot_counts = counts.get(slot) or {}
        values = set(catalog.values.get(slot, [])) | set(slot_counts)
        rarity = {v: _trait_rarity(n_live, float(slot_counts.get(v, 0) or 0)) for v in values}
        ranked = np.sort(np.array(list(rarity.values()), dtype=np.float64))
        for j, look in enumerate(looks):
            r = rarity.get(look[s], math.inf)
            p = np.searchsorted(ranked, r, side="right") / len(ranked) if len(ranked) else 1.0
            out[j, s] = C_MIN + C_SPAN * float(p)
    return out


def nft_rarity(look, supply: dict) -> float:
    """LFG's `nft_rarity`: Σ over the nine traits of `n_live / freq`, with a value nobody
    live wears counted as `n_live` (freq 0 → treated as 1)."""
    n_live = float(supply.get("n_live", 0) or 0)
    counts = supply.get("counts") or {}
    total = 0.0
    for slot, value in zip(SLOTS, look, strict=True):
        freq = float((counts.get(slot) or {}).get(value, 0) or 0)
        total += n_live / freq if freq > 0 else n_live
    return total


def _head_files(path: Path) -> tuple[Path, Path]:
    """`path` is either an .npz file (its .json sits beside it) or a directory holding
    `head.npz` and `head.json`."""
    path = Path(path)
    if path.suffix == ".npz":
        return path, path.with_suffix(".json")
    return path / HEAD_NPZ, path / HEAD_JSON


def save_head(path: Path, head: TasteHead | RarityHead) -> None:
    """Arrays to .npz, scalars (and the head's kind) to .json, each written atomically."""
    npz, js = _head_files(path)
    npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"mu": head.mu, "sd": head.sd, "w": head.w}
    if isinstance(head, TasteHead):
        meta = {"kind": "taste", "lam": float(head.lam), "taste_sd": float(head.taste_sd)}
    elif isinstance(head, RarityHead):
        meta = {"kind": "rarity", "b": float(head.b), "snapshot_hash": head.snapshot_hash,
                "target_mu": float(head.target_mu), "target_sd": float(head.target_sd)}
    else:
        raise TypeError(f"not a head: {type(head).__name__}")
    tmp = npz.with_name(npz.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    tmp_js = js.with_name(js.name + ".tmp")
    tmp_js.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_js, js)


def load_head(path: Path) -> TasteHead | RarityHead:
    npz, js = _head_files(path)
    if not npz.exists() or not js.exists():
        raise FileNotFoundError(f"no head at {path} (need {npz.name} and {js.name})")
    meta = json.loads(js.read_text(encoding="utf-8"))
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k], dtype=np.float64) for k in ("mu", "sd", "w") if k in z}
    if set(arrays) != {"mu", "sd", "w"}:
        raise ValueError(f"{npz}: expected arrays mu, sd, w; found {sorted(arrays)}")
    kind = meta.get("kind")
    if kind == "taste":
        return TasteHead(**arrays, lam=float(meta["lam"]), taste_sd=float(meta["taste_sd"]))
    if kind == "rarity":
        return RarityHead(**arrays, b=float(meta["b"]), snapshot_hash=str(meta["snapshot_hash"]),
                          target_mu=float(meta["target_mu"]), target_sd=float(meta["target_sd"]))
    raise ValueError(f"{js}: unknown head kind {kind!r}")
