"""Score a feature map against the planted taste (spec §3.0, §3.4).

Bradley-Terry logistic regression on feature differences, L2 chosen by 5-fold
cross-validation grouped by look family, a held-out test split by family, and a
family-resampling bootstrap CI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.sample import ProbeSet

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


@dataclass
class ProbeResult:
    heldout: float
    ci_lo: float
    ci_hi: float
    train_acc: float
    cv_acc: float
    lam: float
    n_test: int

    def as_dict(self) -> dict:
        return asdict(self)


def _folds(groups: np.ndarray, k: int) -> np.ndarray:
    uniq = np.unique(groups)
    order = np.random.default_rng(12345).permutation(len(uniq))
    fold_of = {g: int(i % k) for i, g in zip(order, uniq, strict=True)}
    return np.array([fold_of[g] for g in groups])


def _fit(D: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    w = torch.zeros(D.shape[1], device=D.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(D @ w, y)
        loss = bce + lam * (w**2).sum() / len(y)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def _acc(D: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> float:
    return float(((D @ w > 0).float() == y).float().mean())


def fit_bt(D: torch.Tensor, y: torch.Tensor, groups: np.ndarray, device: str,
           lambdas: tuple[float, ...] = LAMBDAS,
           folds: int = 5) -> tuple[torch.Tensor, float, float]:
    fold = _folds(groups, folds)
    best = (-1.0, lambdas[-1])
    for lam in lambdas:
        accs = []
        for k in range(folds):
            tr = torch.as_tensor(np.flatnonzero(fold != k), device=device)
            va = torch.as_tensor(np.flatnonzero(fold == k), device=device)
            accs.append(_acc(D[va], y[va], _fit(D[tr], y[tr], lam)))
        score = float(np.mean(accs))
        if score > best[0] or (score == best[0] and lam > best[1]):
            best = (score, lam)
    return _fit(D, y, best[1]), best[1], best[0]


def family_bootstrap(correct: np.ndarray, family: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    fams = np.unique(family)
    per = {f: correct[family == f] for f in fams}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(fams, size=len(fams), replace=True)
        stats.append(np.concatenate([per[f] for f in pick]).mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _standardize(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    mu = X[rows].mean(0)
    sd = X[rows].std(0)
    sd[sd < 1e-3] = 1e-3
    return ((X - mu) / sd).astype(np.float32)


def evaluate(X: np.ndarray, ps: ProbeSet, device: str = "cpu", seed: int = 0) -> ProbeResult:
    train = ~ps.test
    rows = np.unique(np.concatenate([ps.a_idx[train], ps.b_idx[train]]))
    Xs = _standardize(np.asarray(X, dtype=np.float32), rows)
    D = torch.as_tensor(Xs[ps.a_idx] - Xs[ps.b_idx], device=device)
    y = torch.as_tensor(ps.y, device=device)
    tr = torch.as_tensor(np.flatnonzero(train), device=device)
    te = torch.as_tensor(np.flatnonzero(ps.test), device=device)
    w, lam, cv = fit_bt(D[tr], y[tr], ps.family[train], device)
    correct = ((D[te] @ w > 0).float() == y[te]).cpu().numpy().astype(bool)
    lo, hi = family_bootstrap(correct, ps.family[ps.test], seed=seed)
    return ProbeResult(heldout=float(correct.mean()), ci_lo=lo, ci_hi=hi,
                       train_acc=_acc(D[tr], y[tr], w), cv_acc=cv, lam=lam, n_test=int(len(te)))


def one_hot(looks: list[tuple[str, ...]], cat: Catalog) -> np.ndarray:
    offsets, col = {}, 0
    for slot in SLOTS:
        for value in cat.values[slot]:
            offsets[(slot, value)] = col
            col += 1
    X = np.zeros((len(looks), col), np.float32)
    for i, look in enumerate(looks):
        for slot, value in zip(SLOTS, look, strict=True):
            X[i, offsets[(slot, value)]] = 1.0
    return X


def per_slot_decodability(X: np.ndarray, looks, cat: Catalog, train_mask: np.ndarray,
                          device: str = "cpu") -> dict[str, dict]:
    Xn = _standardize(np.asarray(X, np.float32), np.flatnonzero(train_mask))
    Xs = torch.as_tensor(Xn, device=device)
    tr = torch.as_tensor(np.flatnonzero(train_mask), device=device)
    te = torch.as_tensor(np.flatnonzero(~train_mask), device=device)
    out = {}
    for s, slot in enumerate(SLOTS):
        labels_np = np.array([cat.values[slot].index(look[s]) for look in looks])
        labels = torch.as_tensor(labels_np, device=device)
        k = len(cat.values[slot])
        W = torch.zeros(Xs.shape[1], k, device=device, requires_grad=True)
        opt = torch.optim.LBFGS([W], max_iter=200, line_search_fn="strong_wolfe")

        def closure(opt=opt, W=W, labels=labels):
            opt.zero_grad()
            ce = torch.nn.functional.cross_entropy(Xs[tr] @ W, labels[tr])
            loss = ce + 1e-2 * (W**2).sum() / len(tr)
            loss.backward()
            return loss

        opt.step(closure)
        pred = (Xs[te] @ W).argmax(1)
        chance = float(np.bincount(labels_np[train_mask], minlength=k).max() / train_mask.sum())
        out[slot] = {
            "acc": float((pred == labels[te]).float().mean()),
            "chance": chance,
            "values": k,
        }
    return out


def readout_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.abs(np.asarray(A) - np.asarray(B)).sum(1).mean())


def smooth_ok(d_min: float, d_slot: float, d_full: float) -> bool:
    return d_min <= 0.1 * d_slot and d_slot < d_full


def activity_ok(max_step_frac: float, readout_active_frac: float) -> bool:
    return max_step_frac <= 0.10 and readout_active_frac >= 0.05
