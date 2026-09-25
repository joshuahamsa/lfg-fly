"""Score a feature map against the planted taste (spec §3.0, §3.4).

Bradley-Terry logistic regression on feature differences, L2 chosen by 5-fold
cross-validation grouped by look family, a held-out test split by family, and a
family-resampling bootstrap CI. `evaluate_mlp` is the same protocol with a one-
hidden-layer scorer (§3.4 control 3), and `paired_diff` compares two models on
the same test pairs (§3.0b).

`fit_bt`, `evaluate` and `evaluate_mlp` take an optional per-pair weight `w`
(spec §2 Readout: the critic's strength as the sample weight). It weights the
BCE term, normalized to mean 1 over the rows being fitted, so a flat weight
changes nothing; `None` means flat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.sample import ProbeSet

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
MLP_LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
MLP_HIDDEN = 256


@dataclass
class ProbeResult:
    heldout: float
    ci_lo: float
    ci_hi: float
    train_acc: float
    cv_acc: float
    lam: float
    n_test: int
    # per test pair, in test order: kept for paired comparisons, never in the record
    correct: np.ndarray | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict:
        out = asdict(self)
        out.pop("correct")
        return out


def _folds(groups: np.ndarray, k: int) -> np.ndarray:
    uniq = np.unique(groups)
    order = np.random.default_rng(12345).permutation(len(uniq))
    fold_of = {g: int(i % k) for i, g in zip(order, uniq, strict=True)}
    return np.array([fold_of[g] for g in groups])


def _sample_weights(w, device: str, n: int) -> torch.Tensor | None:
    """Per-pair weights as a float tensor on `device`, or None for flat."""
    if w is None:
        return None
    if isinstance(w, torch.Tensor):
        sw = w.detach().to(device=device, dtype=torch.float32)
    else:
        sw = torch.as_tensor(np.asarray(w, dtype=np.float32), device=device)
    if sw.shape != (n,):
        raise ValueError(f"w has shape {tuple(sw.shape)}, expected ({n},)")
    if bool((sw < 0).any()) or not bool((sw > 0).any()):
        raise ValueError("w must be non-negative with a positive sum")
    return sw


def _normalized(sw: torch.Tensor | None) -> torch.Tensor | None:
    """Weights normalized to mean 1 over the rows being fitted, so the BCE keeps its scale."""
    return None if sw is None else sw / sw.mean()


def _fit(D: torch.Tensor, y: torch.Tensor, lam: float,
         sw: torch.Tensor | None = None) -> torch.Tensor:
    w = torch.zeros(D.shape[1], device=D.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")
    sw = _normalized(sw)

    def closure():
        opt.zero_grad()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(D @ w, y, weight=sw)
        loss = bce + lam * (w**2).sum() / len(y)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def _acc(D: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> float:
    return float(((D @ w > 0).float() == y).float().mean())


def fit_bt(D: torch.Tensor, y: torch.Tensor, groups: np.ndarray, device: str,
           lambdas: tuple[float, ...] = LAMBDAS,
           folds: int = 5, w=None) -> tuple[torch.Tensor, float, float]:
    """Bradley-Terry weights on pair differences `D`, with L2 chosen by `folds`-fold CV grouped
    by `groups`. `w`: optional per-row sample weights (weighted BCE; CV accuracy stays
    unweighted). Returns (weights, lambda, cv accuracy)."""
    sw = _sample_weights(w, device, len(y))
    fold = _folds(groups, folds)
    best = (-1.0, lambdas[-1])
    for lam in lambdas:
        accs = []
        for k in range(folds):
            tr = torch.as_tensor(np.flatnonzero(fold != k), device=device)
            va = torch.as_tensor(np.flatnonzero(fold == k), device=device)
            extra = {} if sw is None else {"sw": sw[tr]}
            accs.append(_acc(D[va], y[va], _fit(D[tr], y[tr], lam, **extra)))
        score = float(np.mean(accs))
        if score > best[0] or (score == best[0] and lam > best[1]):
            best = (score, lam)
    extra = {} if sw is None else {"sw": sw}
    return _fit(D, y, best[1], **extra), best[1], best[0]


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


def evaluate(X: np.ndarray, ps: ProbeSet, device: str = "cpu", seed: int = 0,
             w: np.ndarray | None = None) -> ProbeResult:
    """`w`: optional per-pair weights over every pair of `ps` (only the training pairs' are
    used); None = flat."""
    train = ~ps.test
    rows = np.unique(np.concatenate([ps.a_idx[train], ps.b_idx[train]]))
    Xs = _standardize(np.asarray(X, dtype=np.float32), rows)
    D = torch.as_tensor(Xs[ps.a_idx] - Xs[ps.b_idx], device=device)
    y = torch.as_tensor(ps.y, device=device)
    tr = torch.as_tensor(np.flatnonzero(train), device=device)
    te = torch.as_tensor(np.flatnonzero(ps.test), device=device)
    extra = {} if w is None else {"w": np.asarray(w)[train]}
    wt, lam, cv = fit_bt(D[tr], y[tr], ps.family[train], device, **extra)
    correct = ((D[te] @ wt > 0).float() == y[te]).cpu().numpy().astype(bool)
    lo, hi = family_bootstrap(correct, ps.family[ps.test], seed=seed)
    return ProbeResult(heldout=float(correct.mean()), ci_lo=lo, ci_hi=hi,
                       train_acc=_acc(D[tr], y[tr], wt), cv_acc=cv, lam=lam, n_test=int(len(te)),
                       correct=correct)


def _mlp_scores(Xs: torch.Tensor, params: list[torch.Tensor]) -> torch.Tensor:
    W1, b1, w2 = params
    return torch.relu(Xs @ W1 + b1) @ w2


def _fit_mlp(Xs: torch.Tensor, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor, lam: float,
             hidden: int, seed: int, sw: torch.Tensor | None = None) -> list[torch.Tensor]:
    """A scalar scorer s(look); P(A > B) = sigmoid(s(A) - s(B)). Full-batch LBFGS, L2 on
    both weight matrices, initialised on the CPU from `seed` so every device starts alike.
    `sw`: optional per-pair sample weights (normalized to mean 1 here)."""
    g = torch.Generator().manual_seed(seed)
    d = Xs.shape[1]
    params = [(torch.randn(d, hidden, generator=g) / d**0.5).to(Xs.device),
              torch.zeros(hidden).to(Xs.device),
              (torch.randn(hidden, generator=g) / hidden**0.5).to(Xs.device)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.LBFGS(params, max_iter=300, line_search_fn="strong_wolfe")
    sw = _normalized(sw)

    def closure():
        opt.zero_grad()
        s = _mlp_scores(Xs, params)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(s[a] - s[b], y, weight=sw)
        loss = bce + lam * ((params[0] ** 2).sum() + (params[2] ** 2).sum()) / len(y)
        loss.backward()
        return loss

    opt.step(closure)
    return [p.detach() for p in params]


def _mlp_correct(Xs, params, a, b, y) -> torch.Tensor:
    s = _mlp_scores(Xs, params)
    return ((s[a] - s[b] > 0).float() == y)


def evaluate_mlp(X: np.ndarray, ps: ProbeSet, device: str = "cpu", seed: int = 0,
                 hidden: int = MLP_HIDDEN, lambdas: tuple[float, ...] = MLP_LAMBDAS,
                 folds: int = 5, w: np.ndarray | None = None) -> ProbeResult:
    """`evaluate`'s protocol (standardize on training looks, L2 by family-grouped CV,
    family-bootstrap CI) with a one-hidden-layer scorer in place of the linear one.
    `w`: optional per-pair weights over every pair of `ps`; None = flat."""
    train = ~ps.test
    rows = np.unique(np.concatenate([ps.a_idx[train], ps.b_idx[train]]))
    Xs = torch.as_tensor(_standardize(np.asarray(X, dtype=np.float32), rows), device=device)
    a = torch.as_tensor(ps.a_idx, device=device)
    b = torch.as_tensor(ps.b_idx, device=device)
    y = torch.as_tensor(ps.y, device=device)
    sw = _sample_weights(w, device, len(ps.y))
    tr_idx = np.flatnonzero(train)
    fold = _folds(ps.family[train], folds)
    best = (-1.0, lambdas[-1])
    for lam in lambdas:
        accs = []
        for k in range(folds):
            fit = torch.as_tensor(tr_idx[fold != k], device=device)
            val = torch.as_tensor(tr_idx[fold == k], device=device)
            extra = {} if sw is None else {"sw": sw[fit]}
            params = _fit_mlp(Xs, a[fit], b[fit], y[fit], lam, hidden, seed, **extra)
            accs.append(float(_mlp_correct(Xs, params, a[val], b[val], y[val]).float().mean()))
        score = float(np.mean(accs))
        if score > best[0] or (score == best[0] and lam > best[1]):
            best = (score, lam)
    tr = torch.as_tensor(tr_idx, device=device)
    te = torch.as_tensor(np.flatnonzero(ps.test), device=device)
    extra = {} if sw is None else {"sw": sw[tr]}
    params = _fit_mlp(Xs, a[tr], b[tr], y[tr], best[1], hidden, seed, **extra)
    correct = _mlp_correct(Xs, params, a[te], b[te], y[te]).cpu().numpy().astype(bool)
    lo, hi = family_bootstrap(correct, ps.family[ps.test], seed=seed)
    train_acc = float(_mlp_correct(Xs, params, a[tr], b[tr], y[tr]).float().mean())
    return ProbeResult(heldout=float(correct.mean()), ci_lo=lo, ci_hi=hi, train_acc=train_acc,
                       cv_acc=best[0], lam=best[1], n_test=int(len(te)), correct=correct)


def paired_diff(correct_a: np.ndarray, correct_b: np.ndarray, family: np.ndarray,
                n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Accuracy of model a minus model b on the same test pairs, with a 95% CI from a
    bootstrap that resamples look families and scores both models on each resample."""
    fams, inv = np.unique(family, return_inverse=True)
    d = np.bincount(inv, weights=correct_a.astype(float) - correct_b.astype(float),
                    minlength=len(fams))
    n = np.bincount(inv, minlength=len(fams)).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(fams), (n_boot, len(fams)))
    stats = d[pick].sum(1) / n[pick].sum(1)
    return {"diff": float(correct_a.mean() - correct_b.mean()),
            "lo": float(np.percentile(stats, 2.5)), "hi": float(np.percentile(stats, 97.5))}


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
