"""Controls (spec §3.4): a degree-preserving rewire and a sign shuffle.

Permuting the post column keeps every neuron's in- and out-degree. The self-loops
and duplicates it creates are then repaired by swapping post targets with random
other edges, accepting a swap only if it creates no new self-loop or duplicate.
Swaps also preserve degrees, so the result keeps both degree sequences exactly.
"""

from __future__ import annotations

import numpy as np

from lfg_fly.connectome.build import Graph, edge_list, with_edges, with_sign


def _bad(pre: np.ndarray, post: np.ndarray, n: int) -> np.ndarray:
    keys = pre * n + post
    order = np.argsort(keys, kind="stable")
    dup = np.zeros(len(keys), bool)
    dup[order[1:]] = keys[order][1:] == keys[order][:-1]
    return dup | (pre == post)


def rewire_edges(pre: np.ndarray, post: np.ndarray, n: int, seed: int = 0,
                 max_rounds: int = 200) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)[rng.permutation(len(post))]
    repairs = 0
    for _ in range(max_rounds):
        bad = _bad(pre, post, n)
        if not bad.any():
            return post, repairs
        bi = np.flatnonzero(bad)
        cand = rng.integers(0, len(post), size=len(bi))
        keys = np.sort(pre * n + post)
        new_b = pre[bi] * n + post[cand]
        new_c = pre[cand] * n + post[bi]
        ok = (
            (pre[bi] != post[cand]) & (pre[cand] != post[bi]) & ~bad[cand]
            & ~_in_sorted(keys, new_b) & ~_in_sorted(keys, new_c) & (new_b != new_c)
        )
        # one swap per edge per round; no two accepted swaps may create the same key
        _, first = np.unique(cand, return_index=True)
        keep = np.zeros(len(bi), bool)
        keep[first] = True
        ok &= keep & ~np.isin(cand, bi)
        created = np.concatenate([new_b[ok], new_c[ok]])
        uniq, counts = np.unique(created, return_counts=True)
        clash = np.isin(new_b, uniq[counts > 1]) | np.isin(new_c, uniq[counts > 1])
        ok &= ~clash
        post[bi[ok]], post[cand[ok]] = post[cand[ok]], post[bi[ok]].copy()
        repairs += int(ok.sum())
    raise RuntimeError(f"rewire did not converge in {max_rounds} rounds")


def _in_sorted(sorted_keys: np.ndarray, x: np.ndarray) -> np.ndarray:
    i = np.searchsorted(sorted_keys, x)
    i = np.clip(i, 0, len(sorted_keys) - 1)
    return sorted_keys[i] == x


def rewired_graph(g: Graph, seed: int = 0, device: str = "cpu") -> tuple[Graph, int]:
    pre, post, count = edge_list(g)
    new_post, repairs = rewire_edges(pre, post, g.n, seed=seed)
    return with_edges(g, pre, new_post, count, device), repairs


def sign_shuffled_graph(g: Graph, seed: int = 0) -> Graph:
    return with_sign(g, np.random.default_rng(seed).permutation(g.sign))
