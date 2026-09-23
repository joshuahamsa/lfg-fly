import numpy as np
import pandas as pd

from lfg_fly.connectome import build as B
from lfg_fly.teacher import rewire as W


def _dense_graph(n=30, seed=0):
    # ~260 of 870 possible edges: dense enough that the raw permutation collides,
    # sparse enough that repair swaps exist
    rng = np.random.default_rng(seed)
    pairs = {(int(a), int(b)) for a, b in rng.integers(0, n, (300, 2)) if a != b}
    pre, post = map(np.array, zip(*sorted(pairs), strict=True))
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": ["x"] * n, "type": ["t"] * n, "rootSide": [None] * n})
    nts = ["gaba" if i % 3 == 0 else "acetylcholine" for i in range(n)]
    nt = pd.DataFrame({"body": np.arange(1, n + 1), "consensus_nt": nts})
    weight = rng.integers(1, 9, len(pre))
    edges = pd.DataFrame({"body_pre": pre + 1, "body_post": post + 1, "weight": weight})
    return B.build_graph(ann, nt, edges, min_syn=1)


def test_rewire_preserves_degrees_with_no_loops_or_duplicates():
    g = _dense_graph()
    pre, post, count = B.edge_list(g)
    new_post, repairs = W.rewire_edges(pre, post, g.n, seed=1)
    assert repairs > 0
    # pre is untouched (out-degrees kept by construction); in-degrees must match exactly
    assert np.array_equal(np.bincount(new_post, minlength=g.n), np.bincount(post, minlength=g.n))
    assert not np.any(pre == new_post)
    keys = pre * g.n + new_post
    assert len(np.unique(keys)) == len(keys)
    assert not np.array_equal(new_post, post)


def test_rewired_graph_keeps_counts_and_signs():
    g = _dense_graph()
    r, repairs = W.rewired_graph(g, seed=2)
    assert r.n == g.n and len(r.col) == len(g.col)
    assert sorted(r.count.tolist()) == sorted(g.count.tolist())
    assert np.array_equal(r.sign, g.sign)
    assert r.graph_hash() != g.graph_hash()


def test_sign_shuffle_permutes_labels():
    g = _dense_graph()
    s = W.sign_shuffled_graph(g, seed=3)
    assert sorted(s.sign.tolist()) == sorted(g.sign.tolist())
    assert not np.array_equal(s.sign, g.sign)
