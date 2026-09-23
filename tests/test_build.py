import numpy as np
import pandas as pd
import pytest

from lfg_fly.connectome import build as B


def _tables():
    ann = pd.DataFrame({
        "bodyId": [10, 20, 30, 40, 50],
        "status": ["Traced", "Traced", "Traced", "Traced", "Orphan"],
        "superclass": ["ol_sensory", "ol_intrinsic", "descending_neuron", "vnc_motor", "x"],
        "type": ["R1-R6", "L1", "DNa01", "MN1", "x"],
        "rootSide": ["R", None, None, None, None],
    })
    nt = pd.DataFrame({
        "body": [10, 20, 30],
        "consensus_nt": ["histamine", "acetylcholine", "gaba"],
    })
    edges = pd.DataFrame({
        "body_pre": [10, 20, 20, 30, 10, 50],
        "body_post": [20, 30, 40, 40, 30, 20],
        "weight": [5, 3, 7, 2, 4, 9],
    })
    return ann, nt, edges


def test_build_filters_thresholds_and_signs():
    ann, nt, edges = _tables()
    g = B.build_graph(ann, nt, edges, min_syn=3)
    assert g.n == 4 and list(g.body_id) == [10, 20, 30, 40]
    assert g.nt_missing == 1  # body 40 has no NT row -> unclear (+1)
    assert list(g.sign) == [-1, 1, -1, 1]
    # kept edges (>=3, both traced): 10->20 (5), 20->30 (3), 20->40 (7), 10->30 (4)
    dense = np.zeros((4, 4), dtype=np.float32)
    for post in range(4):
        for k in range(g.crow[post], g.crow[post + 1]):
            dense[post, g.col[k]] = g.ival[k]
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[1, 0] = -5
    expected[2, 1] = 3
    expected[3, 1] = 7
    expected[2, 0] = -4
    assert np.array_equal(dense, expected)
    assert g.insum.tolist() == [1.0, 5.0, 7.0, 7.0]  # unsigned synapse sums, 0 -> 1


def test_unknown_nt_fails_but_missing_is_unclear():
    ann, nt, edges = _tables()
    bad = pd.concat([nt, pd.DataFrame({"body": [40], "consensus_nt": ["tyramine"]})])
    with pytest.raises(ValueError, match="tyramine"):
        B.build_graph(ann, bad, edges)


def test_save_load_roundtrip_and_hash(tmp_path):
    ann, nt, edges = _tables()
    g = B.build_graph(ann, nt, edges)
    B.save_graph(g, tmp_path / "g.npz")
    h = B.load_graph(tmp_path / "g.npz")
    assert h.graph_hash() == g.graph_hash()
    assert list(h.cell_type) == list(g.cell_type)
    assert h.min_syn == 3 and h.nt_missing == 1


def test_with_sign_and_with_edges_keep_neurons():
    ann, nt, edges = _tables()
    g = B.build_graph(ann, nt, edges)
    flipped = B.with_sign(g, -g.sign)
    assert list(flipped.ival) == [-x for x in g.ival]
    rew = B.with_edges(g, np.array([0, 1]), np.array([1, 2]), np.array([5, 3]))
    assert rew.n == g.n and int(rew.crow[-1]) == 2
