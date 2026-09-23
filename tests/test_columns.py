import numpy as np
import pandas as pd
import pytest

from lfg_fly.connectome import build as B
from lfg_fly.connectome import columns as C
from lfg_fly.connectome import neurons as N


def _world():
    ann = pd.DataFrame({
        "bodyId": [1, 2, 3, 11, 12, 13],
        "status": ["Traced"] * 6,
        "superclass": ["ol_sensory"] * 3 + ["ol_intrinsic"] * 3,
        "type": ["R1-R6", "R8y", "R7p", "L1", "L2", "L3"],
        "rootSide": ["L", "R", "R", None, None, None],
        "somaSide": [None, None, None, "L", "R", "R"],
        "assignedOlHex1": [np.nan, np.nan, np.nan, 3.0, 5.0, 6.0],
        "assignedOlHex2": [np.nan, np.nan, np.nan, 4.0, 1.0, 1.0],
    })
    nt = pd.DataFrame({"body": [1, 2, 3], "consensus_nt": ["histamine"] * 3})
    # PR1 -> L1 (col L 3,4) x9, -> L2 (col R 5,1) x1
    # PR2 -> L2 x4, L3 x4 (tie -> sort order) ; PR3 -> none
    edges_any = pd.DataFrame({
        "body_pre": [1, 1, 2, 2],
        "body_post": [11, 12, 12, 13],
        "weight": [9, 1, 4, 4],
    })
    g = B.build_graph(ann, nt, edges_any, min_syn=1)
    return g, N.populations(g), ann, edges_any


def test_assigns_majority_column_and_drops_orphans(monkeypatch):
    g, pops, ann, edges = _world()
    monkeypatch.setattr(C, "MIN_ASSIGNED", 0.5)
    cols = C.assign_columns(g, pops, ann, edges)
    assert cols.total == 3 and cols.dropped == 1
    assert cols.neuron.tolist() == [0, 1]
    assert cols.eye.tolist() == ["L", "R"]
    assert (cols.q.tolist(), cols.r.tolist()) == ([3, 5], [4, 1])  # PR2 tie: (R,5,1) sorts first
    assert cols.share[0] == pytest.approx(0.9)


def test_too_few_assigned_fails():
    g, pops, ann, edges = _world()
    with pytest.raises(ValueError, match="assigned"):
        C.assign_columns(g, pops, ann, edges)  # 2/3 < 0.90


def test_lattice_is_normalized_per_eye(monkeypatch):
    cols = C.Columns(neuron=np.array([0, 1, 2, 3]), eye=np.array(["L", "L", "R", "R"]),
                     q=np.array([0, 2, 10, 10]), r=np.array([0, 0, 0, 4]),
                     share=np.ones(4, np.float32), dropped=0, total=4)
    x, y = C.lattice_xy(cols)
    assert x.tolist()[:2] == [0.0, 1.0]
    assert y.tolist()[2:] == [0.0, 1.0]
    assert x.min() >= 0 and x.max() <= 1 and y.min() >= 0 and y.max() <= 1


def test_roundtrip(tmp_path, monkeypatch):
    g, pops, ann, edges = _world()
    monkeypatch.setattr(C, "MIN_ASSIGNED", 0.5)
    cols = C.assign_columns(g, pops, ann, edges)
    C.save_columns(cols, tmp_path / "c.npz")
    back = C.load_columns(tmp_path / "c.npz")
    assert back.neuron.tolist() == cols.neuron.tolist() and back.dropped == 1
