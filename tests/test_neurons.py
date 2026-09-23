import pandas as pd

from lfg_fly.connectome import build as B
from lfg_fly.connectome import neurons as N


def test_populations_select_by_type_and_superclass():
    ann = pd.DataFrame({
        "bodyId": [1, 2, 3, 4, 5, 6, 7],
        "status": ["Traced"] * 7,
        "superclass": ["ol_sensory", "ol_sensory", "cb_sensory", "vnc_sensory",
                       "descending_neuron", "cb_motor", "cb_intrinsic"],
        "type": ["R1-R6", "R8y", "ORN_DA1", "SNxx", "DNa01", "MNx", "KC"],
        "rootSide": ["L", "R", None, None, None, None, None],
    })
    nt = pd.DataFrame({
        "body": [1, 2, 3],
        "consensus_nt": ["histamine", "histamine", "acetylcholine"],
    })
    edges = pd.DataFrame({"body_pre": [3], "body_post": [5], "weight": [3]})
    g = B.build_graph(ann, nt, edges)
    p = N.populations(g)
    assert p.photoreceptor.tolist() == [0, 1]
    assert p.photoreceptor_type.tolist() == ["R1-R6", "R8y"]
    assert p.orn.tolist() == [2] and p.orn_glomerulus.tolist() == ["DA1"]
    assert p.readout.tolist() == [4, 5]
    assert p.sensory_brain.tolist() == [0, 1, 2]
    assert p.sensory_vnc.tolist() == [3]
    assert p.sensory_any.tolist() == [True, True, True, True, False, False, False]
    assert N.CHANNEL_BY_TYPE["R8y"] == "G" and N.CHANNEL_BY_TYPE["R1-R6"] == "lum"
