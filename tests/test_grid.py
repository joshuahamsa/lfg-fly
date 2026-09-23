import json

import numpy as np
import pandas as pd

from lfg_fly.brain.senses import SLOTS, Retina
from lfg_fly.brain.sim import BrainParams
from lfg_fly.connectome import build as B
from lfg_fly.connectome.neurons import populations
from lfg_fly.teacher import grid as G
from lfg_fly.teacher import sample as T
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.render import ZOrder


def _toy_context(tmp_path, feedforward: bool):
    """ORNs -> readout. Feed-forward: a clean, learnable map. Otherwise: strong recurrence."""
    rng = np.random.default_rng(0)
    n_orn, n_mid, n_ro = 60, 80, 60
    n = n_orn + n_mid + n_ro
    sc = ["cb_sensory"] * n_orn + ["cb_intrinsic"] * n_mid + ["descending_neuron"] * n_ro
    types = [f"ORN_G{i % 12}" for i in range(n_orn)] + ["x"] * (n_mid + n_ro)
    pre, post = [], []
    for o in range(n_orn):
        for m in rng.choice(n_mid, 6, replace=False):
            pre.append(o)
            post.append(n_orn + m)
    for m in range(n_mid):
        for r in rng.choice(n_ro, 5, replace=False):
            pre.append(n_orn + m)
            post.append(n_orn + n_mid + r)
    if not feedforward:
        for _ in range(900):
            a, b = rng.integers(n_orn, n, 2)
            if a != b:
                pre.append(int(a))
                post.append(int(b))
    edges = pd.DataFrame({"pre": pre, "post": post}).drop_duplicates()
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": sc, "type": types, "rootSide": [None] * n})
    nt = pd.DataFrame({"body": np.arange(1, n + 1), "consensus_nt": ["acetylcholine"] * n})
    g = B.build_graph(ann, nt, pd.DataFrame({"body_pre": edges.pre + 1,
                                             "body_post": edges.post + 1,
                                             "weight": 5}), min_syn=1)
    pops = populations(g)
    retina = Retina(neurons=np.array([], np.int64), px=np.array([], np.int64),
                    py=np.array([], np.int64), channel=np.array([], np.int64))
    values = {s: [f"{s}{i}" for i in range(6)] + ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = ["B0", "B1"]
    cat = Catalog(body="male", values=values, odds={}, api="x")
    ps = T.make_probe_set(cat, n_pairs=900, seed=0)
    z = ZOrder(layer_z={s: float(i) for i, s in enumerate(SLOTS)}, overrides={},
               rank={s: i for i, s in enumerate(SLOTS)})
    return G.Context(graph=g, pops=pops, retina=retina, catalog=cat, bank=None, zorder=z,
                     device="cpu", probe_set=ps, seed=0)


def test_learnable_toy_passes_and_chaotic_toy_fails_smoothness(tmp_path):
    good = _toy_context(tmp_path, feedforward=True)
    # all-sensory codes give the toy ~60 identity channels
    # (its 12 glomeruli would bottleneck "nose")
    s_good = G.Setting(BrainParams(kind="rate", g_syn=1.0, bias=0.0, steps=8, burn_in=5),
                       "all-sensory-brain", 1.0)
    a = G.stage_a(good, [s_good], tmp_path / "a.jsonl")
    assert a[0]["smooth"]["ok"] is True
    b = G.stage_b(good, a, tmp_path / "b.jsonl", cap=4, probe_all=True)
    assert b[0]["probe"]["heldout"] > 0.6

    bad = _toy_context(tmp_path, feedforward=False)
    s_bad = G.Setting(BrainParams(kind="lif", g_syn=12.0, bias=0.3, steps=40, burn_in=20),
                      "nose", 3.0)
    a_bad = G.stage_a(bad, [s_bad], tmp_path / "a2.jsonl")
    assert a_bad[0]["passed"] is False


def test_stage_a_is_resumable(tmp_path):
    ctx = _toy_context(tmp_path, feedforward=True)
    s = G.Setting(BrainParams(kind="rate", g_syn=1.0, steps=8, burn_in=5), "nose", 1.0)
    out = tmp_path / "a.jsonl"
    G.stage_a(ctx, [s], out)
    first = out.read_text()
    G.stage_a(ctx, [s], out)
    assert out.read_text() == first  # the second run found the key and skipped it


def test_verdict_applies_gate():
    a = [{"key": "k1", "passed": True}]
    b = [{"key": "k1", "passed": True, "probe": {"heldout": 0.61, "ci_lo": 0.57, "ci_hi": 0.65}}]
    v = G.verdict(a, b, {"rewired": {"heldout": 0.55}}, {"heldout": 0.77}, 0.84)
    assert v["pass"] is True and v["best"]["key"] == "k1"
    b[0]["probe"]["heldout"] = 0.58
    assert G.verdict(a, b, {}, {"heldout": 0.77}, 0.84)["pass"] is False


def test_default_grid_covers_every_brain_and_code():
    grid = G.default_grid()
    assert {s.brain.kind for s in grid} == {"lif", "lif-avg", "lif-volley", "rate"}
    assert {s.code for s in grid} == {"nose", "eyes", "eyes+nose", "all-sensory-brain",
                                      "all-sensory+vnc"}
    assert len({s.key() for s in grid}) == len(grid)
    json.dumps([s.as_dict() for s in grid])
