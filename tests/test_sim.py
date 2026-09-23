import numpy as np
import pandas as pd
import pytest
import torch

from lfg_fly.brain import sim as S
from lfg_fly.connectome import build as B


def _graph(pre, post, weight, superclass, nts):
    n = len(superclass)
    ann = pd.DataFrame({"bodyId": np.arange(1, n + 1), "status": ["Traced"] * n,
                        "superclass": superclass, "type": ["t"] * n, "rootSide": [None] * n})
    nt = pd.DataFrame({"body": np.arange(1, n + 1), "consensus_nt": nts})
    edges = pd.DataFrame({"body_pre": np.asarray(pre) + 1, "body_post": np.asarray(post) + 1,
                          "weight": weight})
    return B.build_graph(ann, nt, edges, min_syn=1)


def _random_graph(n=40, m=300, seed=0):
    rng = np.random.default_rng(seed)
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = pre != post
    pairs = pd.DataFrame({"pre": pre[keep], "post": post[keep]}).drop_duplicates()
    sc = ["cb_sensory"] * 5 + ["cb_intrinsic"] * (n - 10) + ["descending_neuron"] * 5
    nts = ["acetylcholine" if i % 4 else "gaba" for i in range(n)]
    return _graph(pairs.pre.to_numpy(), pairs.post.to_numpy(),
                  rng.integers(1, 9, len(pairs)), sc, nts)


def _masks(g):
    sensory = np.char.find(g.superclass, "sensory") >= 0
    readout = np.flatnonzero(g.superclass == "descending_neuron")
    return sensory, readout


@pytest.mark.parametrize("kind,steps", [("lif", 30), ("lif-volley", 6)])
def test_lif_matches_numpy_reference_exactly(kind, steps):
    g = _random_graph()
    sensory, readout = _masks(g)
    p = S.BrainParams(kind=kind, g_syn=3.0, bias=0.06, steps=steps, burn_in=20)
    rng = np.random.default_rng(1)
    drive = np.zeros((g.n, 7), np.float32)
    drive[:5] = rng.uniform(0, 1.5, (5, 7))
    rest = np.zeros((g.n, 1), np.float32)
    ref = S.simulate_reference(g, sensory, readout, drive, p, rest)
    sim = S.Simulator(g, sensory, readout, device="cpu")
    got = sim.run(torch.as_tensor(drive), p, torch.as_tensor(rest), "zero").features.numpy()
    assert np.array_equal(got, ref)


def test_rate_matches_numpy_reference_to_tolerance():
    g = _random_graph()
    sensory, readout = _masks(g)
    p = S.BrainParams(kind="rate", g_syn=1.0, bias=0.1, steps=30, burn_in=20)
    drive = np.random.default_rng(2).uniform(0, 1, (g.n, 4)).astype(np.float32)
    drive[5:] = 0
    rest = np.zeros((g.n, 1), np.float32)
    ref = S.simulate_reference(g, sensory, readout, drive, p, rest)
    got = S.Simulator(g, sensory, readout, "cpu").run(
        torch.as_tensor(drive), p, torch.as_tensor(rest), "zero").features.numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)


def test_runs_are_deterministic():
    g = _random_graph()
    sensory, readout = _masks(g)
    sim = S.Simulator(g, sensory, readout, "cpu")
    p = S.BrainParams(kind="lif-avg", g_syn=3.0, bias=0.05, trials=4, noise=0.05, seed=9)
    d = torch.rand(g.n, 3)
    rest = torch.zeros(g.n, 1)
    a = sim.run(d, p, rest, "z").features
    b = sim.run(d, p, rest, "z").features
    assert torch.equal(a, b)


def test_refractory_limits_rate():
    # one self-driven neuron with huge input fires every (refractory+1) steps
    g = _graph([0], [1], [1], ["cb_sensory", "descending_neuron"], ["acetylcholine"] * 2)
    sensory, readout = _masks(g)
    p = S.BrainParams(kind="lif", g_syn=0.0, steps=30, burn_in=0, refractory=2)
    drive = torch.zeros(2, 1)
    drive[1] = 5.0  # the readout neuron is driven directly
    res = S.Simulator(g, sensory, readout, "cpu").run(drive, p, torch.zeros(2, 1), "z")
    assert res.features.item() == 10  # 30 steps / (2 refractory + 1)


def test_tonic_bias_lets_inhibitory_eyes_reach_downstream():
    # photoreceptor P (sensory, histamine) -| L (intrinsic) -> D (readout)
    #
    # NOTE (deviation from the brief): the brief used rest_drive = zeros(3, 1)
    # (i.e. rest == dark, both 0). With L and D sharing the same global bias
    # and decay and no other input, that makes the burn-in synchronize L and D
    # to an identical phase-locked oscillation: D always crosses threshold from
    # its OWN accumulated bias at the exact same step L does, so any kick L
    # delivers is always redundant (it lands exactly on D's own threshold
    # crossing, or is discarded by D's own refractory immediately after D fires
    # on its own). This holds for every (g_syn, bias, weight, light) combination
    # tested here -- it is a structural property of a 1-hop symmetric chain
    # under a uniform tonic bias, not a parameter-tuning accident. Verified by
    # direct simulation trace inspection (see task-5-report.md).
    #
    # Giving the resting state a nonzero "mid-grey" baseline on P (matching the
    # spec's own description of the cached rest state: "a 200-step burn-in with
    # a mid-grey retina", i.e. NOT total darkness) breaks that lock: P's
    # baseline firing suppresses L during burn-in but never touches D directly,
    # so L and D leave burn-in out of phase. "dark" (drive=0, i.e. darker than
    # the mid-grey rest baseline) then releases L from that suppression and its
    # spikes land at genuinely different points in D's cycle, changing D's
    # count; "light" (brighter than mid-grey) suppresses L further, same as
    # the resting state. This is what the test's docstring already intends: a
    # baseline the tonic bias then lets light modulate.
    g = _graph([0, 1], [1, 2], [5, 5], ["ol_sensory", "ol_intrinsic", "descending_neuron"],
               ["histamine", "acetylcholine", "acetylcholine"])
    sensory, readout = _masks(g)
    sim = S.Simulator(g, sensory, readout, "cpu")
    rest = torch.zeros(3, 1)
    rest[0] = 1.0  # mid-grey baseline on P, not total darkness
    dark, light = torch.zeros(3, 1), torch.zeros(3, 1)
    light[0] = 2.0
    for bias, should_differ in ((0.0, False), (0.2, True)):
        p = S.BrainParams(kind="lif", g_syn=2.0, bias=bias, steps=40, burn_in=50)
        a = sim.run(dark, p, rest, f"b{bias}").features
        b = sim.run(light, p, rest, f"b{bias}").features
        assert (not torch.equal(a, b)) == should_differ


def test_stats_are_fractions():
    g = _random_graph()
    sensory, readout = _masks(g)
    res = S.Simulator(g, sensory, readout, "cpu").run(
        torch.rand(g.n, 5), S.BrainParams(kind="lif", g_syn=3.0, bias=0.05),
        torch.zeros(g.n, 1), "z")
    assert res.active_frac.shape == (5,) and float(res.active_frac.max()) <= 1.0
    assert 0.0 <= res.max_step_frac <= 1.0
    assert res.readout_spikes.shape == (5,)


def test_params_validate():
    with pytest.raises(ValueError):
        S.BrainParams(kind="banana", g_syn=1.0)
    with pytest.raises(ValueError):
        S.BrainParams(kind="lif-avg", g_syn=1.0, trials=1, noise=0.05)
    with pytest.raises(ValueError):
        S.BrainParams(kind="lif", g_syn=1.0, trials=3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_matches_reference_exactly():
    g = _random_graph(seed=5)
    sensory, readout = _masks(g)
    p = S.BrainParams(kind="lif", g_syn=3.0, bias=0.06, steps=40, burn_in=20)
    drive = np.random.default_rng(3).uniform(0, 1.5, (g.n, 9)).astype(np.float32)
    rest = np.zeros((g.n, 1), np.float32)
    ref = S.simulate_reference(g, sensory, readout, drive, p, rest)
    got = S.Simulator(g, sensory, readout, "cuda").run(
        torch.as_tensor(drive, device="cuda"), p, torch.as_tensor(rest, device="cuda"), "z"
    ).features.cpu().numpy()
    assert np.array_equal(got, ref)
