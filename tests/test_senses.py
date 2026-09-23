import hashlib

import numpy as np
import pytest
import torch

from lfg_fly.brain import senses as Z
from lfg_fly.connectome.columns import Columns
from lfg_fly.connectome.neurons import Populations

ORN = np.arange(100, 160, dtype=np.int64)
GLOM = np.array([f"G{i % 6}" for i in range(60)])


def test_odor_code_golden():
    code = Z.odor_code("Head", "Pirate Hat", ORN, GLOM)
    assert code.neurons.tolist() == [
        103, 109, 115, 121, 127, 133, 139, 145, 151, 157,
        100, 106, 112, 118, 124, 130, 136, 142, 148, 154,
        104, 110, 116, 122, 128, 134, 140, 146, 152, 158,
        102, 108, 114, 120, 126, 132, 138, 144, 150, 156,
    ]
    assert hashlib.sha256(code.weights.tobytes()).hexdigest()[:16] == "53991c51ab6963c6"
    assert code.weights[0] == pytest.approx(0.271866, abs=1e-6)


def test_pool_code_golden():
    code = Z.pool_code("Head", "Pirate Hat", np.arange(1000, 1200, dtype=np.int64))
    assert code.neurons.tolist() == [1194, 1085, 1157, 1168, 1079, 1111, 1070,
                                     1011, 1043, 1072, 1092, 1037, 1029, 1059]
    assert hashlib.sha256(code.weights.tobytes()).hexdigest()[:16] == "50d128800a761497"


def test_codes_separate_values_and_slots():
    a = Z.odor_code("Head", "Crown", ORN, GLOM)
    b = Z.odor_code("Head", "Pirate Hat", ORN, GLOM)
    c = Z.odor_code("Eyes", "Crown", ORN, GLOM)
    same_neurons = set(a.neurons.tolist()) == set(b.neurons.tolist())
    assert not same_neurons or not np.allclose(a.weights, b.weights)
    assert Z.trait_seed("odor/v2", "Head", "Crown") != Z.trait_seed("odor/v2", "Eyes", "Crown")
    assert c.neurons.dtype == np.int64


def _pops(n=20):
    return Populations(
        readout=np.array([18, 19]), photoreceptor=np.array([0, 1, 2]),
        photoreceptor_type=np.array(["R1-R6", "R8y", "R7p"]),
        orn=np.arange(3, 9), orn_glomerulus=np.array(["A", "A", "B", "B", "C", "D"]),
        sensory_brain=np.arange(0, 12), sensory_vnc=np.arange(12, 15),
        sensory_any=np.array([True] * 15 + [False] * (n - 15)),
    )


def _retina():
    cols = Columns(neuron=np.array([0, 1, 2]), eye=np.array(["L", "L", "L"]),
                   q=np.array([0, 4, 0]), r=np.array([0, 0, 4]), share=np.ones(3, np.float32),
                   dropped=0, total=3)

    class G:
        cell_type = np.array(["R1-R6", "R8y", "R7p"] + ["x"] * 17)

    return Z.build_retina(cols, G(), size=8)


def test_retina_samples_channels():
    retina = _retina()
    img = torch.zeros(1, 3, 8, 8)
    img[0, 1] = 1.0  # pure green image
    vals = Z.retina_values(img, retina)
    assert vals.shape == (3, 1)
    assert vals[0, 0] == pytest.approx(0.587, abs=1e-5)  # luminance of pure green
    assert vals[1, 0] == pytest.approx(1.0)               # R8y -> G
    assert vals[2, 0] == pytest.approx(0.0)               # R7p -> B


def test_builder_drive_components():
    pops, retina = _pops(), _retina()
    look = ("bg", "None", "b1", "c1", "m1", "e1", "y1", "h1", "a1")
    nose = Z.InputBuilder("nose", 20, pops, retina, g_in=2.0, device="cpu")
    d = nose.drive([look])
    assert d.shape == (20, 1)
    assert torch.allclose(d[[0, 1, 2], 0], torch.full((3,), 0.5))  # grey retina baseline (g_eye=1)
    assert float(d[3:9].sum()) > 0 and float(d[9:].sum()) == 0
    eyes = Z.InputBuilder("eyes", 20, pops, retina, g_in=2.0, device="cpu")
    img = torch.ones(1, 3, 8, 8)
    de = eyes.drive([look], images=img)
    assert float(de[3:9].sum()) == 0 and torch.allclose(de[[0, 1, 2], 0], torch.full((3,), 2.0))
    assert eyes.rest_drive()[0, 0] == pytest.approx(1.0)  # grey 0.5 * g_eye 2.0
    both = Z.InputBuilder("eyes+nose", 20, pops, retina, g_in=2.0, device="cpu")
    assert both.components == ("eyes", "nose")
    assert both.only("nose").name == "nose" and both.only("eyes").name == "eyes"


def test_concentration_scales_identity_drive():
    pops, retina = _pops(), _retina()
    b = Z.InputBuilder("all-sensory+vnc", 20, pops, retina, g_in=1.0, device="cpu")
    look = ("bg", "None", "b1", "c1", "m1", "e1", "y1", "h1", "a1")
    lo = b.drive([look], conc=np.full((1, 9), 0.4))
    hi = b.drive([look], conc=np.full((1, 9), 0.8))
    ident = slice(3, 20)
    assert torch.allclose(hi[ident] - 0.0, 2 * (lo[ident] - 0.0))
    with pytest.raises(ValueError):
        Z.InputBuilder("smell-o-vision", 20, pops, retina, g_in=1.0, device="cpu")
