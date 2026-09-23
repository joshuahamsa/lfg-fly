import io

import numpy as np
import torch
from PIL import Image

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import catalog as K
from lfg_fly.teacher import render as R

CFG = {
    "layers": [{"name": s, "z": 10 * (i + 1)} for i, s in enumerate(SLOTS)],
    "z_overrides": [{"trait_type": "Eyes", "value": "Laser", "z": 95}],
}


def _png(rgba):
    buf = io.BytesIO()
    Image.new("RGBA", (16, 16), rgba).save(buf, format="PNG")
    return buf.getvalue()


def _catalog(tmp_path, layers):
    values = {s: ["None"] for s in SLOTS if s != "Body"}
    values["Body"] = []
    for (slot, value), rgba in layers.items():
        path = K.layer_cache_path(tmp_path, "male", slot, value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png(rgba))
        values.setdefault(slot, [])
        values[slot] = [value] + [v for v in values[slot] if v != value]
    return K.Catalog(body="male", values=values, odds={}, api="http://lfg")


def test_zorder_uses_layers_and_overrides():
    z = R.zorder_from_config(CFG)
    assert z.key("Background", "x") < z.key("Head", "x")
    assert z.key("Eyes", "Laser") > z.key("Accessory", "x")  # 95 > 90


def test_composite_alpha_over_in_z_order(tmp_path):
    cat = _catalog(tmp_path, {("Background", "Red"): (255, 0, 0, 255),
                              ("Head", "HalfBlue"): (0, 0, 255, 128)})
    bank = R.LayerBank(cat, tmp_path, size=4)
    look = tuple({"Background": "Red", "Head": "HalfBlue"}.get(s, "None") for s in SLOTS)
    img = R.composite([look], bank, R.zorder_from_config(CFG))
    a = 128 / 255
    expected = torch.tensor([1 - a, 0.0, a])
    assert img.shape == (1, 3, 4, 4)
    assert torch.allclose(img[0, :, 0, 0], expected, atol=0.01)


def test_z_override_changes_stacking(tmp_path):
    cat = _catalog(tmp_path, {("Eyes", "Laser"): (0, 255, 0, 255),
                              ("Accessory", "Red"): (255, 0, 0, 255)})
    bank = R.LayerBank(cat, tmp_path, size=4)
    look = tuple({"Eyes": "Laser", "Accessory": "Red"}.get(s, "None") for s in SLOTS)
    img = R.composite([look], bank, R.zorder_from_config(CFG))
    assert np.allclose(img[0, :, 0, 0].numpy(), [0, 1, 0])  # Laser (95) sits above Accessory (90)


def test_load_zorder_caches_pinned_file(tmp_path):
    import yaml

    calls = []

    def get(url, timeout=30):
        calls.append(url)
        return 200, yaml.safe_dump(CFG).encode()

    R.load_zorder(tmp_path, commit="abc123", get=get)
    R.load_zorder(tmp_path, commit="abc123", get=get)
    assert len(calls) == 1 and "/abc123/trait_config.yaml" in calls[0]
