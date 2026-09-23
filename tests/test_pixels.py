import io

import numpy as np
import pytest
import torch
from PIL import Image

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import catalog as K
from lfg_fly.teacher import pixels as X
from lfg_fly.teacher import render as R


def _png(rgba, top_half_only=False):
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0) if top_half_only else rgba)
    if top_half_only:
        im.paste(Image.new("RGBA", (16, 8), rgba), (0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _bank(tmp_path, layers):
    values = {s: ["None"] for s in SLOTS}
    for (slot, value), png in layers.items():
        path = K.layer_cache_path(tmp_path, "male", slot, value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        values[slot] = [value] + [v for v in values[slot] if v != value]
    cat = K.Catalog(body="male", values=values, odds={}, api="http://lfg")
    return R.LayerBank(cat, tmp_path, size=16)  # native size: no resampling


def _look(**traits):
    return tuple(traits.get(s, "None") for s in SLOTS)


def test_character_mask_covers_every_layer_but_the_background(tmp_path):
    bank = _bank(tmp_path, {("Background", "Red"): _png((255, 0, 0, 255)),
                            ("Head", "Half"): _png((0, 0, 255, 128)),
                            ("Body", "Top"): _png((0, 255, 0, 255), top_half_only=True)})
    m = X.character_mask([_look(Background="Red"), _look(Background="Red", Head="Half"),
                          _look(Body="Top")], bank)
    assert m.shape == (3, 1, 16, 16)
    assert torch.all(m[0] == 0)  # the background alone is no character
    assert torch.allclose(m[1], torch.full((1, 16, 16), 128 / 255), atol=0.01)
    assert torch.allclose(m[2, 0, :8], torch.ones(8, 16)) and torch.all(m[2, 0, 8:] == 0)


def test_pixel_stats_are_mean_rgb_and_an_8_bin_luminance_histogram():
    imgs = torch.zeros(2, 3, 4, 4)
    imgs[0] = 1.0  # white
    imgs[1, 0] = 1.0  # red: luminance 0.2126, bin 1
    stats = X.pixel_stats(imgs)
    assert stats.shape == (2, 11)
    assert np.allclose(stats[0, :3], 1.0) and np.allclose(stats[1, :3], [1.0, 0.0, 0.0])
    assert np.allclose(stats[:, 3:].sum(1), 1.0)
    assert stats[0, 3 + 7] == 1.0 and stats[1, 3 + 1] == 1.0


def _figure_ground(char_rgb, bg_rgb):
    """A 4x4 image: the top half is the character, the bottom half the background."""
    img = torch.zeros(1, 3, 4, 4)
    img[0, :, :2] = torch.tensor(char_rgb).view(3, 1, 1)
    img[0, :, 2:] = torch.tensor(bg_rgb).view(3, 1, 1)
    mask = torch.zeros(1, 1, 4, 4)
    mask[0, 0, :2] = 1.0
    return img, mask


@pytest.mark.parametrize(("char", "bg", "contrast", "harmony"), [
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1.0, 0.0),  # black on white: full contrast, no hue
    ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0, 1.0),  # red on red: analogous
    ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0), 0.7874 - 0.2126, 1.0),  # red on cyan: complementary
    ((1.0, 0.0, 0.0), (0.5, 1.0, 0.0), None, -1.0),  # red on chartreuse (90°): clash
])
def test_visual_terms(char, bg, contrast, harmony):
    img, mask = _figure_ground(char, bg)
    t = X.visual_terms(img, mask)
    if contrast is not None:
        assert t["contrast"][0] == pytest.approx(contrast, abs=1e-4)
    assert t["harmony"][0] == pytest.approx(harmony, abs=1e-4)


def test_visual_terms_survive_an_empty_mask():
    img, mask = _figure_ground((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    t = X.visual_terms(img, torch.zeros_like(mask))
    assert np.isfinite(t["contrast"]).all() and np.isfinite(t["harmony"]).all()
