"""Pixel features of a rendered look (spec §3.0b, §3.4).

- `pixel_stats`: mean RGB + an 8-bin luminance histogram, the raw pixel inputs of
  the no-brain controls (§3.4 controls 2 and 3).
- `character_mask` and `visual_terms`: the visual taste's two interaction terms,
  figure-ground contrast and hue harmony between the character and its background.
"""

from __future__ import annotations

import colorsys

import numpy as np
import torch

from lfg_fly.brain.senses import NONE, SLOTS
from lfg_fly.teacher.render import LayerBank

LUMA = (0.2126, 0.7152, 0.0722)
HIST_BINS = 8
BACKGROUND = "Background"


def luminance(imgs: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B,H,W]."""
    w = torch.tensor(LUMA, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
    return (imgs * w).sum(1)


def pixel_stats(imgs: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] in [0,1] -> [B, 3 + 8]: mean RGB, then the fraction of pixels per
    luminance bin (bin 7 includes 1.0)."""
    mean_rgb = imgs.mean((2, 3))
    lum = luminance(imgs).flatten(1)
    bins = (lum * HIST_BINS).long().clamp_(0, HIST_BINS - 1)
    hist = torch.nn.functional.one_hot(bins, HIST_BINS).float().mean(1)
    return torch.cat([mean_rgb, hist], 1).cpu().numpy().astype(np.float32)


def character_mask(looks: list[tuple[str, ...]], bank: LayerBank) -> torch.Tensor:
    """[B,1,S,S]: the coverage of every drawn layer except the Background,
    1 − Π(1 − alpha), which is independent of z-order."""
    s = bank.size
    out = torch.zeros(len(looks), 1, s, s, device=bank.stack.device)
    for b, look in enumerate(looks):
        clear = torch.ones(1, s, s, device=bank.stack.device)
        for slot, value in zip(SLOTS, look, strict=True):
            if slot != BACKGROUND and value != NONE:
                clear = clear * (1.0 - bank.stack[bank.index[(slot, value)]][3:4])
        out[b] = 1.0 - clear
    return out


def _hue_sat(rgb: np.ndarray) -> tuple[float, float]:
    h, s, _ = colorsys.rgb_to_hsv(*(float(c) for c in rgb))
    return h, s


def visual_terms(imgs: torch.Tensor, mask: torch.Tensor) -> dict[str, np.ndarray]:
    """The character's and the background's mean colours (weighted by the mask and its
    complement), then per look:
    - contrast: |L(character) − L(background)|;
    - harmony: s_c · s_b · cos(2Δhue), +1 for matching or complementary hues and −1 for
      hues a quarter-turn apart, fading to 0 as either colour greys out.
    """
    eps = 1e-6
    char = (imgs * mask).sum((2, 3)) / (mask.sum((2, 3)) + eps)
    bg = (imgs * (1.0 - mask)).sum((2, 3)) / ((1.0 - mask).sum((2, 3)) + eps)
    w = torch.tensor(LUMA, device=imgs.device, dtype=imgs.dtype)
    contrast = ((char - bg) @ w).abs().cpu().numpy()
    char, bg = char.clamp(0, 1).cpu().numpy(), bg.clamp(0, 1).cpu().numpy()
    harmony = np.empty(len(char))
    for i, (c, b) in enumerate(zip(char, bg, strict=True)):
        (hc, sc), (hb, sb) = _hue_sat(c), _hue_sat(b)
        harmony[i] = sc * sb * np.cos(4.0 * np.pi * (hc - hb))
    return {"contrast": contrast.astype(np.float64), "harmony": harmony}
