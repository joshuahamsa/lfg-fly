"""Composite looks like LFG does: layers stacked by z (trait_config.yaml at a
pinned LFG commit, incl. z_overrides), None slots skipped, first frame of any
animated layer. Runs on the device at retina resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from lfg_fly.brain.senses import NONE, SLOTS
from lfg_fly.teacher.catalog import Catalog, http_get, layer_cache_path

LFG_TRAIT_CONFIG_COMMIT = "0415a63afdcea11156b57169d34e38bb8d377adc"
RAW_URL = "https://raw.githubusercontent.com/Team-Hamsa/LFG/{commit}/trait_config.yaml"


@dataclass(frozen=True)
class ZOrder:
    layer_z: dict[str, float]
    overrides: dict[tuple[str, str], float]
    rank: dict[str, int]

    def key(self, slot: str, value: str) -> tuple[float, int]:
        return (self.overrides.get((slot, value), self.layer_z[slot]), self.rank[slot])


def zorder_from_config(cfg: dict) -> ZOrder:
    layers = cfg["layers"]
    return ZOrder(
        layer_z={row["name"]: float(row["z"]) for row in layers},
        overrides={
            (o["trait_type"], o["value"]): float(o["z"])
            for o in cfg.get("z_overrides") or []
        },
        rank={row["name"]: i for i, row in enumerate(layers)},
    )


def load_zorder(cache: Path, commit: str = LFG_TRAIT_CONFIG_COMMIT, get=http_get) -> ZOrder:
    path = cache / f"trait_config-{commit}.yaml"
    if not path.exists():
        status, data = get(RAW_URL.format(commit=commit))
        if status != 200:
            raise RuntimeError(f"trait_config.yaml@{commit}: HTTP {status}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return zorder_from_config(yaml.safe_load(path.read_text()))


class LayerBank:
    """Every catalog layer as a premultiplied RGBA tensor [K,4,S,S] on the device."""

    def __init__(self, catalog: Catalog, cache: Path, size: int = 64, device: str = "cpu"):
        entries = [(s, v) for s in SLOTS for v in catalog.values.get(s, []) if v != NONE]
        tiles = []
        for slot, value in entries:
            with Image.open(layer_cache_path(cache, catalog.body, slot, value)) as im:
                im.seek(0)
                rgba = im.convert("RGBA").resize((size, size), Image.LANCZOS)
            a = np.asarray(rgba, dtype=np.float32) / 255.0
            tiles.append(np.concatenate([a[..., :3] * a[..., 3:4], a[..., 3:4]], axis=-1))
        self.size = size
        self.index = {e: i for i, e in enumerate(entries)}
        stack = np.stack(tiles) if tiles else np.zeros((0, size, size, 4), np.float32)
        self.stack = torch.as_tensor(stack).permute(0, 3, 1, 2).contiguous().to(device)


def composite(looks: list[tuple[str, ...]], bank: LayerBank, zorder: ZOrder) -> torch.Tensor:
    s = bank.size
    out = torch.zeros(len(looks), 3, s, s, device=bank.stack.device)
    for b, look in enumerate(looks):
        layers = sorted(
            ((slot, v) for slot, v in zip(SLOTS, look, strict=True) if v != NONE),
            key=lambda sv: zorder.key(*sv),
        )
        acc = torch.zeros(3, s, s, device=bank.stack.device)
        for sv in layers:
            tile = bank.stack[bank.index[sv]]
            acc = tile[:3] + acc * (1.0 - tile[3:4])
        out[b] = acc
    return out.clamp_(0.0, 1.0)
