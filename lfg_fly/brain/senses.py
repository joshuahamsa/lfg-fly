"""The fly's senses (spec §2, Senses). Fixed; never trained."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from lfg_fly.connectome.columns import Columns, lattice_xy
from lfg_fly.connectome.neurons import CHANNEL_BY_TYPE, Populations

SLOTS = ("Background", "Back", "Body", "Clothing", "Mouth", "Eyebrows", "Eyes", "Head", "Accessory")
NON_BODY_SLOTS = tuple(s for s in SLOTS if s != "Body")
NONE = "None"
C_REF = 0.7
CODE_NAMES = ("nose", "eyes", "eyes+nose", "all-sensory-brain", "all-sensory+vnc")
_CHANNEL_INDEX = {"lum": 0, "G": 1, "B": 2}


def trait_seed(tag: str, slot: str, value: str) -> int:
    digest = hashlib.sha256(f"lfg-fly/{tag}|{slot}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class Code:
    neurons: np.ndarray
    weights: np.ndarray


def odor_code(
    slot: str, value: str, orn: np.ndarray, orn_glomerulus: np.ndarray, k: int = 4
) -> Code:
    """k whole glomeruli (ORNs converge by type), each ORN scaled 1/sqrt(type size)."""
    rng = np.random.default_rng(trait_seed("odor/v2", slot, value))
    gloms = np.unique(orn_glomerulus)
    pick = rng.choice(len(gloms), size=k, replace=False)
    u = rng.uniform(0.5, 1.0, size=k)
    neurons, weights = [], []
    for gi, ui in zip(pick, u, strict=True):
        members = orn[orn_glomerulus == gloms[gi]]
        neurons.append(members)
        weights.append(np.full(len(members), ui / np.sqrt(len(members)), dtype=np.float32))
    return Code(np.concatenate(neurons).astype(np.int64), np.concatenate(weights))


def pool_code(
    slot: str, value: str, pool: np.ndarray, k: int = 14, tag: str = "allsens/v1"
) -> Code:
    rng = np.random.default_rng(trait_seed(tag, slot, value))
    neurons = rng.choice(pool, size=k, replace=False).astype(np.int64)
    weights = rng.uniform(0.5, 1.0, size=k).astype(np.float32)
    return Code(neurons, weights)


@dataclass(frozen=True)
class Retina:
    neurons: np.ndarray
    px: np.ndarray
    py: np.ndarray
    channel: np.ndarray


def build_retina(cols: Columns, g, size: int = 64) -> Retina:
    x, y = lattice_xy(cols)
    channel = np.array(
        [_CHANNEL_INDEX[CHANNEL_BY_TYPE[t]] for t in g.cell_type[cols.neuron]], np.int64
    )
    px = np.clip(np.rint(x * (size - 1)), 0, size - 1).astype(np.int64)
    py = np.clip(np.rint(y * (size - 1)), 0, size - 1).astype(np.int64)
    return Retina(neurons=cols.neuron.astype(np.int64), px=px, py=py, channel=channel)


_KERNEL = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


def retina_values(images: torch.Tensor, retina: Retina) -> torch.Tensor:
    """images [B,3,S,S] in [0,1] -> photoreceptor values [P,B] (Gaussian 3x3 at each column)."""
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    planes = torch.stack([0.299 * r + 0.587 * g + 0.114 * b, g, b], dim=1)
    kernel = _KERNEL.to(images.device, images.dtype).expand(3, 1, 3, 3)
    blurred = F.conv2d(F.pad(planes, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
    ch = torch.as_tensor(retina.channel, device=images.device)
    py = torch.as_tensor(retina.py, device=images.device)
    px = torch.as_tensor(retina.px, device=images.device)
    return blurred[:, ch, py, px].T


class InputBuilder:
    """Turns looks (and their renders) into an [N,B] drive for one input code."""

    def __init__(self, name: str, n: int, pops: Populations, retina: Retina, g_in: float,
                 device: str = "cuda"):
        if name not in CODE_NAMES:
            raise ValueError(f"unknown input code {name!r}")
        self.name, self.n, self.pops, self.retina = name, n, pops, retina
        self.g_in, self.device = float(g_in), device
        self.uses_eyes = "eyes" in name
        self.g_eye = self.g_in if self.uses_eyes else 1.0
        self._codes: dict[tuple[str, str], Code] = {}

    @property
    def components(self) -> tuple[str, ...]:
        return {"eyes+nose": ("eyes", "nose")}.get(self.name, (self.name,))

    def only(self, component: str) -> InputBuilder:
        if component not in self.components:
            raise ValueError(f"{self.name} has no component {component!r}")
        other = InputBuilder(component, self.n, self.pops, self.retina, self.g_in, self.device)
        other.g_eye = self.g_eye
        return other

    def _code(self, slot: str, value: str) -> Code:
        key = (slot, value)
        if key not in self._codes:
            if self.name in ("nose", "eyes+nose"):
                self._codes[key] = odor_code(slot, value, self.pops.orn, self.pops.orn_glomerulus)
            elif self.name == "all-sensory-brain":
                self._codes[key] = pool_code(slot, value, self.pops.sensory_brain)
            else:
                pool = np.concatenate([self.pops.sensory_brain, self.pops.sensory_vnc])
                self._codes[key] = pool_code(slot, value, pool, tag="allsens+vnc/v1")
        return self._codes[key]

    def rest_key(self) -> str:
        return f"grey@{self.g_eye:g}"

    def rest_drive(self) -> torch.Tensor:
        drive = torch.zeros(self.n, 1, device=self.device)
        drive[torch.as_tensor(self.retina.neurons, device=self.device)] = 0.5 * self.g_eye
        return drive

    def drive(self, looks: list[tuple[str, ...]], images: torch.Tensor | None = None,
              conc: np.ndarray | None = None) -> torch.Tensor:
        b = len(looks)
        drive = torch.zeros(self.n, b, device=self.device)
        rn = torch.as_tensor(self.retina.neurons, device=self.device)
        if self.uses_eyes:
            if images is None:
                raise ValueError(f"{self.name} needs rendered images")
            drive[rn] = self.g_eye * retina_values(images.to(self.device), self.retina)
        else:
            drive[rn] = 0.5 * self.g_eye
        if self.name == "eyes":
            return drive
        if conc is None:
            conc = np.full((b, len(SLOTS)), C_REF, np.float32)
        rows, cols, vals = [], [], []
        for j, look in enumerate(looks):
            for s, (slot, value) in enumerate(zip(SLOTS, look, strict=True)):
                code = self._code(slot, value)
                rows.append(code.neurons)
                cols.append(np.full(len(code.neurons), j, np.int64))
                vals.append(code.weights * np.float32(self.g_in * conc[j, s]))
        index = (torch.as_tensor(np.concatenate(rows), device=self.device),
                 torch.as_tensor(np.concatenate(cols), device=self.device))
        vals_t = torch.as_tensor(np.concatenate(vals), device=self.device)
        drive.index_put_(index, vals_t, accumulate=True)
        return drive
