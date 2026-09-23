"""Candidate brains on the connectome (spec §2, Candidate brains).

All kinds share: dt = 1 ms, the integer CSR (syn = (W_int @ x) * (1/insum),
bit-reproducible run to run on CUDA too, via brain/spmm.py),
a tonic bias on every NON-sensory neuron, and a cached resting state reached by
a noise-free burn-in under `rest_drive` (the mid-grey retina, no odour).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from lfg_fly.brain.spmm import CudaCsr
from lfg_fly.connectome.build import Graph

BRAIN_KINDS = ("lif", "lif-avg", "lif-volley", "rate")
ACTIVE_RATE = 0.05  # a rate unit counts as "active" above this
SATURATED_RATE = 9.0  # the rate analogue of "firing" in the runaway rule (clip is 10)


@dataclass(frozen=True)
class BrainParams:
    kind: str
    g_syn: float
    bias: float = 0.0
    steps: int = 60
    tau_ms: float = 20.0
    refractory: int = 2
    trials: int = 1
    noise: float = 0.0
    alpha: float = 0.7
    burn_in: int = 200
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in BRAIN_KINDS:
            raise ValueError(f"unknown brain kind {self.kind!r}")
        if self.kind == "lif-avg":
            if self.trials < 2 or self.noise <= 0:
                raise ValueError("lif-avg needs trials >= 2 and noise > 0")
        elif self.trials != 1 or self.noise != 0:
            raise ValueError(f"{self.kind} takes trials=1 and noise=0")

    @property
    def spiking(self) -> bool:
        return self.kind != "rate"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimResult:
    features: torch.Tensor
    active_frac: torch.Tensor
    readout_active_frac: torch.Tensor
    max_step_frac: float
    neurons_fired: torch.Tensor
    total_spikes: torch.Tensor
    readout_spikes: torch.Tensor


class Simulator:
    def __init__(self, g: Graph, sensory_any: np.ndarray, readout: np.ndarray,
                 device: str = "cuda"):
        self.n = g.n
        self.device = device
        if torch.device(device).type == "cuda":
            # torch's CUDA CSR matmul (cuSPARSE) is not bit-reproducible; see brain/spmm.py
            self.W = CudaCsr(g.crow, g.col, g.ival, (g.n, g.n), device)
        else:
            self.W = torch.sparse_csr_tensor(
                torch.as_tensor(g.crow, dtype=torch.int64),
                torch.as_tensor(g.col, dtype=torch.int64),
                torch.as_tensor(g.ival, dtype=torch.float32),
                size=(g.n, g.n),
            )
        self.scale = torch.as_tensor(
            (1.0 / g.insum).astype(np.float32), device=device).unsqueeze(1)
        self.nonsensory = torch.as_tensor(
            ~np.asarray(sensory_any), device=device).float().unsqueeze(1)
        self.readout = torch.as_tensor(np.asarray(readout, dtype=np.int64), device=device)
        self._rest: dict[tuple, dict[str, torch.Tensor]] = {}

    def _syn(self, x: torch.Tensor) -> torch.Tensor:
        return (self.W @ x) * self.scale

    def _lif(self, v, ref, s, drive, p: BrainParams, steps: int, gen, record: bool):
        decay = math.exp(-1.0 / p.tau_ms)
        count = torch.zeros_like(v)
        max_frac = 0.0
        zero = torch.zeros((), device=v.device)
        for _ in range(steps):
            x = drive
            if gen is not None:
                x = x + p.noise * torch.randn(v.shape, generator=gen, device=v.device)
            v = v * decay + p.g_syn * self._syn(s) + x
            v = torch.where(ref > 0, zero, v)
            s = (v >= 1.0).float()
            v = torch.where(s > 0, zero, v)
            ref = torch.clamp(ref - 1.0, min=0.0) + p.refractory * s
            count += s
            if record:
                max_frac = max(max_frac, float(s.mean(0).max()))
        return v, ref, s, count, max_frac

    def _rate(self, h, drive, p: BrainParams, steps: int, record: bool):
        max_frac = 0.0
        for _ in range(steps):
            pre = torch.relu(p.g_syn * self._syn(h) + drive)
            h = (1 - p.alpha) * h + p.alpha * torch.clamp(pre, 0.0, 10.0)
            if record:
                max_frac = max(max_frac, float((h > SATURATED_RATE).float().mean(0).max()))
        return h, max_frac

    def rest(self, p: BrainParams, rest_drive: torch.Tensor, key: str) -> dict[str, torch.Tensor]:
        cache_key = (p.kind, p.g_syn, p.bias, p.tau_ms, p.refractory, p.alpha, p.burn_in, key)
        if cache_key not in self._rest:
            drive = rest_drive + p.bias * self.nonsensory
            if p.spiking:
                z = torch.zeros(self.n, 1, device=self.device)
                v, ref, s, _, _ = self._lif(
                    z, z.clone(), z.clone(), drive, p, p.burn_in, None, False)
                self._rest[cache_key] = {"v": v, "ref": ref, "s": s}
            else:
                z = torch.zeros(self.n, 1, device=self.device)
                h, _ = self._rate(z, drive, p, p.burn_in, False)
                self._rest[cache_key] = {"h": h}
        return self._rest[cache_key]

    @torch.no_grad()
    def run(self, drive: torch.Tensor, p: BrainParams, rest_drive: torch.Tensor,
            rest_key: str) -> SimResult:
        state = self.rest(p, rest_drive, rest_key)
        b = drive.shape[1]
        full = drive + p.bias * self.nonsensory
        if p.spiking:
            r = p.trials
            full = full.repeat(1, r)  # [N, R*B] as R blocks of B
            v, ref, s = (state[k].expand(self.n, r * b).clone() for k in ("v", "ref", "s"))
            gen = None
            if p.noise > 0:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(p.seed)
            _, _, _, count, max_frac = self._lif(v, ref, s, full, p, p.steps, gen, True)
            count = count.view(self.n, r, b).mean(1)
            ro = count[self.readout]
            active = (count > 0).float()
            return SimResult(
                features=ro.T.contiguous(),
                active_frac=active.mean(0),
                readout_active_frac=(ro > 0).float().mean(0),
                max_step_frac=max_frac,
                neurons_fired=active.sum(0).long(),
                total_spikes=count.sum(0),
                readout_spikes=ro.sum(0),
            )
        h = state["h"].expand(self.n, b).clone()
        h, max_frac = self._rate(h, full, p, p.steps, True)
        ro = h[self.readout]
        active = (h > ACTIVE_RATE).float()
        return SimResult(
            features=ro.T.contiguous(),
            active_frac=active.mean(0),
            readout_active_frac=(ro > ACTIVE_RATE).float().mean(0),
            max_step_frac=max_frac,
            neurons_fired=active.sum(0).long(),
            total_spikes=h.sum(0),
            readout_spikes=ro.sum(0),
        )


def simulate_reference(g: Graph, sensory_any: np.ndarray, readout: np.ndarray, drive: np.ndarray,
                       p: BrainParams, rest_drive: np.ndarray) -> np.ndarray:
    """Dense numpy twin of Simulator.run for noise-free kinds (tests only; tiny graphs)."""
    if p.noise:
        raise ValueError("the reference covers noise-free kinds only")
    n = g.n
    W = np.zeros((n, n), np.float32)
    post = np.repeat(np.arange(n), np.diff(g.crow))
    W[post, g.col] = g.ival
    scale = (1.0 / g.insum).astype(np.float32)[:, None]
    bias = (p.bias * (~np.asarray(sensory_any))).astype(np.float32)[:, None]
    syn = lambda x: (W @ x).astype(np.float32) * scale  # noqa: E731

    if p.spiking:
        decay = np.float32(math.exp(-1.0 / p.tau_ms))

        def lif(v, ref, s, d, steps):
            count = np.zeros_like(v)
            for _ in range(steps):
                v = v * decay + np.float32(p.g_syn) * syn(s) + d
                v = np.where(ref > 0, np.float32(0), v).astype(np.float32)
                s = (v >= 1.0).astype(np.float32)
                v = np.where(s > 0, np.float32(0), v).astype(np.float32)
                ref = np.maximum(ref - 1.0, 0.0).astype(np.float32) + np.float32(p.refractory) * s
                count += s
            return v, ref, s, count

        z = np.zeros((n, 1), np.float32)
        v, ref, s, _ = lif(z, z.copy(), z.copy(), rest_drive + bias, p.burn_in)
        b = drive.shape[1]
        v, ref, s = (np.repeat(x, b, axis=1) for x in (v, ref, s))
        _, _, _, count = lif(v, ref, s, drive + bias, p.steps)
        return count[readout].T

    def rate(h, d, steps):
        for _ in range(steps):
            pre = np.clip(np.maximum(p.g_syn * syn(h) + d, 0), 0, 10)
            h = ((1 - p.alpha) * h + p.alpha * pre).astype(np.float32)
        return h

    h = rate(np.zeros((n, 1), np.float32), rest_drive + bias, p.burn_in)
    h = rate(np.repeat(h, drive.shape[1], axis=1), drive + bias, p.steps)
    return h[readout].T
