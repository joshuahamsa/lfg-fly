"""The connectome as an integer-valued CSR (rows = post, cols = pre).

Weights are stored as synapse COUNTS plus a per-neuron sign. The simulator
multiplies spikes by sign*count (exact integers in fp32, because every input sum
is < 2^24) and only then scales each row by g_syn / insum. That keeps spiking
runs bitwise reproducible on the GPU, whatever order cuSPARSE reduces in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,  # photoreceptors: without the tonic bias they can only inhibit
    "dopamine": 1,
    "serotonin": 1,
    "octopamine": 1,
    "unclear": 1,
}
ANN_COLUMNS = ["bodyId", "status", "superclass", "type", "rootSide", "somaSide",
               "assignedOlHex1", "assignedOlHex2"]
_STR = "<U48"


@dataclass(frozen=True)
class Graph:
    body_id: np.ndarray
    superclass: np.ndarray
    cell_type: np.ndarray
    root_side: np.ndarray
    sign: np.ndarray
    crow: np.ndarray
    col: np.ndarray
    count: np.ndarray
    insum: np.ndarray
    min_syn: int
    nt_missing: int

    @property
    def n(self) -> int:
        return len(self.body_id)

    @property
    def ival(self) -> np.ndarray:
        return self.sign[self.col].astype(np.float32) * self.count.astype(np.float32)

    def graph_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.crow, self.col, self.count, self.sign):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(self.min_syn).encode())
        return h.hexdigest()


def load_annotations(path: Path) -> pd.DataFrame:
    return pd.read_feather(path, columns=ANN_COLUMNS)


def load_nt(path: Path) -> pd.DataFrame:
    return pd.read_feather(path, columns=["body", "consensus_nt"])


def load_edges(path: Path, min_syn: int) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.feather as pf

    table = pf.read_table(path, columns=["body_pre", "body_post", "weight"])
    if min_syn > 1:
        table = table.filter(pc.greater_equal(table["weight"], min_syn))
    return table.to_pandas()


def signs_from_nt(bodies: np.ndarray, nt: pd.DataFrame) -> tuple[np.ndarray, int]:
    series = pd.Series(nt["consensus_nt"].to_numpy(), index=nt["body"].to_numpy())
    series = series[~series.index.duplicated(keep="first")]
    values = series.reindex(bodies)
    missing = int(values.isna().sum())
    unknown = sorted(set(values.dropna().unique()) - set(SIGN))
    if unknown:
        raise ValueError(f"unknown consensus_nt values: {unknown}")
    return values.fillna("unclear").map(SIGN).to_numpy(dtype=np.int8), missing


def csr_from_edges(pre, post, count, n: int, device: str = "cpu"):
    import torch

    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)
    count = np.asarray(count)
    keys = torch.as_tensor(post, device=device) * n + torch.as_tensor(pre, device=device)
    order = torch.argsort(keys).cpu().numpy()
    pre, post, count = pre[order], post[order], count[order]
    crow = np.zeros(n + 1, dtype=np.int64)
    crow[1:] = np.cumsum(np.bincount(post, minlength=n))
    insum = np.bincount(post, weights=count, minlength=n).astype(np.float64)
    insum[insum == 0] = 1.0
    if insum.max() >= 2**24:
        raise ValueError("an input sum reaches 2^24: fp32 integer sums would stop being exact")
    return crow, pre, count.astype(np.int32), insum


def _strings(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna("").astype(str).to_numpy(dtype=_STR)


def build_graph(ann: pd.DataFrame, nt: pd.DataFrame, edges: pd.DataFrame, *,
                min_syn: int = 3, device: str = "cpu") -> Graph:
    traced = ann[ann["status"] == "Traced"].reset_index(drop=True)
    bodies = traced["bodyId"].to_numpy(dtype=np.int64)
    index = pd.Index(bodies)
    kept = edges[edges["weight"] >= min_syn]
    pre = index.get_indexer(kept["body_pre"].to_numpy())
    post = index.get_indexer(kept["body_post"].to_numpy())
    ok = (pre >= 0) & (post >= 0)
    sign, missing = signs_from_nt(bodies, nt)
    crow, col, count, insum = csr_from_edges(
        pre[ok], post[ok], kept["weight"].to_numpy()[ok], len(bodies), device
    )
    return Graph(
        body_id=bodies,
        superclass=_strings(traced, "superclass"),
        cell_type=_strings(traced, "type"),
        root_side=_strings(traced, "rootSide"),
        sign=sign,
        crow=crow,
        col=col,
        count=count,
        insum=insum,
        min_syn=min_syn,
        nt_missing=missing,
    )


def with_edges(g: Graph, pre, post, count, device: str = "cpu") -> Graph:
    crow, col, cnt, insum = csr_from_edges(pre, post, count, g.n, device)
    return replace(g, crow=crow, col=col, count=cnt, insum=insum)


def with_sign(g: Graph, sign: np.ndarray) -> Graph:
    return replace(g, sign=np.asarray(sign, dtype=np.int8))


def edge_list(g: Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pre, post, count) of every edge, in CSR order."""
    post = np.repeat(np.arange(g.n, dtype=np.int64), np.diff(g.crow))
    return g.col.copy(), post, g.count.copy()


_ARRAYS = ("body_id", "superclass", "cell_type", "root_side", "sign",
           "crow", "col", "count", "insum")


def save_graph(g: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"min_syn": g.min_syn, "nt_missing": g.nt_missing, "hash": g.graph_hash()})
    np.savez(path, meta=np.array(meta), **{name: getattr(g, name) for name in _ARRAYS})


def load_graph(path: Path) -> Graph:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        g = Graph(min_syn=int(meta["min_syn"]), nt_missing=int(meta["nt_missing"]),
                  **{name: z[name] for name in _ARRAYS})
    if g.graph_hash() != meta["hash"]:
        raise ValueError(f"{path}: graph hash mismatch")
    return g
