"""Give every photoreceptor an ommatidial column (spec §2, Eyes).

No photoreceptor carries assignedOlHex in v1.0; only columnar ol_intrinsic cells
do (L1-L3, L5, C2, C3, Mi1/4/9, Tm1/2/4/9/20, T1). A photoreceptor takes the
column (somaSide, hex1, hex2) receiving most of its output synapses among
hex-labelled cells, at ANY synapse weight. Its eye is that column's somaSide
(fallback: its own rootSide).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lfg_fly.connectome.build import Graph
from lfg_fly.connectome.neurons import Populations

MIN_ASSIGNED = 0.90


@dataclass(frozen=True)
class Columns:
    neuron: np.ndarray
    eye: np.ndarray
    q: np.ndarray
    r: np.ndarray
    share: np.ndarray
    dropped: int
    total: int


def load_pr_edges(path: Path, pr_bodies: np.ndarray) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as pf

    table = pf.read_table(path, columns=["body_pre", "body_post", "weight"])
    keep = pa.array(pr_bodies, type=pa.int64())
    table = table.filter(pc.is_in(table["body_pre"], value_set=keep))
    return table.to_pandas()


def assign_columns(
    g: Graph, pops: Populations, ann: pd.DataFrame, edges_any: pd.DataFrame
) -> Columns:
    hexed = ann[
        (ann["status"] == "Traced")
        & ann["assignedOlHex1"].notna()
        & ann["assignedOlHex2"].notna()
    ]
    column_of = pd.DataFrame({
        "body_post": hexed["bodyId"].to_numpy(dtype=np.int64),
        "side": hexed["somaSide"].fillna("").astype(str).to_numpy(),
        "q": hexed["assignedOlHex1"].astype(int).to_numpy(),
        "r": hexed["assignedOlHex2"].astype(int).to_numpy(),
    })
    pr_bodies = g.body_id[pops.photoreceptor]
    pr_edges = edges_any[edges_any["body_pre"].isin(pr_bodies)]
    e = pr_edges.merge(column_of, on="body_post", how="inner")
    agg = e.groupby(["body_pre", "side", "q", "r"], as_index=False)["weight"].sum()
    agg["share"] = agg["weight"] / agg.groupby("body_pre")["weight"].transform("sum")
    best = (
        agg.sort_values(
            ["body_pre", "weight", "side", "q", "r"],
            ascending=[True, False, True, True, True],
        )
        .drop_duplicates("body_pre")
    )
    idx = pd.Index(g.body_id).get_indexer(best["body_pre"].to_numpy())
    side = best["side"].to_numpy().astype("<U1")
    eye = np.where(np.isin(side, ["L", "R"]), side, g.root_side[idx].astype("<U1"))
    ok = np.isin(eye, ["L", "R"])
    total = len(pops.photoreceptor)
    assigned = int(ok.sum())
    if assigned < MIN_ASSIGNED * total:
        raise ValueError(
            f"only {assigned}/{total} photoreceptors assigned a column (< {MIN_ASSIGNED:.0%})"
        )
    order = np.argsort(idx[ok])
    return Columns(
        neuron=idx[ok][order].astype(np.int64),
        eye=eye[ok][order],
        q=best["q"].to_numpy()[ok][order].astype(np.int32),
        r=best["r"].to_numpy()[ok][order].astype(np.int32),
        share=best["share"].to_numpy()[ok][order].astype(np.float32),
        dropped=total - assigned,
        total=total,
    )


def lattice_xy(cols: Columns) -> tuple[np.ndarray, np.ndarray]:
    """Axial hex (q, r) -> (q + r/2, r*sqrt(3)/2), min-max normalized within each eye."""
    x = cols.q + cols.r / 2.0
    y = cols.r * (np.sqrt(3.0) / 2.0)
    xn, yn = np.empty_like(x, dtype=np.float64), np.empty_like(y, dtype=np.float64)
    for side in ("L", "R"):
        m = cols.eye == side
        if not m.any():
            continue
        for src, dst in ((x, xn), (y, yn)):
            lo, hi = src[m].min(), src[m].max()
            dst[m] = 0.5 if hi == lo else (src[m] - lo) / (hi - lo)
    return xn, yn


def save_columns(cols: Columns, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, neuron=cols.neuron, eye=cols.eye, q=cols.q, r=cols.r, share=cols.share,
             dropped=np.array(cols.dropped), total=np.array(cols.total))


def load_columns(path: Path) -> Columns:
    with np.load(path, allow_pickle=False) as z:
        return Columns(neuron=z["neuron"], eye=z["eye"], q=z["q"], r=z["r"], share=z["share"],
                       dropped=int(z["dropped"]), total=int(z["total"]))
