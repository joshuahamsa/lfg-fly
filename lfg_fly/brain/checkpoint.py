"""The checkpoint and the loaded brain (spec §2 Checkpoint and versioning).

A `fly-v<N>` manifest pins everything behind a decision: the feather sha256s and
the graph hash, the threshold, the sign map and the NT-missing count, the column
assignment, the brain setting (every constant, the input code), `c_ref`, the
sense-code versions, the LFG commit whose trait_config the renderer used, the
catalog, and the taste head's stats. The head's weights and normalization sit
beside it (`head.npz` + `head.json`, via `readout.save_head`).

`FlyBrain.load` rebuilds the Phase 0 pipeline from the manifest the way
`cli._load_context` and `teacher.grid.features_for` do (graph, populations,
retina from the columns file, `Simulator`, `InputBuilder`, and the layer bank and
pinned z-order when the code has eyes), refusing a graph or column assignment
that differs from the pinned one. The brain knows nothing about the chain: looks
in, features and scores out (spec §1).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lfg_fly import paths
from lfg_fly.brain import readout as R
from lfg_fly.brain.senses import C_REF, SLOTS, InputBuilder, build_retina
from lfg_fly.brain.sim import BrainParams, Simulator
from lfg_fly.connectome import build as B
from lfg_fly.connectome.columns import Columns, load_columns
from lfg_fly.connectome.neurons import populations
from lfg_fly.teacher import grid as G
from lfg_fly.teacher.catalog import Catalog
from lfg_fly.teacher.render import LFG_TRAIT_CONFIG_COMMIT, LayerBank, ZOrder, load_zorder

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
CATALOG_BODY = "male"  # the hero's body class (spec §4.1 step 5)
DEFAULT_CODES = {"odor": "v2", "allsens": "v1"}  # senses.odor_code / pool_code tags
RETINA_SIZE = 64
IMAGE_CACHE_MAX = 4096  # rendered looks kept for re-scoring before the cache is dropped


class CheckpointMismatch(RuntimeError):
    """The data on disk is not what the manifest pins."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Manifest:
    """Spec §2 Checkpoint and versioning. Changing any pinned field is a new version."""

    version: str
    graph_hash: str
    feathers: dict[str, str]  # feather name -> sha256, from `fly fetch`
    min_syn: int
    sign_map: dict
    nt_missing: int
    columns_hash: str | None  # `columns_hash(cols)`; None pins nothing
    setting: dict  # teacher.grid.Setting.as_dict(): brain params, code, g_in
    c_ref: float = C_REF
    codes: dict = field(default_factory=lambda: dict(DEFAULT_CODES))
    lfg_trait_config_commit: str = LFG_TRAIT_CONFIG_COMMIT
    catalog_hash: str | None = None
    # heldout, ci_lo, ci_hi, lam, n_train, n_test, gate_pass
    taste: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def setting_from_dict(d: dict) -> G.Setting:
    return G.Setting(BrainParams(**d["brain"]), str(d["code"]), float(d["g_in"]))


def columns_hash(cols: Columns) -> str:
    """sha256 over the photoreceptor column assignment (neuron, eye, q, r)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(cols.neuron, dtype=np.int64).tobytes())
    h.update("|".join(str(e) for e in cols.eye).encode())
    h.update(np.ascontiguousarray(cols.q, dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(cols.r, dtype=np.int32).tobytes())
    return h.hexdigest()


def catalog_hash(cat: Catalog) -> str:
    """sha256 of the catalog's canonical JSON (as `grid.context_fingerprint` hashes it)."""
    return hashlib.sha256(cat.to_json().encode()).hexdigest()


def write_checkpoint(directory: Path, manifest: Manifest, head: R.TasteHead) -> None:
    if not isinstance(head, R.TasteHead):
        raise TypeError("a checkpoint holds the taste head; the rarity head is a data-dir "
                        "artifact named by its snapshot hash")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    R.save_head(directory, head)
    tmp = directory / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(directory / MANIFEST)


def read_checkpoint(directory: Path) -> tuple[Manifest, R.TasteHead]:
    directory = Path(directory)
    manifest = Manifest.from_dict(json.loads((directory / MANIFEST).read_text(encoding="utf-8")))
    head = R.load_head(directory)
    if not isinstance(head, R.TasteHead):
        raise ValueError(f"{directory}: the checkpoint's head is not a taste head")
    return manifest, head


class FlyBrain:
    """The loaded fly: looks in, readout features and scores out (spec §1, §2)."""

    def __init__(self, manifest: Manifest, head: R.TasteHead, graph: B.Graph, pops, retina,
                 catalog: Catalog, sim: Simulator, builder: InputBuilder, setting: G.Setting,
                 device: str, catalog_dir: Path, bank: LayerBank | None = None,
                 zorder: ZOrder | None = None, catalog_matches: bool = True):
        self.manifest, self.head, self.graph, self.pops, self.retina = (
            manifest, head, graph, pops, retina)
        self.catalog, self.sim, self.builder, self.setting, self.device = (
            catalog, sim, builder, setting, device)
        self.catalog_dir = Path(catalog_dir)
        self.bank, self._zorder, self.catalog_matches = bank, zorder, catalog_matches
        self._ctx = G.Context(graph=graph, pops=pops, retina=retina, catalog=catalog, bank=bank,
                              zorder=zorder, device=device, probe_set=None)
        self._images: dict = {}
        # stats of the last run
        self.stats: dict = {"neurons_fired": 0, "ms": 0.0, "n_looks": 0}

    @classmethod
    def load(cls, checkpoint_dir: Path, device: str, catalog_dir: Path) -> FlyBrain:
        """Graph, populations, retina, Simulator, InputBuilder and head from a checkpoint.

        The graph is `FLY_DATA_DIR/graph/graph-syn<min_syn>.npz` and must hash to the
        manifest's `graph_hash`; the columns file beside it must hash to `columns_hash`
        when that is pinned; the code's sign map must equal the pinned one. The catalog
        is `<catalog_dir>/catalog-male.json`; a hash different from the manifest's is
        logged (the wardrobe may grow) and reported as `catalog_matches`. The layer bank
        and the pinned z-order are built only when the input code has eyes.
        """
        checkpoint_dir, catalog_dir = Path(checkpoint_dir), Path(catalog_dir)
        manifest, head = read_checkpoint(checkpoint_dir)
        setting = setting_from_dict(manifest.setting)
        if manifest.sign_map != B.SIGN:
            raise CheckpointMismatch(f"{checkpoint_dir}: the pinned sign map differs from the "
                                     "code's (connectome.build.SIGN)")
        gdir = paths.graph_dir()
        g = B.load_graph(gdir / f"graph-syn{manifest.min_syn}.npz")
        if g.graph_hash() != manifest.graph_hash:
            raise CheckpointMismatch(f"{checkpoint_dir}: graph hash {g.graph_hash()[:16]} is not "
                                     f"the pinned {manifest.graph_hash[:16]}")
        if g.nt_missing != manifest.nt_missing:
            log.warning("%s: graph has %d NT-missing bodies, the manifest pins %d",
                        checkpoint_dir, g.nt_missing, manifest.nt_missing)
        cols = load_columns(gdir / f"columns-syn{manifest.min_syn}.npz")
        if manifest.columns_hash is not None and columns_hash(cols) != manifest.columns_hash:
            raise CheckpointMismatch(f"{checkpoint_dir}: the column assignment is not the "
                                     "pinned one")
        pops = populations(g)
        retina = build_retina(cols, g, size=RETINA_SIZE)
        cat = Catalog.from_json(
            (catalog_dir / f"catalog-{CATALOG_BODY}.json").read_text(encoding="utf-8"))
        matches = manifest.catalog_hash is None or catalog_hash(cat) == manifest.catalog_hash
        if not matches:
            log.warning("%s: the catalog in %s is not the one the head was trained on "
                        "(pinned %s)", checkpoint_dir, catalog_dir, manifest.catalog_hash[:16])
        builder = InputBuilder(setting.code, g.n, pops, retina, setting.g_in, device)
        bank = zorder = None
        if builder.uses_eyes:
            zorder = load_zorder(catalog_dir, manifest.lfg_trait_config_commit)
            bank = LayerBank(cat, catalog_dir, size=RETINA_SIZE, device=device)
        sim = Simulator(g, pops.sensory_any, pops.readout, device)
        return cls(manifest, head, g, pops, retina, cat, sim, builder, setting, device,
                   catalog_dir, bank=bank, zorder=zorder, catalog_matches=matches)

    @property
    def zorder(self) -> ZOrder | None:
        """The pinned z-order, loaded with the bank when the code has eyes."""
        return self._zorder

    def pinned_zorder(self) -> ZOrder:
        """The z-order of the manifest's trait_config commit (loads it from the catalog
        dir's cache on first use), for the loop's `/api/economy` cross-check."""
        if self._zorder is None:
            self._zorder = load_zorder(self.catalog_dir, self.manifest.lfg_trait_config_commit)
        return self._zorder

    def _c_ref(self, n: int) -> np.ndarray:
        return np.full((n, len(SLOTS)), self.manifest.c_ref, dtype=np.float32)

    def features(self, looks, conc: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Readout features [n, n_readout] via `teacher.grid.features_for`, and the run's
        stats. `conc` [n, 9] are the per-trait concentrations; None pins every trait to
        the manifest's `c_ref` (the taste simulation)."""
        looks = [tuple(lk) for lk in looks]
        n = len(looks)
        if conc is None:
            conc = self._c_ref(n)
        else:
            conc = np.asarray(conc, dtype=np.float32)
            if conc.shape != (n, len(SLOTS)):
                raise ValueError(f"conc has shape {conc.shape}, expected ({n}, {len(SLOTS)})")
        if n == 0:
            self.stats = {"neurons_fired": 0, "ms": 0.0, "n_looks": 0}
            return np.zeros((0, len(self.pops.readout)), np.float32), self.stats
        if len(self._images) > IMAGE_CACHE_MAX:
            self._images.clear()
        t0 = time.perf_counter()
        X, stats = G.features_for(self._ctx, self.sim, self.setting, looks, self._images,
                                  conc=conc, builder=self.builder)
        ms = (time.perf_counter() - t0) * 1000.0
        # features_for reports the mean active fraction over the batch; times N that is the
        # mean number of neurons that fired per look
        self.stats = {**stats, "neurons_fired": int(round(stats["active_frac"] * self.graph.n)),
                      "ms": ms, "n_looks": n}
        return X, self.stats

    def taste(self, looks) -> np.ndarray:
        """Taste under `c_ref` (spec §2: rarity-invariant by construction) → `head.score`."""
        X, _ = self.features(looks)
        return self.head.score(X)

    def rarity(self, looks, conc: np.ndarray, head: R.RarityHead) -> np.ndarray:
        """The second, live-concentration simulation through the nightly rarity head."""
        X, _ = self.features(looks, conc)
        return head.score(X)
