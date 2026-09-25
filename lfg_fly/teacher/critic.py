"""The critic's round (spec §3.2): pairs, cards, the QC schedule, batches, assembly.

A round is a set of look pairs rendered as 1024×512 cards (A left, B right) that
Claude Code workflow agents judge 40 at a time. Quality control is built into the
schedule: FLIP_FRAC of the pairs are shown again with the sides swapped (position
bias; a pair the critic flips on is dropped), and DUP_FRAC are judged again by a
second agent (agreement `a`, and the implied ceiling (1+sqrt(2a-1))/2). Flips and
dups live in their own batches, so no agent sees a pair twice.

Everything here is deterministic in (catalog, seed). The manifest records every
pair, item and batch, and `assemble` turns the agents' result files into
`data/critic/<round>.jsonl`, in the pair's canonical orientation.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lfg_fly.brain.senses import NONE, SLOTS
from lfg_fly.teacher.catalog import Catalog, layer_cache_path
from lfg_fly.teacher.render import ZOrder
from lfg_fly.teacher.sample import Look, Pair, make_pairs

ROUND1_SEED = 2000  # disjoint from the probes' seeds (0 selection, 1000 confirmation)
CARD_SIDE = 512
CARD_SIZE = (2 * CARD_SIDE, CARD_SIDE)
BATCH = 40
FLIP_FRAC = 0.10
DUP_FRAC = 0.10
STRENGTHS = (1, 2, 3)
FONT_CANDIDATES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",)


@dataclass(frozen=True)
class Item:
    """One thing an agent judges: a pair shown in some orientation."""

    item_id: str
    pair_id: int
    kind: str  # main | flip | dup
    swapped: bool  # True: the card's left (A) is the pair's canonical b
    image: str  # file name under the round directory

    def as_dict(self) -> dict:
        return asdict(self)


def round_pairs(cat: Catalog, n_pairs: int = 3000, pairs_per_family: int = 3,
                seed: int = ROUND1_SEED) -> list[Pair]:
    return make_pairs(cat, n_pairs // pairs_per_family, pairs_per_family, seed=seed)


def schedule(n_pairs: int, seed: int, flip_frac: float = FLIP_FRAC, dup_frac: float = DUP_FRAC,
             batch: int = BATCH) -> tuple[list[Item], list[list[str]]]:
    """Items and batches for a round. Main items are shuffled into batches of `batch`;
    flips and dups are shuffled into their own batches, so a pair's second showing is
    always judged by a different agent than its first."""
    rng = np.random.default_rng(seed + 1)
    n_flip = int(round(flip_frac * n_pairs))
    n_dup = int(round(dup_frac * n_pairs))
    # disjoint, so no pair has two second showings (and no QC batch shows a pair twice)
    picked = rng.choice(n_pairs, size=n_flip + n_dup, replace=False).tolist()
    flips, dups = sorted(picked[:n_flip]), sorted(picked[n_flip:])
    items = [Item(f"p{i:04d}", i, "main", False, f"p{i:04d}.png") for i in range(n_pairs)]
    items += [Item(f"p{i:04d}f", i, "flip", True, f"p{i:04d}f.png") for i in flips]
    items += [Item(f"p{i:04d}d", i, "dup", False, f"p{i:04d}.png") for i in dups]
    mains = [it.item_id for it in items if it.kind == "main"]
    qc = [it.item_id for it in items if it.kind != "main"]
    mains = [mains[i] for i in rng.permutation(len(mains))]
    qc = [qc[i] for i in rng.permutation(len(qc))]
    batches = [mains[i: i + batch] for i in range(0, len(mains), batch)]
    batches += [qc[i: i + batch] for i in range(0, len(qc), batch)]
    return items, batches


class PilBank:
    """Catalog layers as RGBA PIL images at CARD_SIDE, loaded on demand (first frame)."""

    def __init__(self, cat: Catalog, cache: Path, side: int = CARD_SIDE):
        self.cat, self.cache, self.side = cat, cache, side
        self._tiles: dict[tuple[str, str], Image.Image] = {}

    def tile(self, slot: str, value: str) -> Image.Image:
        key = (slot, value)
        if key not in self._tiles:
            with Image.open(layer_cache_path(self.cache, self.cat.body, slot, value)) as im:
                im.seek(0)
                rgba = im.convert("RGBA")
                if rgba.size != (self.side, self.side):
                    rgba = rgba.resize((self.side, self.side), Image.LANCZOS)
            self._tiles[key] = rgba
        return self._tiles[key]


def composite_pil(look: Look, bank: PilBank, zorder: ZOrder) -> Image.Image:
    """LFG's compose: layers alpha-over in z order, None skipped, on a transparent base."""
    out = Image.new("RGBA", (bank.side, bank.side), (0, 0, 0, 0))
    layers = sorted(((s, v) for s, v in zip(SLOTS, look, strict=True) if v != NONE),
                    key=lambda sv: zorder.key(*sv))
    for slot, value in layers:
        out.alpha_composite(bank.tile(slot, value))
    return out


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def card(left: Image.Image, right: Image.Image) -> Image.Image:
    """A 1024×512 card: `left` labelled A, `right` labelled B, a divider between."""
    side = left.size[0]
    out = Image.new("RGB", (2 * side, side), (24, 24, 24))
    out.paste(left.convert("RGB"), (0, 0), left)
    out.paste(right.convert("RGB"), (side, 0), right)
    draw = ImageDraw.Draw(out)
    draw.rectangle([side - 2, 0, side + 1, side], fill=(255, 255, 255))
    font = _font(max(24, side // 10))
    for x, label in ((0, "A"), (side, "B")):
        box = draw.textbbox((0, 0), label, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        pad = side // 40
        draw.rectangle([x + pad, pad, x + pad + w + 2 * pad, pad + h + 2 * pad],
                       fill=(0, 0, 0))
        draw.text((x + 2 * pad - box[0], 2 * pad - box[1]), label, fill=(255, 255, 255),
                  font=font)
    return out


def write_round(cat: Catalog, cache: Path, zorder: ZOrder, out_dir: Path, round_name: str,
                n_pairs: int = 3000, seed: int = ROUND1_SEED, batch: int = BATCH) -> dict:
    """Render every card and write the manifest and the per-batch files.

    Resumable: cards that exist are not re-rendered. Returns the manifest."""
    pairs = round_pairs(cat, n_pairs=n_pairs, seed=seed)
    items, batches = schedule(len(pairs), seed, batch=batch)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batches").mkdir(exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)
    bank = PilBank(cat, cache)
    looks: dict[Look, Image.Image] = {}

    def render(look: Look) -> Image.Image:
        if look not in looks:
            looks[look] = composite_pil(look, bank, zorder)
        return looks[look]

    for it in items:
        if it.kind == "dup":
            continue  # a dup reuses the main card
        path = out_dir / it.image
        if path.exists():
            continue
        pr = pairs[it.pair_id]
        a, b = (pr.b, pr.a) if it.swapped else (pr.a, pr.b)
        card(render(a), render(b)).save(path, format="PNG", optimize=False)
    by_id = {it.item_id: it for it in items}
    batch_ids = []
    for i, ids in enumerate(batches):
        name = f"b{i:03d}"
        batch_ids.append(name)
        (out_dir / "batches" / f"{name}.json").write_text(json.dumps({
            "batch": name, "round": round_name,
            "items": [{"item_id": x, "image": str(out_dir / by_id[x].image)} for x in ids],
        }, indent=1), encoding="utf-8")
    manifest = {
        "round": round_name, "seed": seed, "n_pairs": len(pairs), "batch": batch,
        "catalog_body": cat.body, "card_size": list(CARD_SIZE),
        "pairs": [{"pair_id": i, "family": p.family, "near": p.near, "a": list(p.a),
                   "b": list(p.b)} for i, p in enumerate(pairs)],
        "items": [it.as_dict() for it in items],
        "batches": {name: ids for name, ids in zip(batch_ids, batches, strict=True)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- assembly and QC


class ResultError(ValueError):
    """A batch result file is missing, malformed or incomplete."""


def read_results(manifest: dict, results_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """Every verdict keyed by item_id, and the batches that are missing or incomplete.

    A verdict is kept iff its item is in the batch, its winner is A or B and its
    strength is 1–3. A batch with any item missing is listed as incomplete (its
    good verdicts are still kept)."""
    verdicts: dict[str, dict] = {}
    bad: list[str] = []
    for name, ids in manifest["batches"].items():
        path = results_dir / f"{name}.json"
        if not path.exists():
            bad.append(name)
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            rows = rows["verdicts"] if isinstance(rows, dict) else rows
        except (ValueError, KeyError, TypeError):
            bad.append(name)
            continue
        want = set(ids)
        got = set()
        for r in rows:
            if not isinstance(r, dict) or r.get("item_id") not in want:
                continue
            if r.get("winner") not in ("A", "B") or r.get("strength") not in STRENGTHS:
                continue
            if r["item_id"] in got:
                continue
            got.add(r["item_id"])
            verdicts[r["item_id"]] = {
                "winner": r["winner"], "strength": int(r["strength"]),
                "idea_A": str(r.get("idea_A", ""))[:80], "idea_B": str(r.get("idea_B", ""))[:80],
                "reason": str(r.get("reason", ""))[:200], "batch": name,
            }
        if got != want:
            bad.append(name)
    return verdicts, bad


def _canonical(v: dict, swapped: bool) -> dict:
    """A verdict in the pair's canonical orientation (winner "A" = the pair's `a`)."""
    if not swapped:
        return dict(v)
    return {**v, "winner": "A" if v["winner"] == "B" else "B",
            "idea_A": v["idea_B"], "idea_B": v["idea_A"]}


def assemble(manifest: dict, verdicts: dict[str, dict]) -> list[dict]:
    """One record per pair with a main verdict, flip and dup verdicts attached and
    mapped to the canonical orientation, and `flipped` (the position-bias check failed:
    the critic chose the left side both times)."""
    by_pair: dict[int, dict] = {}
    for it in manifest["items"]:
        v = verdicts.get(it["item_id"])
        if v is None:
            continue
        by_pair.setdefault(it["pair_id"], {})[it["kind"]] = _canonical(v, it["swapped"])
    out = []
    for p in manifest["pairs"]:
        got = by_pair.get(p["pair_id"], {})
        if "main" not in got:
            continue
        main = got["main"]
        rec = {"pair_id": p["pair_id"], "family": p["family"], "near": p["near"],
               "a": p["a"], "b": p["b"], **main}
        if "flip" in got:
            rec["flip"] = got["flip"]
            rec["flipped"] = got["flip"]["winner"] != main["winner"]
        if "dup" in got:
            rec["dup"] = got["dup"]
            rec["agree"] = got["dup"]["winner"] == main["winner"]
        out.append(rec)
    return out


def qc(records: list[dict]) -> dict:
    """Flip rate, agreement and the implied ceiling (spec §3.2, quality control)."""
    flips = [r["flipped"] for r in records if "flipped" in r]
    agree = [r["agree"] for r in records if "agree" in r]
    a = float(np.mean(agree)) if agree else float("nan")
    ceiling = (1 + math.sqrt(max(0.0, 2 * a - 1))) / 2 if agree else float("nan")
    strengths = np.array([r["strength"] for r in records]) if records else np.array([])
    return {
        "n_pairs": len(records),
        "n_flip_checked": len(flips),
        "flip_rate": float(np.mean(flips)) if flips else float("nan"),
        "n_dropped": int(sum(flips)),
        "n_dup_checked": len(agree),
        "agreement": a,
        "implied_ceiling": ceiling,
        "a_wins": (float(np.mean([r["winner"] == "A" for r in records])) if records
                   else float("nan")),
        "strength_hist": {str(s): int((strengths == s).sum()) for s in STRENGTHS},
    }


def write_round_data(records: list[dict], out_dir: Path, round_name: str) -> dict:
    """`data/critic/<round>.jsonl` (one pair per line) and `<round>-qc.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{round_name}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    summary = qc(records)
    (out_dir / f"{round_name}-qc.json").write_text(json.dumps(summary, indent=1, sort_keys=True)
                                                   + "\n", encoding="utf-8")
    return summary


def read_round_data(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
