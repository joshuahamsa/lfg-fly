import io
import json

import numpy as np
from PIL import Image

from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher import catalog as K
from lfg_fly.teacher import critic as C
from lfg_fly.teacher import render as R

CFG = {"layers": [{"name": s, "z": 10 * (i + 1)} for i, s in enumerate(SLOTS)],
       "z_overrides": []}


def _png(rgba, size=8):
    buf = io.BytesIO()
    Image.new("RGBA", (size, size), rgba).save(buf, format="PNG")
    return buf.getvalue()


def _catalog(tmp_path):
    values = {s: [] for s in SLOTS}
    for slot in SLOTS:
        for j, rgba in enumerate([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 128)]):
            value = f"{slot}{j}"
            path = K.layer_cache_path(tmp_path, "male", slot, value)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_png(rgba))
            values[slot].append(value)
        if slot != "Body":
            values[slot].append("None")
    return K.Catalog(body="male", values=values, odds={}, api="http://lfg")


def test_schedule_keeps_second_showings_away_from_the_first():
    items, batches = C.schedule(300, seed=5, batch=40)
    kinds = {k: sum(1 for it in items if it.kind == k) for k in ("main", "flip", "dup")}
    assert kinds == {"main": 300, "flip": 30, "dup": 30}
    by_id = {it.item_id: it for it in items}
    assert all(len(b) <= 40 for b in batches)
    assert sorted(x for b in batches for x in b) == sorted(by_id)
    for b in batches:
        pairs = [by_id[x].pair_id for x in b]
        assert len(set(pairs)) == len(pairs), "an agent would see a pair twice"
    flip = next(it for it in items if it.kind == "flip")
    assert flip.swapped and flip.image.endswith("f.png")
    dup = next(it for it in items if it.kind == "dup")
    assert not dup.swapped and dup.image == f"p{dup.pair_id:04d}.png"


def test_schedule_is_deterministic():
    a = C.schedule(120, seed=1)
    b = C.schedule(120, seed=1)
    assert a == b
    assert C.schedule(120, seed=2)[1] != a[1]


def test_card_labels_left_a_and_right_b(tmp_path):
    left = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    right = Image.new("RGBA", (64, 64), (0, 0, 255, 255))
    im = C.card(left, right)
    assert im.size == (128, 64)
    px = np.asarray(im)
    assert tuple(px[40, 10]) == (255, 0, 0) and tuple(px[40, 118]) == (0, 0, 255)
    assert tuple(px[32, 63]) == (255, 255, 255)  # the divider


def test_write_round_renders_cards_and_batches(tmp_path):
    cat = _catalog(tmp_path)
    out = tmp_path / "r"
    m = C.write_round(cat, tmp_path, R.zorder_from_config(CFG), out, "rt", n_pairs=12, seed=3,
                      batch=4)
    assert m["n_pairs"] == 12 and len(m["pairs"]) == 12
    n_flip = sum(1 for it in m["items"] if it["kind"] == "flip")
    assert n_flip == 1 and sum(1 for it in m["items"] if it["kind"] == "dup") == 1
    cards = sorted(p.name for p in out.glob("*.png"))
    assert len(cards) == 12 + n_flip
    with Image.open(out / "p0000.png") as im:
        assert im.size == C.CARD_SIZE
    batches = sorted((out / "batches").glob("*.json"))
    assert len(batches) == len(m["batches"])
    b0 = json.loads(batches[0].read_text())
    assert b0["items"][0]["image"].endswith(".png") and b0["round"] == "rt"
    # resumable: a second call renders nothing new
    stamp = {p: p.stat().st_mtime_ns for p in out.glob("*.png")}
    C.write_round(cat, tmp_path, R.zorder_from_config(CFG), out, "rt", n_pairs=12, seed=3,
                  batch=4)
    assert {p: p.stat().st_mtime_ns for p in out.glob("*.png")} == stamp


def _manifest_and_results(tmp_path, n=20):
    cat = _catalog(tmp_path)
    out = tmp_path / "r"
    m = C.write_round(cat, tmp_path, R.zorder_from_config(CFG), out, "rt", n_pairs=n, seed=3,
                      batch=5)
    results = out / "results"
    for name, ids in m["batches"].items():
        rows = [{"item_id": x, "winner": "A", "strength": 2, "idea_A": "a", "idea_B": "b",
                 "reason": "r"} for x in ids]
        (results / f"{name}.json").write_text(json.dumps({"verdicts": rows}))
    return m, out


def test_read_results_flags_missing_and_incomplete_batches(tmp_path):
    m, out = _manifest_and_results(tmp_path)
    names = sorted(m["batches"])
    (out / "results" / f"{names[0]}.json").unlink()
    rows = json.loads((out / "results" / f"{names[1]}.json").read_text())["verdicts"]
    rows[0]["winner"] = "C"
    (out / "results" / f"{names[1]}.json").write_text(json.dumps({"verdicts": rows}))
    verdicts, bad = C.read_results(m, out / "results")
    assert sorted(bad) == sorted(names[:2])
    assert len(verdicts) == len(m["items"]) - len(m["batches"][names[0]]) - 1


def test_assemble_maps_flips_to_canonical_orientation_and_scores_qc(tmp_path):
    m, out = _manifest_and_results(tmp_path)
    verdicts, bad = C.read_results(m, out / "results")
    assert bad == []
    # every verdict said "A" (the left side), so the flipped showing disagrees with the main
    recs = C.assemble(m, verdicts)
    assert len(recs) == m["n_pairs"]
    flip_rec = next(r for r in recs if "flip" in r)
    assert flip_rec["winner"] == "A" and flip_rec["flip"]["winner"] == "B"
    assert flip_rec["flipped"] is True
    dup_rec = next(r for r in recs if "dup" in r)
    assert dup_rec["agree"] is True
    q = C.qc(recs)
    assert q["flip_rate"] == 1.0 and q["agreement"] == 1.0 and q["implied_ceiling"] == 1.0
    assert q["n_dropped"] == sum(1 for it in m["items"] if it["kind"] == "flip") == 2
    summary = C.write_round_data(recs, tmp_path / "data", "rt")
    assert summary == q
    back = C.read_round_data(tmp_path / "data" / "rt.jsonl")
    assert back == json.loads(json.dumps(recs, sort_keys=True))
