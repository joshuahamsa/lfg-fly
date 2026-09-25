"""Day records (contract §body/records.py; spec §1 stamps, §4.1 step 6, §4.2)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date

import pytest

from lfg_fly import paths
from lfg_fly.body import records as R

STAMP = R.Stamp(network="testnet", lfg_api_base="http://localhost:8177", wallet="rFlyWallet")
LOOK_A = ("Blue", "None", "male", "Hoodie", "Grin", "Flat", "Laser", "Crown", "None")
LOOK_B = ("Blue", "None", "male", "Hoodie", "Grin", "Flat", "Monocle", "Pirate Hat", "None")


def make(date_="2026-09-25", state="pending", after=LOOK_B, stamp=STAMP, **kw) -> R.Record:
    fields = dict(
        date=date_,
        stamp=stamp,
        version="fly-v1",
        rarity_head="ab" * 32,
        seed=R.date_seed_hex(date_, "fly-v1"),
        hero="000800003C6D1B2A" + "00" * 24,
        before=list(LOOK_A),
        after=list(after),
        changes=[{"slot": "Eyes", "value": "Monocle"}, {"slot": "Head", "value": "Pirate Hat"}],
        candidates=[
            {"look": list(after), "changes": [{"slot": "Eyes", "value": "Monocle"}], "taste": 0.4,
             "rarity": 0.1, "cost": 0.0, "score": 0.43, "considered": True},
        ],
        considered=1,
        total=1,
        neuron_stats={"neurons_fired": 9812, "ms": 61},
        inputs_hash={"supply": "00" * 32, "economy": "11" * 32, "closet": "22" * 32},
        state=state,
    )
    fields.update(kw)
    return R.Record(**fields)


# ---------------------------------------------------------------- path_for


def test_path_for_is_the_date_under_the_network_records(_data_dir):
    expected = _data_dir / "testnet" / "records" / "2026-09-25.json"
    assert R.path_for("testnet", "2026-09-25") == expected
    assert R.path_for("mainnet", date(2026, 9, 25)) == paths.records_dir("mainnet") / expected.name
    for bad in ("2026-9-25", "20260925", "../x", "2026-09-25.json", "", "2026-13-40"):
        with pytest.raises(ValueError):
            R.path_for("testnet", bad)
    with pytest.raises(ValueError):
        R.path_for("devnet", "2026-09-25")


# ---------------------------------------------------------------- write / read


def test_write_then_read_round_trips(_data_dir):
    rec = make()
    p = R.write(rec)
    assert p == R.path_for("testnet", "2026-09-25") and p.exists()
    assert [q.name for q in p.parent.iterdir()] == ["2026-09-25.json"]  # no tmp left over
    back = R.read(p, STAMP)
    assert back == rec
    assert isinstance(back.stamp, R.Stamp)
    on_disk = json.loads(p.read_text())
    assert on_disk["stamp"] == {"network": "testnet", "lfg_api_base": "http://localhost:8177",
                                "wallet": "rFlyWallet"}
    assert on_disk["state"] == "pending" and on_disk["after"] == list(LOOK_B)


def test_write_overwrites_in_place_as_the_state_advances(_data_dir):
    p = R.write(make(state="pending"))
    R.write(make(state="submitted", equip_id="sid-1", submitted_at="2026-09-25T15:00:03+00:00"))
    R.write(make(state="done", equip_id="sid-1", resolution="committed",
                 submitted_at="2026-09-25T15:00:03+00:00"))
    back = R.read(p, STAMP)
    assert (back.state, back.equip_id, back.resolution) == ("done", "sid-1", "committed")
    assert back.submitted_at == "2026-09-25T15:00:03+00:00"
    assert [q.name for q in p.parent.iterdir()] == ["2026-09-25.json"]


def test_write_is_atomic_a_failed_write_leaves_the_old_record_intact(_data_dir):
    p = R.write(make(state="pending"))
    with pytest.raises(TypeError):
        R.write(make(state="submitted", neuron_stats={"bad": object()}))
    assert R.read(p, STAMP).state == "pending"
    assert [q.name for q in p.parent.iterdir()] == ["2026-09-25.json"]


def test_write_refuses_an_unknown_state_or_a_bad_date(_data_dir):
    with pytest.raises(ValueError, match="state"):
        R.write(make(state="maybe"))
    with pytest.raises(ValueError):
        R.write(make(date_="yesterday"))
    d = paths.records_dir("testnet")
    assert not d.exists() or not any(d.iterdir())


def test_states_and_terminality():
    assert R.STATES == ("dry_run", "pending", "submitted", "done", "failed", "UNKNOWN")
    assert set(R.NON_TERMINAL_STATES) == {"pending", "submitted", "UNKNOWN"}
    assert set(R.STATES) - set(R.NON_TERMINAL_STATES) == {"dry_run", "done", "failed"}


# ---------------------------------------------------------------- stamps


@pytest.mark.parametrize(
    "field,other",
    [
        ("network", "mainnet"),
        ("lfg_api_base", "http://localhost:8176"),
        ("wallet", "rSomeoneElse"),
    ],
)
def test_read_refuses_a_record_whose_stamp_differs_in_any_field(_data_dir, field, other):
    p = R.write(make())
    expect = dataclasses.replace(STAMP, **{field: other})
    with pytest.raises(R.StampMismatch) as exc:
        R.read(p, expect)
    assert field in str(exc.value)
    assert issubclass(R.StampMismatch, RuntimeError)


def test_read_refuses_a_record_from_the_other_network_written_in_our_dir(_data_dir):
    # the 2026-09-08 fixture-leak class: a mainnet artifact sitting in testnet's records/
    foreign = dataclasses.replace(STAMP, network="mainnet", lfg_api_base="http://localhost:8176")
    rec = make(stamp=foreign)
    p = R.path_for("testnet", rec.date)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dataclasses.asdict(rec)))
    with pytest.raises(R.StampMismatch):
        R.read(p, STAMP)


def test_read_tolerates_unknown_keys_but_not_missing_ones(_data_dir):
    p = R.write(make())
    d = json.loads(p.read_text())
    d["future_field"] = 1
    p.write_text(json.dumps(d))
    assert R.read(p, STAMP) == make()
    del d["hero"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hero"):
        R.read(p, STAMP)
    d["hero"] = "x"
    d["state"] = "whatever"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="state"):
        R.read(p, STAMP)
    del d["stamp"]
    p.write_text(json.dumps(d))
    with pytest.raises(R.StampMismatch):
        R.read(p, STAMP)


# ---------------------------------------------------------------- non_terminal


def test_non_terminal_returns_pending_submitted_unknown_oldest_first(_data_dir):
    R.write(make("2026-09-20", "done"))
    R.write(make("2026-09-21", "failed"))
    R.write(make("2026-09-22", "UNKNOWN"))
    R.write(make("2026-09-23", "dry_run"))
    R.write(make("2026-09-25", "pending"))
    R.write(make("2026-09-24", "submitted"))
    got = R.non_terminal("testnet", STAMP)
    assert [(r.date, r.state) for r in got] == [
        ("2026-09-22", "UNKNOWN"),
        ("2026-09-24", "submitted"),
        ("2026-09-25", "pending"),
    ]
    assert R.non_terminal("mainnet", STAMP) == []  # nothing there at all


def test_non_terminal_stops_on_a_foreign_record_even_a_terminal_one(_data_dir):
    R.write(make("2026-09-25", "pending"))
    foreign = dataclasses.replace(STAMP, wallet="rSomeoneElse")
    rec = make("2026-09-01", "done", stamp=foreign)
    R.path_for("testnet", rec.date).write_text(json.dumps(dataclasses.asdict(rec)))
    with pytest.raises(R.StampMismatch):
        R.non_terminal("testnet", STAMP)


def test_non_terminal_ignores_files_that_are_not_records(_data_dir):
    R.write(make("2026-09-25", "pending"))
    d = paths.records_dir("testnet")
    (d / "notes.txt").write_text("hi")
    (d / ".2026-09-26.json.tmp-123").write_text("{")
    (d / "2026-09-26.json.bak").write_text("{")
    assert [r.date for r in R.non_terminal("testnet", STAMP)] == ["2026-09-25"]


# ---------------------------------------------------------------- recent_looks (30-day tabu source)


def test_recent_looks_is_the_after_of_done_records_within_the_window(_data_dir):
    today = date(2026, 9, 25)
    look_old = ("Red",) + LOOK_A[1:]
    look_edge = ("Green",) + LOOK_A[1:]
    look_new = ("Gold",) + LOOK_A[1:]
    R.write(make("2026-08-25", "done", after=look_old))  # 31 days ago: out
    R.write(make("2026-08-26", "done", after=look_edge))  # exactly 30 days ago: in
    R.write(make("2026-09-20", "done", after=look_new))
    R.write(make("2026-09-21", "failed", after=("Pink",) + LOOK_A[1:]))  # never worn
    R.write(make("2026-09-22", "UNKNOWN", after=("Grey",) + LOOK_A[1:]))  # not known worn
    R.write(make("2026-09-23", "dry_run", after=("Teal",) + LOOK_A[1:]))  # never submitted
    R.write(make("2026-09-25", "done", after=LOOK_B))  # today counts
    got = R.recent_looks("testnet", STAMP, days=30, today=today)
    assert got == {look_edge, look_new, LOOK_B}
    assert all(isinstance(look, tuple) for look in got)
    assert R.recent_looks("testnet", STAMP, days=5, today=today) == {look_new, LOOK_B}
    assert R.recent_looks("testnet", STAMP, days=0, today=today) == {LOOK_B}
    assert R.recent_looks("mainnet", STAMP, days=30, today=today) == set()


def test_recent_looks_refuses_a_foreign_record_in_the_window(_data_dir):
    today = date(2026, 9, 25)
    foreign = dataclasses.replace(STAMP, network="mainnet")
    rec = make("2026-09-24", "done", stamp=foreign)
    R.path_for("testnet", rec.date).parent.mkdir(parents=True)
    R.path_for("testnet", rec.date).write_text(json.dumps(dataclasses.asdict(rec)))
    with pytest.raises(R.StampMismatch):
        R.recent_looks("testnet", STAMP, days=30, today=today)


# ---------------------------------------------------------------- date_seed


def test_date_seed_is_sha256_of_date_bar_version():
    digest = hashlib.sha256(b"2026-09-25|fly-v1").digest()
    assert R.date_seed("2026-09-25", "fly-v1") == int.from_bytes(digest[:8], "little")
    assert R.date_seed("2026-09-25", "fly-v1") == R.date_seed("2026-09-25", "fly-v1")
    assert R.date_seed("2026-09-26", "fly-v1") != R.date_seed("2026-09-25", "fly-v1")
    assert R.date_seed("2026-09-25", "fly-v2") != R.date_seed("2026-09-25", "fly-v1")
    assert 0 <= R.date_seed("2026-09-25", "fly-v1") < 2**64
    assert R.date_seed_hex("2026-09-25", "fly-v1") == digest.hex()
    assert R.date_seed(date(2026, 9, 25), "fly-v1") == R.date_seed("2026-09-25", "fly-v1")


def test_date_seed_drives_numpy_reproducibly():
    import numpy as np

    s = R.date_seed("2026-09-25", "fly-v1")
    a = np.random.default_rng(s).random(4)
    b = np.random.default_rng(R.date_seed("2026-09-25", "fly-v1")).random(4)
    assert np.array_equal(a, b)
