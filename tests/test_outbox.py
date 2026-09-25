"""The outbox (contract §body/outbox.py; spec §1 alerts, §4.2, §4.5 posts without X credentials)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from lfg_fly import paths
from lfg_fly.body import outbox

WHEN = datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc)


def test_write_creates_a_utc_timestamped_file(_data_dir):
    p = outbox.write("testnet", "alert", {"msg": "UNKNOWN record 2026-09-24"}, when=WHEN)
    assert p.parent == paths.outbox_dir("testnet") == _data_dir / "testnet" / "outbox"
    assert p.name == "20260925T150000.000000Z-alert.json"
    d = json.loads(p.read_text())
    assert d == {
        "kind": "alert",
        "network": "testnet",
        "when": "2026-09-25T15:00:00+00:00",
        "payload": {"msg": "UNKNOWN record 2026-09-24"},
    }
    assert [q.name for q in p.parent.iterdir()] == [p.name]  # no tmp file left behind


def test_write_defaults_to_now_utc(_data_dir):
    before = datetime.now(timezone.utc)
    p = outbox.write("mainnet", "post", {"text": "Day 1."})
    when = datetime.fromisoformat(json.loads(p.read_text())["when"])
    assert when.tzinfo is not None
    slack = timedelta(seconds=1)
    assert before - slack <= when <= datetime.now(timezone.utc) + slack
    assert p.name.endswith("Z-post.json")


def test_write_normalises_the_timestamp_to_utc(_data_dir):
    plus_two = timezone(timedelta(hours=2))
    p = outbox.write("testnet", "alert", {}, when=WHEN.astimezone(plus_two))
    assert p.name == "20260925T150000.000000Z-alert.json"
    # a naive datetime is taken as UTC (the fly's clock is UTC everywhere)
    q = outbox.write("testnet", "alert", {}, when=WHEN.replace(tzinfo=None, second=1))
    assert q.name == "20260925T150001.000000Z-alert.json"


def test_same_instant_twice_never_overwrites(_data_dir):
    p1 = outbox.write("testnet", "alert", {"n": 1}, when=WHEN)
    p2 = outbox.write("testnet", "alert", {"n": 2}, when=WHEN)
    assert p1 != p2 and p1.exists() and p2.exists()
    assert json.loads(p1.read_text())["payload"] == {"n": 1}
    assert json.loads(p2.read_text())["payload"] == {"n": 2}
    assert p2.name.endswith("-alert.json")


def test_write_refuses_unknown_kind_or_network(_data_dir):
    with pytest.raises(ValueError, match="kind"):
        outbox.write("testnet", "tweet", {}, when=WHEN)
    with pytest.raises(ValueError):
        outbox.write("devnet", "alert", {}, when=WHEN)
    assert not (_data_dir / "testnet" / "outbox").exists() or not any(
        (_data_dir / "testnet" / "outbox").iterdir()
    )


def test_unserialisable_payload_leaves_nothing_behind(_data_dir):
    with pytest.raises(TypeError):
        outbox.write("testnet", "alert", {"bad": object()}, when=WHEN)
    d = paths.outbox_dir("testnet")
    assert not d.exists() or list(d.iterdir()) == []


def test_alerts_lists_uncleared_alerts_oldest_first(_data_dir):
    later = outbox.write("testnet", "alert", {"n": 2}, when=WHEN + timedelta(minutes=1))
    outbox.write("testnet", "post", {"text": "not an alert"}, when=WHEN)
    earlier = outbox.write("testnet", "alert", {"n": 1}, when=WHEN)
    outbox.write("mainnet", "alert", {"n": 99}, when=WHEN)  # other network

    got = outbox.alerts("testnet")
    assert [a["payload"] for a in got] == [{"n": 1}, {"n": 2}]
    assert [a["path"] for a in got] == [earlier, later]
    assert all(a["kind"] == "alert" and a["network"] == "testnet" for a in got)

    outbox.clear("testnet", earlier)
    assert [a["path"] for a in outbox.alerts("testnet")] == [later]
    assert not earlier.exists()
    cleared = earlier.with_name(earlier.name + ".cleared")
    assert cleared.exists() and json.loads(cleared.read_text())["payload"] == {"n": 1}

    outbox.clear("testnet", str(later))  # a str path works too
    assert outbox.alerts("testnet") == []


def test_alerts_is_empty_without_an_outbox(_data_dir):
    assert outbox.alerts("testnet") == []
    assert outbox.alerts("mainnet") == []


def test_alerts_surfaces_a_corrupt_file_instead_of_hiding_it(_data_dir):
    good = outbox.write("testnet", "alert", {"n": 1}, when=WHEN)
    bad = good.with_name("20260925T150001.000000Z-alert.json")
    bad.write_text("{not json")
    got = outbox.alerts("testnet")
    assert [a["path"] for a in got] == [good, bad]
    assert got[1]["kind"] == "alert" and got[1]["payload"] is None and "error" in got[1]


def test_clear_refuses_a_path_outside_the_network_outbox(_data_dir, tmp_path):
    p = outbox.write("mainnet", "alert", {"n": 1}, when=WHEN)
    with pytest.raises(ValueError):
        outbox.clear("testnet", p)  # mainnet's alert, testnet's outbox
    stray = tmp_path / "elsewhere-alert.json"
    stray.write_text("{}")
    with pytest.raises(ValueError):
        outbox.clear("mainnet", stray)
    assert p.exists() and stray.exists()
    with pytest.raises(FileNotFoundError):
        outbox.clear("mainnet", paths.outbox_dir("mainnet") / "missing-alert.json")
