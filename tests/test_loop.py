"""The move loop against the fake LFG (contract §body/loop.py; spec §4.1, §4.2, §4.4, §7).

The brain is a stand-in with `taste` / `rarity` callables, the ledger a stand-in
that passes the identity check and serves the hero's URI, the signer a real
xrpl Wallet signing a master-key proof (``fake_lfg.ProofSigner``). No pytest-asyncio
in the venv: every coroutine runs under ``asyncio.run``.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import fake_lfg as FL
import numpy as np
import pytest
from PIL import Image

from lfg_fly import paths
from lfg_fly.body import loop as L
from lfg_fly.body import outbox
from lfg_fly.body import records as R
from lfg_fly.body.client import LfgError
from lfg_fly.body.config import FlyConfig
from lfg_fly.brain.readout import RarityHead
from lfg_fly.brain.senses import SLOTS
from lfg_fly.teacher.render import zorder_from_config

fake_lfg = FL.fake_lfg  # the shared fixture, bound by name
HERO = "A" * 64
TODAY = date(2026, 9, 25)
NOW = datetime(2026, 9, 25, 15, 0, 0, tzinfo=timezone.utc)
CROWN = ("Blue", "None", "male", "Hoodie", "None", "None", "Laser", "Crown", "None")
PIRATE = ("Blue", "None", "male", "Hoodie", "None", "None", "Laser", "Pirate Hat", "None")
PIRATE_MONOCLE = ("Blue", "None", "male", "Hoodie", "None", "None", "Monocle", "Pirate Hat", "None")
HEAD, EYES = SLOTS.index("Head"), SLOTS.index("Eyes")


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ doubles


@dataclass
class FakeHead:
    taste_sd: float = 1.0


@dataclass
class FakeCatalog:
    values: dict = field(default_factory=lambda: {
        "Background": ["Blue", "Red"], "Back": ["None"], "Body": ["male"],
        "Clothing": ["Hoodie"], "Mouth": ["None"], "Eyebrows": ["None"],
        "Eyes": ["Laser", "Monocle"], "Head": ["Crown", "Pirate Hat", "Beanie"],
        "Accessory": ["None"],
    })


class FakeBrain:
    """taste: Pirate Hat +2, Monocle +1, Beanie -5; rarity: 0 unless a head is given."""

    def __init__(self, z_order: dict | None = None):
        self.head = FakeHead()
        self.catalog = FakeCatalog()
        self.stats = {"neurons_fired": 0, "ms": 0.0}
        self.taste_calls: list[list[tuple]] = []
        self.rarity_calls: list[tuple] = []
        self._z = z_order
        self.manifest = argparse.Namespace(version="fly-v1")

    def taste(self, looks):
        looks = [tuple(lk) for lk in looks]
        self.taste_calls.append(looks)
        out = np.zeros(len(looks))
        for i, lk in enumerate(looks):
            out[i] += 2.0 * (lk[HEAD] == "Pirate Hat") + 1.0 * (lk[EYES] == "Monocle")
            out[i] -= 5.0 * (lk[HEAD] == "Beanie")
        self.stats = {"neurons_fired": 9812, "ms": 61.0, "n_looks": len(looks)}
        return out

    def rarity(self, looks, conc, head):
        self.rarity_calls.append((len(looks), np.asarray(conc).shape, head))
        return np.full(len(looks), float(head.b))

    def pinned_zorder(self):
        cfg = self._z or FL.FakeLfgState().z_order
        layers = [{"name": s, "z": z} for s, z in cfg["layers"].items()]
        return zorder_from_config({"layers": layers, "z_overrides": cfg.get("z_overrides", [])})


class FakeLedger:
    def __init__(self, uri: str | None = None, index: int = 1000):
        self.uri, self.index = uri, index
        self.identity_calls: list[str | None] = []
        self.uri_calls: list[tuple[str, str]] = []

    async def identity_check(self, expected_hash):
        self.identity_calls.append(expected_hash)
        return argparse.Namespace(network_id=1, ledger_32570=expected_hash, endpoint="fake")

    async def validated_ledger_index(self) -> int:
        return self.index

    async def nft_uri(self, nft_id: str, owner: str):
        self.uri_calls.append((nft_id, owner))
        return self.uri


def attrs(look):
    return [{"trait_type": s, "value": v} for s, v in zip(SLOTS, look, strict=True)]


def metadata_of(look):
    async def fetch(uri: str) -> dict:
        return {"attributes": attrs(look), "image": "ipfs://img"}

    return fetch


async def fake_image(url: str) -> Image.Image:
    return Image.new("RGB", (64, 64), (10, 200, 10))


def cfg_for(base: str, **over) -> FlyConfig:
    kw = dict(network="testnet", api_base=base, enabled=True, temperature=0.001,
              expected_ledger_hash="18" * 32)
    kw.update(over)
    return FlyConfig(**kw)


@pytest.fixture
def no_x_env(monkeypatch):
    for k in ("FLY_X_API_KEY", "FLY_X_API_SECRET", "FLY_X_ACCESS_TOKEN", "FLY_X_ACCESS_SECRET",
              "FLY_HERO"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def world(fake_lfg, no_x_env):
    """A fake LFG whose Closet holds a Pirate Hat, a Monocle and a Beanie."""
    base, state = fake_lfg
    state.add_closet("Head", "Pirate Hat")
    state.add_closet("Eyes", "Monocle")
    state.add_closet("Head", "Beanie")
    signer = FL.ProofSigner()
    return argparse.Namespace(base=base, state=state, signer=signer, wallet=signer.account,
                              brain=FakeBrain(), ledger=FakeLedger())


def do_move(w, *, dry_run=False, cfg=None, today=TODAY, now=NOW, **kw):
    cfg = cfg or cfg_for(w.base)
    return run(L.move(cfg, w.brain, dry_run=dry_run, today=today, ledger=w.ledger,
                      signer=w.signer, now=lambda: now, fetch_image=fake_image,
                      sleep=_no_sleep, **kw))


async def _no_sleep(seconds: float) -> None:
    _no_sleep.slept.append(seconds)


_no_sleep.slept = []


def posts(state) -> list[str]:
    return [p for m, p in state.requests if m == "POST"]


def stamp_for(w, base=None) -> R.Stamp:
    return R.Stamp(network="testnet", lfg_api_base=(base or w.base).rstrip("/"), wallet=w.wallet)


def files_under(root) -> list:
    return [p for p in paths.data_dir().rglob("*") if p.is_file()]


# ------------------------------------------------------------------ gates (§4.4)


def test_refuses_when_not_enabled_and_not_dry_run(world):
    w = world
    with pytest.raises(L.LoopRefused, match="FLY_ENABLED"):
        do_move(w, cfg=cfg_for(w.base, enabled=False))
    assert w.state.requests == []  # nothing was asked of LFG
    assert w.ledger.identity_calls == []
    assert not paths.records_dir("testnet").exists()


def test_dry_run_runs_without_fly_enabled(world):
    w = world
    rec = do_move(w, dry_run=True, cfg=cfg_for(w.base, enabled=False))
    assert rec.state == "dry_run"


def test_single_flight_lock_refuses_a_second_runner(world):
    w = world
    lock = L.lock_path("testnet")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(L.LoopRefused, match="lock"):
            do_move(w, dry_run=True)
    assert w.state.requests == []
    # released: the next run goes through
    assert do_move(w, dry_run=True).state == "dry_run"


def test_identity_check_runs_before_sign_in_and_a_failure_stops_everything(world):
    w = world

    class Refusing(FakeLedger):
        async def identity_check(self, expected_hash):
            raise RuntimeError("wrong chain")

    w.ledger = Refusing()
    with pytest.raises(RuntimeError, match="wrong chain"):
        do_move(w, dry_run=True)
    assert w.state.requests == []


# ------------------------------------------------------------------ dry run (§4.1 step 7)


def test_dry_run_writes_a_record_and_never_equips(world):
    w = world
    rec = do_move(w, dry_run=True)
    assert rec.state == "dry_run"
    assert rec.stamp == stamp_for(w)
    assert rec.hero == HERO
    assert tuple(rec.before) == CROWN
    # greedy: Pirate Hat (+2) then Monocle (+1), then stay wins
    assert tuple(rec.after) == PIRATE_MONOCLE
    assert rec.changes == [{"slot": "Eyes", "value": "Monocle"},
                           {"slot": "Head", "value": "Pirate Hat"}]
    assert rec.seed == R.date_seed_hex(TODAY, "fly-v1")
    assert rec.version == "fly-v1"
    assert rec.considered == rec.total == len(rec.candidates) > 0
    assert all(c["considered"] for c in rec.candidates)
    assert rec.neuron_stats["neurons_fired"] == 9812 and rec.neuron_stats["ms"] > 0
    assert set(rec.inputs_hash) == {"supply", "economy", "closet"}
    assert all(len(h) == 64 for h in rec.inputs_hash.values())
    assert rec.equip_id is None and rec.post is None
    # on disk, as written
    on_disk = R.read(R.path_for("testnet", TODAY), stamp_for(w))
    assert on_disk.state == "dry_run" and on_disk.after == rec.after
    # never equipped, never posted; the hero is untouched
    assert "/api/equip" not in posts(w.state)
    assert w.state.look(HERO)["Head"] == "Crown"
    assert not paths.outbox_dir("testnet").exists()
    # the loop talked to LFG in the spec's order
    gets = [p for m, p in w.state.requests if m == "GET"]
    assert gets.index("/api/nfts") < gets.index("/api/economy") < gets.index("/api/rarity/supply")


def test_dry_run_logs_out_and_keeps_the_token_off_disk(world):
    w = world
    do_move(w, dry_run=True)
    assert w.state.logouts == 1
    assert w.state.tokens == {}  # the fake forgot it on logout
    tokens = [s["token"] for s in w.state.sessions] if w.state.sessions and \
        "token" in w.state.sessions[0] else []
    for f in files_under(paths.data_dir()):
        text = f.read_text(errors="ignore")
        for tok in tokens:
            assert tok not in text
        assert "Bearer" not in text
        assert "session_token" not in text


def test_the_brain_saw_every_candidate_in_one_batch_per_step(world):
    w = world
    rec = do_move(w, dry_run=True)
    # step 1: stay + Pirate Hat + Beanie + Monocle = 4; step 2: stay + Beanie + Monocle
    # (Crown is back in the closet after the fake's... no: the dry run never equips, so the
    # closet is unchanged) -> stay, Beanie, Monocle, and Crown? Crown is not in the closet.
    assert [len(b) for b in w.brain.taste_calls] == [4, 3, 2]
    assert rec.total == 4 + 3 + 2


# ------------------------------------------------------------------ the real move (§4.1 step 7–9)


def test_done_equip_is_recorded_done_and_posted_to_the_outbox(world):
    w = world
    rec = do_move(w)
    assert rec.state == "done"
    assert rec.resolution == "committed"
    assert rec.equip_id and rec.equip_id.startswith("eq-")
    assert rec.submitted_at is not None
    assert rec.equip_status["state"] == "done"
    # one POST /api/equip with all changes, slot-diffed
    assert posts(w.state).count("/api/equip") == 1
    assert w.state.look(HERO)["Head"] == "Pirate Hat"
    assert w.state.look(HERO)["Eyes"] == "Monocle"
    # posted to the outbox (no X credentials), card beside it
    assert rec.post["channel"] == "outbox" and rec.post["reason"] == "no_credentials"
    assert os.path.exists(rec.post["path"]) and os.path.exists(rec.post["card"])
    payload = json.loads(open(rec.post["path"]).read())["payload"]
    assert payload["after"] == list(PIRATE_MONOCLE) and payload["day"] == 1
    assert "Day 1" in rec.post["text"] and "Pirate Hat" in rec.post["text"]
    with Image.open(rec.post["card"]) as card:
        assert card.size == (1200, 675)
    # the record on disk is the final one
    on_disk = R.read(R.path_for("testnet", TODAY), stamp_for(w))
    assert on_disk.state == "done" and on_disk.post == rec.post
    assert w.state.logouts == 1
    assert outbox.alerts("testnet") == []


def test_record_is_written_pending_before_the_equip_is_posted(world, monkeypatch):
    w = world
    seen: list[tuple[str, str | None]] = []
    real_equip = L.LfgClient.equip

    async def spy(self, nft_id, changes):
        rec = R.read(R.path_for("testnet", TODAY), stamp_for(w))
        seen.append((rec.state, rec.submitted_at))
        return await real_equip(self, nft_id, changes)

    monkeypatch.setattr(L.LfgClient, "equip", spy)
    do_move(w)
    assert seen == [("pending", NOW.isoformat())]


def test_failed_reverted_is_recorded_failed_and_not_posted(world):
    w = world
    w.state.equip_outcome = "failed_reverted"
    rec = do_move(w)
    assert rec.state == "failed" and rec.resolution == "reverted"
    assert rec.error == "modify failed"
    assert rec.post is None
    assert not list(paths.outbox_dir("testnet").glob("*-post.json")) \
        if paths.outbox_dir("testnet").exists() else True
    assert w.state.look(HERO)["Head"] == "Crown"
    assert w.state.logouts == 1


def test_failed_uncertain_is_unknown_with_an_alert_and_the_next_run_refuses(world):
    w = world
    w.state.equip_outcome = "failed_uncertain"
    rec = do_move(w)
    assert rec.state == "UNKNOWN" and rec.resolution == "uncertain"
    alerts = outbox.alerts("testnet")
    assert len(alerts) == 1 and alerts[0]["payload"]["kind"] == "unknown_move"
    assert alerts[0]["payload"]["date"] == "2026-09-25"
    assert w.state.logouts == 1

    # the next run: too early to resolve (< 10 min) -> waits; still UNKNOWN -> stops, no move
    w.state.equip_outcome = "done"
    w.ledger = FakeLedger(uri="ipfs://QmHero")
    fetch = metadata_of(("Red",) + CROWN[1:])  # neither the intended nor the original look
    with pytest.raises(L.LoopStopped, match="UNKNOWN"):
        do_move(w, today=TODAY + timedelta(days=1), now=NOW + timedelta(minutes=2),
                fetch_metadata=fetch)
    assert posts(w.state).count("/api/equip") == 1  # no second equip, ever
    assert w.ledger.uri_calls == [(HERO, w.wallet)]
    assert len(outbox.alerts("testnet")) == 2
    assert R.read(R.path_for("testnet", TODAY), stamp_for(w)).state == "UNKNOWN"
    assert w.state.logouts == 2  # logged out on the way out


def test_unknown_resolves_from_the_ledger_never_lfg(world):
    w = world
    w.state.equip_outcome = "failed_uncertain"
    do_move(w)
    n_before = len(w.state.requests)
    # the ledger says the hero wears the intended look -> done; then the next day moves on
    w.state.equip_outcome = "done"
    w.ledger = FakeLedger(uri="ipfs://QmHero")
    rec = do_move(w, today=TODAY + timedelta(days=1), now=NOW + timedelta(days=1),
                  fetch_metadata=metadata_of(PIRATE_MONOCLE))
    yesterday = R.read(R.path_for("testnet", TODAY), stamp_for(w))
    assert yesterday.state == "done" and yesterday.resolution == "uncertain"
    assert "ledger" in (yesterday.error or "").lower() or yesterday.equip_status is not None
    assert rec.date == "2026-09-26" and rec.state == "done"
    # LFG was never asked about yesterday's session again (§4.2: the ledger, not the index);
    # the only /api/equip/<id> polls of the second run belong to the second day's own equip
    equip_gets = [p for m, p in w.state.requests[n_before:] if m == "GET" and "/api/equip/" in p]
    assert equip_gets and all(p.endswith(rec.equip_id) for p in equip_gets)
    assert not any(p.endswith(yesterday.equip_id) for p in equip_gets)


def test_unknown_resolving_to_failed_leaves_the_day_alone(world):
    w = world
    w.state.equip_outcome = "failed_uncertain"
    do_move(w)
    w.ledger = FakeLedger(uri="ipfs://QmHero")
    with pytest.raises(L.LoopRefused, match="already"):
        do_move(w, today=TODAY, now=NOW + timedelta(hours=1), fetch_metadata=metadata_of(CROWN))
    assert R.read(R.path_for("testnet", TODAY), stamp_for(w)).state == "failed"
    assert posts(w.state).count("/api/equip") == 1


def test_one_move_per_day(world):
    w = world
    do_move(w)
    with pytest.raises(L.LoopRefused, match="already"):
        do_move(w)
    assert posts(w.state).count("/api/equip") == 1
    # a dry run after the move is refused too: the day's record is final
    with pytest.raises(L.LoopRefused, match="already"):
        do_move(w, dry_run=True)


def test_a_failed_day_is_never_resubmitted(world):
    w = world
    w.state.equip_outcome = "failed_reverted"
    do_move(w)
    w.state.equip_outcome = "done"
    with pytest.raises(L.LoopRefused, match="already"):
        do_move(w)
    assert posts(w.state).count("/api/equip") == 1


def test_logout_even_when_the_equip_raises(world):
    w = world
    # another economy action is running: POST /api/equip answers 409
    w.state.equip_sessions["eq-other"] = {"id": "eq-other", "kind": "equip", "state": "running",
                                          "error": None, "displaced": [], "resolution": None,
                                          "platform": "web", "_polls": 99, "_outcome": "hang",
                                          "_nft_id": HERO, "_changes": []}
    with pytest.raises(LfgError) as info:
        do_move(w)
    assert info.value.status == 409
    assert w.state.logouts == 1 and w.state.tokens == {}
    rec = R.read(R.path_for("testnet", TODAY), stamp_for(w))
    assert rec.state == "failed" and "409" in (rec.error or "")


def test_a_lost_equip_answer_is_unknown(world, monkeypatch):
    w = world

    async def boom(self, nft_id, changes):
        raise OSError("connection reset")

    monkeypatch.setattr(L.LfgClient, "equip", boom)
    with pytest.raises(OSError):
        do_move(w)
    rec = R.read(R.path_for("testnet", TODAY), stamp_for(w))
    assert rec.state == "UNKNOWN"
    assert len(outbox.alerts("testnet")) == 1
    assert w.state.logouts == 1


def test_a_hanging_equip_session_is_unknown(world):
    w = world
    w.state.equip_outcome = "hang"
    rec = do_move(w, equip_timeout=0.05, equip_poll=0.01)
    assert rec.state == "UNKNOWN" and rec.equip_id
    assert len(outbox.alerts("testnet")) == 1
    assert w.state.logouts == 1


def test_a_submitted_record_whose_session_finished_is_reconciled_from_lfg(world):
    """A crash after POST /api/equip: the record says `submitted`; the next run asks
    /api/equip/{id} first (§4.2 table) and does not move again that day."""
    w = world
    rec = do_move(w, dry_run=True)
    rec.state, rec.equip_id, rec.submitted_at = "submitted", None, NOW.isoformat()
    # start a real equip session on the fake for the same changes, as the crashed run did
    sid = run(_start_equip(w, rec))
    rec.equip_id = sid
    R.write(rec)
    with pytest.raises(L.LoopRefused, match="already"):
        do_move(w, now=NOW + timedelta(minutes=1))
    fixed = R.read(R.path_for("testnet", TODAY), stamp_for(w))
    assert fixed.state == "done" and fixed.equip_status["id"] == sid
    assert posts(w.state).count("/api/equip") == 1


async def _start_equip(w, rec) -> str:
    async with L.LfgClient(w.base) as client:
        await client.sign_in(w.signer, w.ledger)
        r = await client.equip(rec.hero, rec.changes)
        return r["id"]


def test_a_submitted_record_whose_session_is_gone_goes_to_the_ledger(world):
    w = world
    rec = do_move(w, dry_run=True)
    rec.state, rec.equip_id, rec.submitted_at = "submitted", "eq-gone", NOW.isoformat()
    R.write(rec)
    w.ledger = FakeLedger(uri="ipfs://QmHero")
    with pytest.raises(L.LoopRefused, match="already"):
        do_move(w, now=NOW + timedelta(minutes=30), fetch_metadata=metadata_of(PIRATE_MONOCLE))
    assert R.read(R.path_for("testnet", TODAY), stamp_for(w)).state == "done"
    assert w.ledger.uri_calls == [(HERO, w.wallet)]


def test_a_pending_record_from_a_crashed_run_is_resolved_before_moving(world):
    w = world
    rec = do_move(w, dry_run=True)
    rec.state, rec.submitted_at = "pending", None  # crashed before the submit stamp
    R.write(rec)
    w.ledger = FakeLedger(uri="ipfs://QmHero")
    # the ledger shows the original look: nothing happened -> failed; the day is spent
    with pytest.raises(L.LoopRefused, match="already"):
        do_move(w, fetch_metadata=metadata_of(CROWN))
    assert R.read(R.path_for("testnet", TODAY), stamp_for(w)).state == "failed"


def test_reconcile_waits_out_the_ten_minutes_then_resolves(world):
    w = world
    w.state.equip_outcome = "failed_uncertain"
    do_move(w)
    w.state.equip_outcome = "done"
    w.ledger = FakeLedger(uri="ipfs://QmHero")
    _no_sleep.slept.clear()
    rec = do_move(w, today=TODAY + timedelta(days=1), now=NOW + timedelta(minutes=3),
                  fetch_metadata=metadata_of(PIRATE_MONOCLE))
    assert rec.state == "done"
    assert _no_sleep.slept and abs(_no_sleep.slept[0] - 7 * 60) < 1e-6


# ------------------------------------------------------------------ stamps (§1)


def test_a_foreign_record_stops_the_loop_with_an_alert(world):
    w = world
    rec = do_move(w, dry_run=True)
    rec.stamp = R.Stamp(network="testnet", lfg_api_base=w.base, wallet="rSomeoneElse")
    rec.date = "2026-09-20"
    R.write(rec)
    with pytest.raises(L.LoopStopped, match="stamp"):
        do_move(w)
    alerts = outbox.alerts("testnet")
    assert len(alerts) == 1 and alerts[0]["payload"]["kind"] == "stamp_mismatch"
    assert "/api/equip" not in posts(w.state)
    assert w.state.logouts == 2  # the dry run above, and the stopped run on its way out


# ------------------------------------------------------------------ state reads (§4.1 step 3)


def test_z_order_mismatch_refuses(world):
    w = world
    w.state.z_order = {"layers": {s: 10 + i for i, s in enumerate(SLOTS)}, "z_overrides": []}
    with pytest.raises(L.LoopRefused, match="z_order"):
        do_move(w, dry_run=True)
    assert not R.path_for("testnet", TODAY).exists()
    assert w.state.logouts == 1


def test_z_order_override_mismatch_refuses_and_a_matching_one_passes(world):
    w = world
    w.state.z_order["z_overrides"] = [{"trait_type": "Head", "value": "Crown", "z": 99}]
    with pytest.raises(L.LoopRefused, match="z_order"):
        do_move(w, dry_run=True)
    w.brain = FakeBrain(z_order=w.state.z_order)
    assert do_move(w, dry_run=True).state == "dry_run"


def test_economy_disabled_surfaces_as_lfgerror_after_logout(world):
    w = world
    w.state.economy_enabled = False
    with pytest.raises(LfgError) as info:
        do_move(w, dry_run=True)
    assert info.value.code == "economy_disabled"
    assert w.state.logouts == 1


# ------------------------------------------------------------------ the hero (§4.1 step 3)


def test_hero_is_the_first_mutable_non_blank_male(world):
    w = world
    w.state.characters.clear()
    w.state.add_character("B" * 64, body="ape", traits={"Head": "Crown"})
    w.state.add_character("C" * 64, body="male", traits={"Head": "Crown"}, mutable=False)
    w.state.add_character("D" * 64, body="male", blank=True)
    w.state.add_character("E" * 64, body="male", traits={"Head": "Crown"})
    w.state.add_character("F" * 64, body="male", traits={"Head": "Crown"})
    assert do_move(w, dry_run=True).hero == "E" * 64


def test_hero_override_by_argument_and_environment(world, monkeypatch):
    w = world
    w.state.add_character("E" * 64, body="male", traits={"Head": "Crown"})
    assert do_move(w, dry_run=True, hero="E" * 64).hero == "E" * 64
    monkeypatch.setenv("FLY_HERO", "E" * 64)
    assert do_move(w, dry_run=True, today=TODAY + timedelta(days=1)).hero == "E" * 64


def test_hero_follows_the_last_record(world):
    w = world
    w.state.add_character("E" * 64, body="male", traits={"Head": "Crown"})
    do_move(w, hero="E" * 64)
    rec = do_move(w, dry_run=True, today=TODAY + timedelta(days=1))
    assert rec.hero == "E" * 64


def test_no_eligible_hero_refuses(world):
    w = world
    w.state.characters.clear()
    w.state.add_character("B" * 64, body="ape", traits={"Head": "Crown"})
    with pytest.raises(L.LoopRefused, match="hero"):
        do_move(w, dry_run=True)
    w.state.add_character("C" * 64, body="male", traits={"Head": "Crown"}, mutable=False)
    with pytest.raises(L.LoopRefused, match="hero"):
        do_move(w, dry_run=True, hero="C" * 64)
    with pytest.raises(L.LoopRefused, match="hero"):
        do_move(w, dry_run=True, hero="Z" * 64)


# ------------------------------------------------------------------ decide (§4.1 steps 4–5)


def test_tabu_keeps_the_fly_off_a_recent_look(world):
    w = world
    old = do_move(w, dry_run=True, today=TODAY - timedelta(days=10))
    old.state, old.after = "done", list(PIRATE)
    old.changes = [{"slot": "Head", "value": "Pirate Hat"}]
    R.write(old)
    rec = do_move(w, dry_run=True)
    # step 1 cannot go to PIRATE (tabu): Monocle wins instead; step 2 then reaches
    # PIRATE_MONOCLE, which is not tabu
    assert tuple(rec.after) == PIRATE_MONOCLE
    assert [c["value"] for c in rec.changes] == ["Monocle", "Pirate Hat"]
    first_step_looks = {tuple(c["look"]) for c in rec.candidates[:4]}
    assert PIRATE not in first_step_looks
    # a look older than the window is not tabu
    older = R.read(R.path_for("testnet", TODAY - timedelta(days=10)), stamp_for(w))
    older.date = (TODAY - timedelta(days=31)).isoformat()
    R.write(older)
    os.remove(R.path_for("testnet", TODAY - timedelta(days=10)))
    os.remove(R.path_for("testnet", TODAY))
    rec = do_move(w, dry_run=True)
    assert [c["value"] for c in rec.changes] == ["Monocle", "Pirate Hat"] or \
        tuple(rec.candidates[1]["look"]) == PIRATE


def test_the_day_seed_drives_the_generator(world, monkeypatch):
    w = world
    seeds: list[int] = []
    real = L.np.random.default_rng

    def spy(seed=None):
        seeds.append(seed)
        return real(seed)

    monkeypatch.setattr(L.np.random, "default_rng", spy)
    do_move(w, dry_run=True)
    assert seeds == [R.date_seed(TODAY, "fly-v1")]
    do_move(w, dry_run=True, today=TODAY + timedelta(days=1))
    assert seeds[1] == R.date_seed(TODAY + timedelta(days=1), "fly-v1") != seeds[0]


def test_same_day_dry_runs_are_reproducible_at_high_temperature(world):
    w = world
    cfg = cfg_for(w.base, temperature=50.0)
    outs = {tuple(do_move(w, dry_run=True, cfg=cfg).after) for _ in range(4)}
    assert len(outs) == 1


def test_legality_uses_the_closet_and_the_layer_gate(world):
    w = world
    # Pirate Hat does not resolve on a male body -> not a candidate
    w.state.layers = {("male", "Eyes", "Monocle"), ("male", "Head", "Beanie")}
    rec = do_move(w, dry_run=True)
    looks = {tuple(c["look"]) for c in rec.candidates}
    assert all(lk[HEAD] != "Pirate Hat" for lk in looks)
    assert tuple(rec.after)[EYES] == "Monocle"


def test_none_is_legal_only_with_the_closets_none_unit(world):
    w = world
    w.state.layers = set()  # nothing resolves: only None could be legal
    rec = do_move(w, dry_run=True)
    assert rec.changes == [] and rec.state == "dry_run"
    w.state.add_closet("Head", "None")
    os.remove(R.path_for("testnet", TODAY))
    rec = do_move(w, dry_run=True)
    looks = {tuple(c["look"]) for c in rec.candidates}
    assert any(lk[HEAD] == "None" for lk in looks)


def test_a_stay_day_is_done_without_an_equip_and_still_posts(world):
    w = world
    w.state.closet_assets.clear()
    rec = do_move(w)
    assert rec.state == "done" and rec.changes == [] and rec.after == rec.before
    assert rec.equip_id is None
    assert "/api/equip" not in posts(w.state)
    assert rec.post["channel"] == "outbox"
    assert "Kept the look" in rec.post["text"]


def test_rarity_head_is_used_when_present_and_named_in_the_record(world):
    w = world
    head = RarityHead(mu=np.zeros(3), sd=np.ones(3), w=np.zeros(3), b=0.5,
                      snapshot_hash="ab" * 32, target_mu=0.0, target_sd=1.0)
    w.state.rarity_supply = {"network": "testnet", "as_of": 1, "n_live": 10,
                             "counts": {"Head": {"Crown": 8, "Pirate Hat": 1}}}
    rec = do_move(w, dry_run=True, rarity_head=head)
    assert rec.rarity_head == "ab" * 32
    assert w.brain.rarity_calls and w.brain.rarity_calls[0][1] == (4, 9)
    assert all(c["rarity"] == 0.5 for c in rec.candidates)
    # scores carry beta * taste_sd * rarity
    stay = rec.candidates[0]
    assert stay["changes"] == [] and abs(stay["score"] - (0.0 + 0.3 * 0.5)) < 1e-9


def test_rarity_head_is_loaded_from_the_snapshots_dir(world):
    from lfg_fly.brain import readout

    head = RarityHead(mu=np.zeros(3), sd=np.ones(3), w=np.zeros(3), b=0.25,
                      snapshot_hash="cd" * 32, target_mu=0.0, target_sd=1.0)
    base = paths.rarity_head_path("testnet", "cd" * 32)
    readout.save_head(base.with_name(base.name + ".npz"), head)
    assert L.load_rarity_head("testnet").snapshot_hash == "cd" * 32
    rec = do_move(world, dry_run=True)
    assert rec.rarity_head == "cd" * 32
    assert all(c["rarity"] == 0.25 for c in rec.candidates)


def test_without_a_rarity_head_rarity_is_zero_and_disclosed(world):
    w = world
    assert L.load_rarity_head("testnet") is None
    rec = do_move(w, dry_run=True)
    assert rec.rarity_head is None
    assert all(c["rarity"] == 0.0 for c in rec.candidates)
    assert w.brain.rarity_calls == []


# ------------------------------------------------------------------ helpers


def test_zorder_matches():
    z = FakeBrain().pinned_zorder()
    api = FL.FakeLfgState().z_order
    assert L.zorder_mismatch(api, z) is None
    assert L.zorder_mismatch({"layers": {}, "z_overrides": []}, z)
    assert L.zorder_mismatch(
        {**api, "z_overrides": [{"trait_type": "Head", "value": "X", "z": 1}]}, z)
    floaty = {"layers": {k: float(v) for k, v in api["layers"].items()}, "z_overrides": []}
    assert L.zorder_mismatch(floaty, z) is None


def test_inputs_hash_is_canonical():
    a = L.sha256_json({"b": 1, "a": [1, 2]})
    b = L.sha256_json({"a": [1, 2], "b": 1})
    assert a == b and len(a) == 64
    assert L.sha256_json({"a": 1}) != a


def test_pick_hero_rules():
    chars = [
        {"nft_id": "1", "body": "ape", "mutable": True, "blank": False},
        {"nft_id": "2", "body": "male", "mutable": False, "blank": False},
        {"nft_id": "3", "body": "male", "mutable": True, "blank": True},
        {"nft_id": "4", "body": "male", "mutable": True, "blank": False},
    ]
    assert L.pick_hero(chars, None, None)["nft_id"] == "4"
    assert L.pick_hero(chars, None, "4")["nft_id"] == "4"
    assert L.pick_hero(chars, "4", "1")["nft_id"] == "4"  # the explicit choice wins
    with pytest.raises(L.LoopRefused):
        L.pick_hero(chars, "2", None)
    with pytest.raises(L.LoopRefused):
        L.pick_hero(chars[:3], None, None)


def test_read_testnet_wallet_file(tmp_path):
    p = tmp_path / "wallet.json"
    p.write_text(json.dumps({"network": "testnet", "address": "rABC", "master_seed": "sMASTER",
                             "regular_seed": "sREGULAR"}))
    assert L.read_wallet_file(p) == ("rABC", "sREGULAR")
    p.write_text(json.dumps({"network": "mainnet", "address": "rABC", "regular_seed": "sR"}))
    with pytest.raises(ValueError):
        L.read_wallet_file(p, network="testnet")
    p.write_text(json.dumps({"address": "rABC"}))
    with pytest.raises(ValueError):
        L.read_wallet_file(p)


# ------------------------------------------------------------------ cli_move


def test_cli_move_registers_the_flags():
    from lfg_fly.body import cli_move

    parser = argparse.ArgumentParser()
    cli_move.register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["move"])
    assert (args.dry_run, args.date, args.hero, args.device) == (False, None, None, "cuda")
    args = parser.parse_args(["move", "--dry-run", "--date", "2026-09-25", "--hero", "A" * 64,
                              "--device", "cpu"])
    assert args.dry_run and args.date == "2026-09-25" and args.hero == "A" * 64
    assert args.device == "cpu"
    assert callable(args.func)


def test_cli_move_runs_the_loop_and_prints_the_record(monkeypatch, capsys, tmp_path):
    from lfg_fly.body import cli_move

    seen = {}

    async def fake_move(cfg, brain, *, dry_run, today, hero, **kw):
        seen.update(cfg=cfg, brain=brain, dry_run=dry_run, today=today, hero=hero)
        return R.Record(date="2026-09-25", stamp=R.Stamp("testnet", cfg.api_base, "rW"),
                        version="fly-v1", rarity_head=None, seed="00", hero="A" * 64,
                        before=list(CROWN), after=list(PIRATE),
                        changes=[{"slot": "Head", "value": "Pirate Hat"}],
                        candidates=[], considered=2, total=2,
                        neuron_stats={"ms": 1, "neurons_fired": 2},
                        inputs_hash={}, state="dry_run")

    monkeypatch.setattr(cli_move, "_load_brain", lambda cfg, device: "brain")
    monkeypatch.setattr(cli_move, "DOTENV", tmp_path / "no-such-env")
    monkeypatch.setattr(L, "move", fake_move)
    monkeypatch.setenv("FLY_NETWORK", "testnet")
    monkeypatch.delenv("FLY_ENABLED", raising=False)
    parser = argparse.ArgumentParser()
    cli_move.register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["move", "--dry-run", "--date", "2026-09-25", "--device", "cpu"])
    assert args.func(args) == 0
    assert seen["dry_run"] is True and seen["today"] == date(2026, 9, 25)
    assert seen["brain"] == "brain" and seen["cfg"].network == "testnet"
    out = capsys.readouterr().out
    assert "dry_run" in out and "Pirate Hat" in out


def test_cli_move_reports_a_refusal_as_exit_2(monkeypatch, capsys, tmp_path):
    from lfg_fly.body import cli_move

    async def refusing(cfg, brain, **kw):
        raise L.LoopRefused("FLY_ENABLED is not 1")

    monkeypatch.setattr(cli_move, "_load_brain", lambda cfg, device: "brain")
    monkeypatch.setattr(cli_move, "DOTENV", tmp_path / "no-such-env")
    monkeypatch.setattr(L, "move", refusing)
    parser = argparse.ArgumentParser()
    cli_move.register(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["move", "--device", "cpu"]).func(
        parser.parse_args(["move", "--device", "cpu"])) == 2
    assert "FLY_ENABLED" in capsys.readouterr().out


def test_cli_move_reports_an_lfg_refusal_as_one_line(monkeypatch, capsys):
    """A 503 agent_disabled from LFG is a one-line message and exit 2, not a traceback."""
    import argparse

    from lfg_fly.body import cli_move
    from lfg_fly.body.client import LfgError

    async def refused(*_a, **_k):
        raise LfgError(503, "agent_disabled", {"code": "agent_disabled"})

    monkeypatch.setattr(cli_move.loop, "move", refused)
    monkeypatch.setattr(cli_move, "_load_brain", lambda cfg, device: object())
    monkeypatch.setattr(cli_move, "load_dotenv", lambda path: 0)
    monkeypatch.setenv("FLY_NETWORK", "testnet")
    parser = argparse.ArgumentParser()
    cli_move.register(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["move", "--dry-run"])
    assert args.func(args) == 2
    out = capsys.readouterr().out
    assert "503 agent_disabled" in out and "AGENT_SIGNIN_ENABLED" in out
