"""The fly's voice (spec §3.2 naming at runtime, §4.5 posts). Hermetic: no network,
no ~/fly-data (conftest), no body modules (fakes stand in for outbox and Record)."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field
from datetime import date

import pytest
from PIL import Image

import lfg_fly.voice as voice
from lfg_fly.brain.senses import NONE, SLOTS
from lfg_fly.voice import card as card_mod
from lfg_fly.voice import naming, post

# ------------------------------------------------------------------ fixtures / fakes

BASE = ("Sky", "Wings", "Male", "Suit", "Grin", "Flat", "Laser", "Crown", "Pipe")


def look(**over: str) -> tuple[str, ...]:
    """BASE with some slots overridden by name."""
    vals = dict(zip(SLOTS, BASE, strict=True))
    for k, v in over.items():
        assert k in vals, k
        vals[k] = v
    return tuple(vals[s] for s in SLOTS)


def rec(a, b, winner="A", idea_a="Idea A", idea_b="Idea B", **extra) -> dict:
    """A critic record as lfg_fly.teacher.critic.assemble writes it."""
    return {"pair_id": 0, "family": 0, "near": True, "a": list(a), "b": list(b),
            "winner": winner, "strength": 2, "idea_A": idea_a, "idea_B": idea_b,
            "reason": "", "batch": "b000", **extra}


@dataclass
class FakeStamp:
    network: str = "testnet"
    lfg_api_base: str = "http://localhost:8177"
    wallet: str = "rFLY"


@dataclass
class FakeRecord:
    """Only the fields the voice reads (the body's Record has more)."""

    date: str = "2026-09-25"
    stamp: FakeStamp = field(default_factory=FakeStamp)
    hero: str = "000A" * 16
    before: list = field(default_factory=lambda: list(look()))
    after: list = field(default_factory=lambda: list(look(Head="Pirate Hat", Eyes="Monocle")))
    changes: list = field(default_factory=lambda: [{"slot": "Head", "value": "Pirate Hat"},
                                                   {"slot": "Eyes", "value": "Monocle"}])
    considered: int = 147
    total: int = 147
    neuron_stats: dict = field(default_factory=lambda: {"neurons_fired": 9812, "ms": 61.2})
    state: str = "done"
    post: dict | None = None


@dataclass
class FakeConfig:
    network: str = "testnet"
    x_monthly_budget: int = 40


@pytest.fixture
def no_x_env(monkeypatch):
    for k in ("FLY_X_API_KEY", "FLY_X_API_SECRET", "FLY_X_ACCESS_TOKEN", "FLY_X_ACCESS_SECRET"):
        monkeypatch.delenv(k, raising=False)


# ------------------------------------------------------------------ naming (§3.2)


def test_slot_weights_follow_visual_prominence():
    assert naming.SLOT_WEIGHTS["Background"] == 2
    assert naming.SLOT_WEIGHTS["Clothing"] == 2
    assert naming.SLOT_WEIGHTS["Head"] == 2
    for s in ("Back", "Body", "Mouth", "Eyebrows", "Eyes", "Accessory"):
        assert naming.SLOT_WEIGHTS[s] == 1
    assert set(naming.SLOT_WEIGHTS) == set(SLOTS)
    assert naming.TOTAL_WEIGHT == 12


def test_similarity_is_weighted_overlap():
    assert naming.similarity(look(), look()) == 1.0
    other = tuple(v + "x" for v in look())
    assert naming.similarity(look(), other) == 0.0
    # one heavy slot differs: 10/12; one light slot differs: 11/12
    assert naming.similarity(look(), look(Head="Hat")) == pytest.approx(10 / 12)
    assert naming.similarity(look(), look(Eyes="Dots")) == pytest.approx(11 / 12)


def test_nearest_name_exact_match_is_that_looks_idea():
    a, b = look(), look(Head="Hat", Clothing="Robe", Background="Cave")
    records = [rec(a, b, winner="A", idea_a="Crowned Dandy", idea_b="Cave Monk")]
    assert naming.nearest_name(a, records) == "Crowned Dandy"
    # the loser is a judged look too: it carries its own idea
    assert naming.nearest_name(b, records) == "Cave Monk"


def test_nearest_name_threshold_boundary():
    a = look()
    records = [rec(a, look(Head="Hat", Clothing="Robe", Background="Cave"))]
    # 8/12 = 0.667 shared (differs in Back, Mouth, Eyebrows, Eyes: 4 light slots)
    near = look(Back="Nope", Mouth="Nope", Eyebrows="Nope", Eyes="Nope")
    assert naming.similarity(near, a) == pytest.approx(8 / 12)
    assert naming.nearest_name(near, records) == "Idea A"
    # 7/12 = 0.583 shared (differs in Head and 3 light slots) -> untitled
    far = look(Head="Nope", Mouth="Nope", Eyebrows="Nope", Eyes="Nope")
    assert naming.similarity(far, a) == pytest.approx(7 / 12)
    assert naming.nearest_name(far, records) is None
    # the threshold is a parameter
    assert naming.nearest_name(far, records, threshold=0.5) == "Idea A"


def test_nearest_name_picks_the_highest_weighted_overlap():
    hero = look()
    light = look(Eyes="Dots", Mouth="Frown")  # differs in two light slots -> 10/12
    heavy = look(Background="Cave")  # differs in one heavy slot -> 10/12
    heavier = look(Eyes="Dots")  # differs in one light slot -> 11/12
    other = tuple(v + "z" for v in hero)
    records = [rec(light, other, idea_a="Light"), rec(heavy, other, idea_a="Heavy"),
               rec(other, heavier, winner="A", idea_a="Other", idea_b="Heavier")]
    assert naming.nearest_name(hero, records) == "Heavier"


def test_nearest_name_tie_prefers_the_winner():
    hero = look()
    a = look(Back="Cape")  # 11/12
    b = look(Mouth="Frown")  # 11/12
    assert naming.nearest_name(hero, [rec(a, b, winner="B", idea_a="A idea",
                                          idea_b="B idea")]) == "B idea"
    assert naming.nearest_name(hero, [rec(a, b, winner="A", idea_a="A idea",
                                          idea_b="B idea")]) == "A idea"


def test_nearest_name_skips_flipped_pairs_empty_ideas_and_malformed_records():
    hero = look()
    other = tuple(v + "z" for v in hero)
    flipped = rec(hero, other, idea_a="Flipped Idea", flipped=True)
    empty = rec(hero, other, idea_a="   ")
    short = rec(hero[:5], other, idea_a="Short")
    assert naming.nearest_name(hero, [flipped, empty, short]) is None
    # a non-flipped record with the same look wins as usual
    ok = rec(hero, other, idea_a="Good Idea", flipped=False)
    assert naming.nearest_name(hero, [flipped, empty, short, ok]) == "Good Idea"


def test_nearest_name_strips_and_caps_the_idea():
    hero = look()
    other = tuple(v + "z" for v in hero)
    assert naming.nearest_name(hero, [rec(hero, other, idea_a="  Retired Space Pirate ")]) == (
        "Retired Space Pirate")
    long = rec(hero, other, idea_a="x" * 200)
    got = naming.nearest_name(hero, [long])
    assert got is not None and len(got) <= naming.MAX_NAME_CHARS


def test_nearest_name_empty_records():
    assert naming.nearest_name(look(), []) is None


def test_nearest_returns_the_score_too():
    hero = look()
    other = tuple(v + "z" for v in hero)
    name, score = naming.nearest(hero, [rec(look(Eyes="Dots"), other, idea_a="Near")])
    assert name == "Near" and score == pytest.approx(11 / 12)
    assert naming.nearest(hero, []) == (None, 0.0)


def test_load_critic_records_reads_every_round(tmp_path):
    d = tmp_path / "critic"
    d.mkdir()
    r1 = rec(look(), look(Head="Hat"), idea_a="One")
    r2 = rec(look(), look(Head="Cap"), idea_a="Two")
    (d / "r1.jsonl").write_text(json.dumps(r1) + "\n")
    (d / "r2.jsonl").write_text(json.dumps(r2) + "\n\n")
    (d / "r1-qc.json").write_text("{}")  # not a round file
    got = naming.load_critic_records(d)
    assert [r["idea_A"] for r in got] == ["One", "Two"]
    assert naming.load_critic_records(tmp_path / "missing") == []


# ------------------------------------------------------------------ card (§4.5)


def _solid(size, color, mode="RGB"):
    return Image.new(mode, size, color)


def test_before_after_is_1200_by_675_rgb():
    im = card_mod.before_after(_solid((512, 512), (200, 30, 30)), _solid((512, 512), (30, 30, 200)),
                               12, [{"slot": "Head", "value": "Pirate Hat", "from": "Crown"}])
    assert im.size == card_mod.CARD_SIZE == (1200, 675)
    assert im.mode == "RGB"


def test_before_after_shows_before_left_and_after_right():
    im = card_mod.before_after(_solid((512, 512), (200, 30, 30)), _solid((512, 512), (30, 30, 200)),
                               3, [])
    (lx, ly), (rx, ry) = card_mod.panel_centres()
    r, g, b = im.getpixel((lx, ly))
    assert r > 150 and b < 80, (r, g, b)
    r, g, b = im.getpixel((rx, ry))
    assert b > 150 and r < 80, (r, g, b)


def test_before_after_accepts_any_size_or_mode():
    wide = _solid((900, 300), (10, 200, 10), "RGBA")
    grey = _solid((1500, 1500), 128, "L")
    im = card_mod.before_after(wide, grey, 1, [{"slot": "Eyes", "value": "Monocle"}])
    assert im.size == (1200, 675)
    # the wide image is letterboxed: the panel's top-centre shows the panel background, not green
    (lx, ly), _ = card_mod.panel_centres()
    top = im.getpixel((lx, ly - card_mod.PANEL // 2 + 4))
    assert top[1] < 150, top
    mid = im.getpixel((lx, ly))
    assert mid[1] > 150, mid


def test_before_after_with_many_changes_and_none_values_does_not_overflow():
    changes = [{"slot": s, "value": "A Very Long Trait Value Name", "from": NONE}
               for s in ("Head", "Eyes", "Clothing", "Accessory", "Back")]
    im = card_mod.before_after(_solid((64, 64), (1, 2, 3)), _solid((64, 64), (3, 2, 1)),
                               999, changes)
    assert im.size == (1200, 675)


def test_change_lines_reads_from_and_none():
    lines = card_mod.change_lines([{"slot": "Head", "value": "Pirate Hat", "from": "Crown"},
                                   {"slot": "Accessory", "value": NONE, "from": "Pipe"},
                                   {"slot": "Eyes", "value": "Monocle"}])
    assert lines == ["Head: Crown → Pirate Hat", "Accessory: Pipe → nothing", "Eyes → Monocle"]
    assert card_mod.change_lines([]) == [card_mod.STAY_LINE]


def test_with_from_fills_the_before_value():
    before, after = look(), look(Head="Pirate Hat", Eyes="Monocle")
    changes = [{"slot": "Head", "value": "Pirate Hat"}, {"slot": "Eyes", "value": "Monocle"}]
    got = card_mod.with_from(changes, before)
    assert got == [{"slot": "Head", "value": "Pirate Hat", "from": "Crown"},
                   {"slot": "Eyes", "value": "Monocle", "from": "Laser"}]
    assert changes[0] == {"slot": "Head", "value": "Pirate Hat"}  # input untouched
    assert card_mod.with_from(changes, list(before)) == got
    assert card_mod.with_from(changes, None) == changes
    del after


# ------------------------------------------------------------------ post text (§4.5)


EXAMPLE = ('🪰 Day 12. Tried 147 outfits in 61 ms of fly-brain time; 9,812 neurons fired. '
           'Head: Crown → Pirate Hat · Eyes: Laser → Monocle. '
           'Critic\'s read: "Retired Space Pirate".')


def test_compose_matches_the_spec_example():
    assert post.compose(FakeRecord(), "Retired Space Pirate", 12) == EXAMPLE


def test_compose_untitled_when_no_name():
    text = post.compose(FakeRecord(), None, 12)
    assert "untitled" in text.lower()
    assert "Critic's read" not in text
    assert text.startswith("🪰 Day 12. Tried 147 outfits")


def test_compose_discloses_a_cap_as_considered_of_total():
    text = post.compose(FakeRecord(considered=120, total=147), None, 1)
    assert "Tried 120 of 147 outfits" in text


def test_compose_stay_day():
    r = FakeRecord(after=list(look()), changes=[])
    text = post.compose(r, None, 5)
    assert card_mod.STAY_LINE in text
    assert "→" not in text
    assert "9,812 neurons fired" in text


def test_compose_reads_none_as_nothing_and_singular_outfit():
    r = FakeRecord(after=list(look(Accessory=NONE)), changes=[{"slot": "Accessory", "value": NONE}],
                   considered=1, total=1, neuron_stats={"neurons_fired": 1, "ms": 0.4})
    text = post.compose(r, "Bare", 2)
    assert "Accessory: Pipe → nothing" in text
    assert "Tried 1 outfit in" in text
    assert "1 neuron fired" in text
    assert "0 ms" in text


def test_compose_tolerates_missing_stats_and_dict_records():
    r = {"before": list(look()), "after": list(look(Head="Hat")),
         "changes": [{"slot": "Head", "value": "Hat"}], "considered": 3, "total": 3,
         "neuron_stats": {}}
    text = post.compose(r, None, 1)
    assert "Head: Crown → Hat" in text
    assert "0 neurons fired" in text


def test_compose_stays_within_x_limit():
    name = "An Absurdly Long Critic Idea That Nobody Would Ever Type Out In Full " * 3
    changes = [{"slot": s, "value": "Extraordinarily Long Trait Value " * 2} for s in
               ("Head", "Eyes", "Clothing")]
    after = list(look())
    for c in changes:
        after[SLOTS.index(c["slot"])] = c["value"]
    r = FakeRecord(after=after, changes=changes, considered=123456, total=999999,
                   neuron_stats={"neurons_fired": 123456789, "ms": 123456})
    text = post.compose(r, name, 12345)
    assert post.x_length(text) <= post.X_MAX_CHARS
    assert text.startswith("🪰 Day 12345.")
    # the example itself is well within the limit and is never shortened
    assert post.compose(FakeRecord(), "Retired Space Pirate", 12) == EXAMPLE


def test_x_length_weights_like_x():
    assert post.x_length("abc") == 3
    assert post.x_length("🪰") == 2
    assert post.x_length("→") == 2
    assert post.x_length("é") == 1
    assert post.x_length("") == 0


def test_day_number():
    assert post.day_number("2026-09-25", "2026-09-25") == 1
    assert post.day_number(date(2026, 9, 14), date(2026, 9, 25)) == 12
    assert post.day_number("2026-09-14", date(2026, 9, 25)) == 12


# ------------------------------------------------------------------ publish (§4.5)


def _fake_writer(tmp_path, calls):
    out = tmp_path / "outbox"

    def write(network, kind, payload, when=None):
        out.mkdir(parents=True, exist_ok=True)
        calls.append((network, kind, payload, when))
        p = out / f"20260925T150000Z-{kind}.json"
        p.write_text(json.dumps(payload))
        return p

    return write


def test_publish_without_credentials_goes_to_the_outbox(tmp_path, no_x_env):
    calls: list = []
    im = _solid((1200, 675), (5, 5, 5))
    record = FakeRecord()
    text = post.compose(record, "Retired Space Pirate", 12)
    got = post.publish(FakeConfig(), text, im, record, writer=_fake_writer(tmp_path, calls),
                       name="Retired Space Pirate", day=12)
    assert got["channel"] == "outbox"
    assert got["reason"] == "no_credentials"
    (network, kind, payload, _when), = calls
    assert network == "testnet" and kind == "post"
    assert payload["text"] == text
    assert payload["date"] == "2026-09-25" and payload["hero"] == record.hero
    assert payload["before"] == record.before and payload["after"] == record.after
    assert payload["changes"] == record.changes
    assert payload["name"] == "Retired Space Pirate" and payload["day"] == 12
    assert payload["reason"] == "no_credentials"
    # the PNG sits beside the JSON with the same stem (the writer names the file)
    json_path = tmp_path / "outbox" / "20260925T150000Z-post.json"
    png_path = json_path.with_suffix(".png")
    assert got["path"] == str(json_path) and got["card"] == str(png_path)
    assert png_path.exists()
    with Image.open(png_path) as back:
        assert back.size == (1200, 675) and back.format == "PNG"
    # no link anywhere in what goes out
    assert "http" not in text


def test_publish_with_credentials_but_no_poster_still_goes_to_the_outbox(tmp_path, monkeypatch):
    for k in ("FLY_X_API_KEY", "FLY_X_API_SECRET", "FLY_X_ACCESS_TOKEN", "FLY_X_ACCESS_SECRET"):
        monkeypatch.setenv(k, "x")
    calls: list = []
    got = post.publish(FakeConfig(), "hi", _solid((1200, 675), (0, 0, 0)), FakeRecord(),
                       writer=_fake_writer(tmp_path, calls))
    assert got["channel"] == "outbox" and got["reason"] == "no_poster"
    assert len(calls) == 1
    # neither the payload nor the result carries the credentials (their value is "x")
    for blob in (calls[0][2], got):
        dumped = json.dumps(blob)
        assert '"x"' not in dumped and "secret" not in dumped.lower()
        assert "token" not in dumped.lower() and "FLY_X" not in dumped


def test_publish_with_a_poster_posts_and_skips_the_outbox(tmp_path, no_x_env):
    calls: list = []
    posted: list = []

    def poster(text, png):
        posted.append((text, png))
        return {"id": "1", "url": "https://x.com/lfgfly/status/1"}

    got = post.publish(FakeConfig(), "hi", _solid((1200, 675), (9, 9, 9)), FakeRecord(),
                       writer=_fake_writer(tmp_path, calls), poster=poster)
    assert got["channel"] == "x" and got["result"]["id"] == "1" and got["text"] == "hi"
    assert calls == []
    (text, png), = posted
    assert text == "hi" and png[:8] == b"\x89PNG\r\n\x1a\n"


def test_publish_reports_the_reason_it_was_given(tmp_path, monkeypatch):
    """The loop passes `x_post.poster_for`'s reason (no credentials, or the month's budget);
    it lands in the result and the payload as is."""
    for k in ("FLY_X_API_KEY", "FLY_X_API_SECRET", "FLY_X_ACCESS_TOKEN", "FLY_X_ACCESS_SECRET"):
        monkeypatch.setenv(k, "x")
    calls: list = []
    got = post.publish(FakeConfig(), "hi", _solid((1200, 675), (0, 0, 0)), FakeRecord(),
                       writer=_fake_writer(tmp_path, calls), reason="x_budget: 40 of 40 posts")
    assert got["channel"] == "outbox" and got["reason"] == "x_budget: 40 of 40 posts"
    assert calls[0][2]["reason"] == "x_budget: 40 of 40 posts"


def test_publish_falls_back_to_the_outbox_when_the_poster_fails(tmp_path, no_x_env):
    """X's refusal never loses the post: the text and the card go to the outbox with the
    failure as the reason (§4.5)."""
    calls: list = []

    def poster(text, png):
        raise RuntimeError("media upload: X answered HTTP 403: forbidden")

    got = post.publish(FakeConfig(), "hi", _solid((1200, 675), (9, 9, 9)), FakeRecord(),
                       writer=_fake_writer(tmp_path, calls), poster=poster)
    assert got["channel"] == "outbox"
    assert got["reason"] == "x_failed: RuntimeError: media upload: X answered HTTP 403: forbidden"
    (network, kind, payload, _when), = calls
    assert kind == "post" and payload["reason"] == got["reason"] and payload["text"] == "hi"
    assert (tmp_path / "outbox" / "20260925T150000Z-post.png").exists()


def test_publish_reads_x_credentials_from_the_environment(monkeypatch):
    for k in ("FLY_X_API_KEY", "FLY_X_API_SECRET", "FLY_X_ACCESS_TOKEN", "FLY_X_ACCESS_SECRET"):
        monkeypatch.delenv(k, raising=False)
    assert post.x_credentials() is None
    env = {"FLY_X_API_KEY": "k", "FLY_X_API_SECRET": "s", "FLY_X_ACCESS_TOKEN": "t",
           "FLY_X_ACCESS_SECRET": "ts"}
    assert post.x_credentials(env) == {"api_key": "k", "api_secret": "s", "access_token": "t",
                                       "access_secret": "ts"}
    assert post.x_credentials({**env, "FLY_X_ACCESS_SECRET": ""}) is None


def test_publish_default_writer_is_imported_lazily():
    """Another agent owns lfg_fly.body.outbox: post.py must not import it at module level."""
    tree = ast.parse(inspect.getsource(post))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("lfg_fly.body"), ast.dump(node)
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("lfg_fly.body") for a in node.names)
    assert "lfg_fly.body.outbox" in inspect.getsource(post.publish) or (
        "lfg_fly.body" in inspect.getsource(post._default_writer))


def test_package_exports():
    assert voice.nearest_name is naming.nearest_name
    assert voice.before_after is card_mod.before_after
    assert voice.compose is post.compose and voice.publish is post.publish
