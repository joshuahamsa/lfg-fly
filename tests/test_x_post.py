"""Posting to X (spec §4.5): OAuth 1.0a, the monthly budget and the two-call poster.

Hermetic: the signing is checked against the published HMAC-SHA1 test vector, the poster
against an injected transport and against a local aiohttp server in a thread that verifies
the OAuth header independently and parses the multipart upload. FLY_DATA_DIR is the
autouse tmp dir. No network, no X.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import threading
from email.parser import BytesParser
from email.policy import HTTP
from urllib.parse import parse_qsl, quote, unquote, urlsplit

import pytest
from aiohttp import web

from lfg_fly import paths
from lfg_fly.voice import x_post as X

# Twitter's documented "Creating a signature" example (the OAuth 1.0a HMAC-SHA1 vector).
VECTOR_CREDS = {
    "api_key": "xvz1evFS4wEEPTGEFPHBog",
    "api_secret": "kAcSOqF21Fu85e7zjz7ZN2U4ZRhfV3WpwPAoE3Z7kBw",
    "access_token": "370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb",
    "access_secret": "LswwdoUaIvS8ltyTt5jkRh4J50vUPVVHtR2YPi5kE",
}
VECTOR_NONCE = "kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg"
VECTOR_TIMESTAMP = 1318622958
VECTOR_SIGNATURE = "hCtSmYh+iHYCEqBWrE7C7hYmtUk="

CREDS = {"api_key": "ck-key", "api_secret": "cs-secret", "access_token": "at-token",
         "access_secret": "as-secret"}
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


class Cfg:
    def __init__(self, budget: int = 40):
        self.network = "testnet"
        self.x_monthly_budget = budget


# ------------------------------------------------------------------ OAuth 1.0a


def test_oauth1_signature_matches_the_documented_vector():
    url = "https://api.twitter.com/1.1/statuses/update.json?include_entities=true"
    form = [("status", "Hello Ladies + Gentlemen, a signed OAuth request!")]
    header = X.oauth1_header("POST", url, VECTOR_CREDS, form=form, nonce=VECTOR_NONCE,
                             timestamp=VECTOR_TIMESTAMP)
    assert header == (
        'OAuth oauth_consumer_key="xvz1evFS4wEEPTGEFPHBog", '
        'oauth_nonce="kYjzVBB8Y0ZFabxSWbWovY3uYSQ2pTgmZeNu2VS4cg", '
        'oauth_signature="hCtSmYh%2BiHYCEqBWrE7C7hYmtUk%3D", '
        'oauth_signature_method="HMAC-SHA1", oauth_timestamp="1318622958", '
        'oauth_token="370773112-GmHxMAgYyLbNEtIKZeRNFsMKPR9EyMZeS9weJAEb", '
        'oauth_version="1.0"'
    )


def verify_oauth(method: str, url: str, header: str, creds: dict) -> bool:
    """An independent RFC 5849 check of an `Authorization: OAuth` header (no code shared
    with x_post): rebuild the base string from the header's own parameters and the URL."""
    assert header.startswith("OAuth ")
    fields = {k: unquote(v) for k, v in re.findall(r'(\w+)="([^"]*)"', header[6:])}
    given = fields.pop("oauth_signature")
    assert fields["oauth_signature_method"] == "HMAC-SHA1" and fields["oauth_version"] == "1.0"
    assert fields["oauth_consumer_key"] == creds["api_key"]
    assert fields["oauth_token"] == creds["access_token"]
    params = list(fields.items()) + parse_qsl(urlsplit(url).query, keep_blank_values=True)

    def enc(s: str) -> str:
        return quote(s, safe="")

    u = urlsplit(url)
    port = f":{u.port}" if u.port and u.port not in (80, 443) else ""
    base_url = f"{u.scheme.lower()}://{u.hostname.lower()}{port}{u.path}"
    normalized = "&".join(f"{k}={v}" for k, v in sorted((enc(k), enc(v)) for k, v in params))
    base = "&".join((method.upper(), enc(base_url), enc(normalized)))
    key = f"{enc(creds['api_secret'])}&{enc(creds['access_secret'])}".encode()
    want = base64.b64encode(hmac.new(key, base.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(want, given)


def test_oauth1_header_is_fresh_verifiable_and_body_independent():
    url = "https://api.x.com/2/media/upload?x=1&y=a%20b"
    a = X.oauth1_header("POST", url, CREDS)
    b = X.oauth1_header("POST", url, CREDS)
    assert a != b  # a fresh nonce every time
    assert verify_oauth("POST", url, a, CREDS) and verify_oauth("POST", url, b, CREDS)
    assert not verify_oauth("GET", url, a, CREDS)
    assert not verify_oauth("POST", url, a, {**CREDS, "access_secret": "other"})
    assert not verify_oauth("POST", url.replace("y=a%20b", "y=a"), a, CREDS)
    # a form body is signed; a multipart or JSON body is not (RFC 5849 §3.4.1.3.1)
    nonce, ts = "n" * 32, 1_700_000_000
    plain = X.oauth1_header("POST", url, CREDS, nonce=nonce, timestamp=ts)
    formed = X.oauth1_header("POST", url, CREDS, form=[("status", "x")], nonce=nonce,
                             timestamp=ts)
    assert plain == X.oauth1_header("POST", url, CREDS, nonce=nonce, timestamp=ts)
    assert plain != formed
    for secret in ("cs-secret", "as-secret"):
        assert secret not in a


def test_multipart_body_holds_the_fields_and_the_file():
    content_type, body = X.multipart([("media_category", "tweet_image")], "media", "card.png",
                                     "image/png", PNG)
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode()
                                              + b"\r\n\r\n" + body)
    parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
    assert set(parts) == {"media_category", "media"}
    assert parts["media_category"].get_content() == "tweet_image"
    assert parts["media"].get_filename() == "card.png"
    assert parts["media"].get_content_type() == "image/png"
    assert parts["media"].get_payload(decode=True) == PNG
    assert body.endswith(b"--\r\n")


# ------------------------------------------------------------------ the monthly budget


def test_monthly_budget_counts_per_network_and_month(_data_dir):
    b = X.MonthlyBudget("testnet", 2, month="2026-09")
    assert b.path == _data_dir / "testnet" / "x-posts-2026-09.json"
    assert b.count == 0 and b.remaining == 2 and not b.path.exists()
    b.check()
    b.charge(chars=10)
    b.charge(chars=11)
    assert b.count == 2 and b.remaining == 0
    with pytest.raises(X.BudgetExhausted, match="2 of 2 posts used in 2026-09"):
        b.check()
    with pytest.raises(X.BudgetExhausted):
        b.charge()
    data = json.loads(b.path.read_text())
    assert data["network"] == "testnet" and data["month"] == "2026-09"
    assert [p["chars"] for p in data["posts"]] == [10, 11] and all("at" in p for p in data["posts"])
    # the next month starts afresh; another network's file is refused
    assert X.MonthlyBudget("testnet", 2, month="2026-10").remaining == 2
    (_data_dir / "mainnet").mkdir(parents=True)
    (_data_dir / "mainnet" / "x-posts-2026-09.json").write_text(b.path.read_text())
    with pytest.raises(X.XPostError, match="mainnet"):
        X.MonthlyBudget("mainnet", 2, month="2026-09").count  # noqa: B018
    with pytest.raises(ValueError):
        X.MonthlyBudget("testnet", -1)
    assert X.MonthlyBudget("testnet", 0).remaining == 0


# ------------------------------------------------------------------ the poster


def _transport(calls: list, *, upload=None, post=None):
    upload = upload or (200, json.dumps({"data": {"id": "710511363345354753"}}).encode())
    post = post or (201, json.dumps({"data": {"id": "1445880548472328192", "text": "hi"}}).encode())

    def send(method, url, headers, body):
        calls.append((method, url, dict(headers), body))
        return upload if url == X.MEDIA_UPLOAD_URL else post

    return send


def test_poster_uploads_the_card_then_posts_with_its_media_id(_data_dir):
    calls: list = []
    budget = X.MonthlyBudget("testnet", 40, month="2026-09")
    poster = X.XPoster(CREDS, budget, transport=_transport(calls))
    out = poster("🪰 Day 1. hi", PNG)
    assert out == {"id": "1445880548472328192", "media_id": "710511363345354753",
                   "url": "https://x.com/i/status/1445880548472328192"}
    (m1, u1, h1, b1), (m2, u2, h2, b2) = calls
    assert (m1, u1) == ("POST", X.MEDIA_UPLOAD_URL) and (m2, u2) == ("POST", X.TWEETS_URL)
    assert verify_oauth(m1, u1, h1["Authorization"], CREDS)
    assert verify_oauth(m2, u2, h2["Authorization"], CREDS)
    assert h1["Content-Type"].startswith("multipart/form-data; boundary=") and PNG in b1
    assert b'name="media_category"\r\n\r\ntweet_image' in b1
    assert h2["Content-Type"] == "application/json"
    assert json.loads(b2) == {"text": "🪰 Day 1. hi",
                              "media": {"media_ids": ["710511363345354753"]}}
    assert budget.count == 1
    for blob in (repr(poster), str(poster), json.dumps(out), budget.path.read_text()):
        for secret in CREDS.values():
            assert secret not in blob


def test_poster_charges_before_the_upload_and_reports_a_refusal_without_secrets(_data_dir):
    budget = X.MonthlyBudget("testnet", 40, month="2026-09")
    calls: list = []
    refused = (401, json.dumps({"title": "Unauthorized", "detail": "bad token"}).encode())
    poster = X.XPoster(CREDS, budget, transport=_transport(calls, upload=refused))
    with pytest.raises(X.XPostError, match="media upload: X answered HTTP 401") as info:
        poster("hi", PNG)
    assert "bad token" in str(info.value) and len(calls) == 1
    assert budget.count == 1  # a refused post still counts: charged before the attempt
    for secret in CREDS.values():
        assert secret not in str(info.value)

    def broken(method, url, headers, body):
        raise ConnectionError("no route to host")

    with pytest.raises(X.XPostError, match="media upload: ConnectionError: no route"):
        X.XPoster(CREDS, budget, transport=broken)("hi", PNG)
    junk = (200, b"<html>not json</html>")
    with pytest.raises(X.XPostError, match="not JSON"):
        X.XPoster(CREDS, budget, transport=_transport([], upload=junk))("hi", PNG)
    no_id = (200, json.dumps({"data": {}}).encode())
    with pytest.raises(X.XPostError, match="no media id"):
        X.XPoster(CREDS, budget, transport=_transport([], upload=no_id))("hi", PNG)
    no_post = (201, json.dumps({"data": {"text": "hi"}}).encode())
    with pytest.raises(X.XPostError, match="no post id"):
        X.XPoster(CREDS, budget, transport=_transport([], post=no_post))("hi", PNG)
    # the v1.1 media shape is still understood
    old = (200, json.dumps({"media_id": 7, "media_id_string": "7"}).encode())
    calls.clear()
    legacy = X.XPoster(CREDS, budget, transport=_transport(calls, upload=old))("hi", PNG)
    assert legacy["media_id"] == "7"
    assert json.loads(calls[1][3])["media"]["media_ids"] == ["7"]
    with pytest.raises(X.BudgetExhausted):
        X.XPoster(CREDS, X.MonthlyBudget("testnet", 0), transport=_transport([]))("hi", PNG)
    with pytest.raises(ValueError, match="incomplete"):
        X.XPoster({**CREDS, "api_secret": ""}, budget)


class LocalX:
    """A stand-in for X's two endpoints on 127.0.0.1, served from a thread so the poster's
    blocking urllib transport can talk to it; it verifies every OAuth header on its own."""

    def __init__(self):
        self.seen: list[dict] = []
        self.base_url = ""
        self._loop = None
        self._thread = None
        self._runner = None

    async def _upload(self, request: web.Request) -> web.Response:
        ok = verify_oauth("POST", str(request.url), request.headers.get("Authorization", ""), CREDS)
        form = await request.post()
        media = form["media"]
        self.seen.append({"path": request.path, "oauth_ok": ok,
                          "category": form.get("media_category"),
                          "filename": media.filename, "content_type": media.content_type,
                          "bytes": media.file.read()})
        if not ok:
            return web.json_response({"title": "Unauthorized"}, status=401)
        return web.json_response({"data": {"id": "42", "media_key": "3_42"}})

    async def _tweets(self, request: web.Request) -> web.Response:
        ok = verify_oauth("POST", str(request.url), request.headers.get("Authorization", ""), CREDS)
        body = await request.json()
        self.seen.append({"path": request.path, "oauth_ok": ok, "json": body,
                          "content_type": request.content_type})
        if not ok:
            return web.json_response({"title": "Unauthorized"}, status=401)
        return web.json_response({"data": {"id": "99", "text": body.get("text")}}, status=201)

    def __enter__(self) -> LocalX:
        ready = threading.Event()

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def serve() -> None:
                app = web.Application()
                app.router.add_post("/2/media/upload", self._upload)
                app.router.add_post("/2/tweets", self._tweets)
                self._runner = web.AppRunner(app)
                await self._runner.setup()
                site = web.TCPSite(self._runner, "127.0.0.1", 0)
                await site.start()
                host, port = self._runner.addresses[0][:2]
                self.base_url = f"http://{host}:{port}"
                ready.set()

            loop.run_until_complete(serve())
            loop.run_forever()
            loop.run_until_complete(self._runner.cleanup())
            loop.close()

        self._thread = threading.Thread(target=run, name="local-x", daemon=True)
        self._thread.start()
        assert ready.wait(10) and self.base_url
        return self

    def __exit__(self, *exc) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


def test_poster_over_the_wire_against_a_local_server(_data_dir, monkeypatch):
    """The default urllib transport end to end: a server on a non-default port verifies the
    OAuth 1.0a header independently, parses the multipart upload and gets the PNG intact."""
    with LocalX() as x:
        monkeypatch.setattr(X, "MEDIA_UPLOAD_URL", x.base_url + "/2/media/upload")
        monkeypatch.setattr(X, "TWEETS_URL", x.base_url + "/2/tweets")
        budget = X.MonthlyBudget("testnet", 40, month="2026-09")
        out = X.XPoster(CREDS, budget)("🪰 Day 1. Tried 3 outfits.", PNG)
        assert out["id"] == "99" and out["media_id"] == "42"
        upload, tweet = x.seen
        assert upload["path"] == "/2/media/upload" and upload["oauth_ok"]
        assert upload["category"] == "tweet_image" and upload["filename"] == "card.png"
        assert upload["content_type"] == "image/png" and upload["bytes"] == PNG
        assert tweet["path"] == "/2/tweets" and tweet["oauth_ok"]
        assert tweet["content_type"] == "application/json"
        assert tweet["json"] == {"text": "🪰 Day 1. Tried 3 outfits.",
                                 "media": {"media_ids": ["42"]}}
        # wrong credentials are refused by the server and reported without the secret
        bad = {**CREDS, "access_secret": "not-the-secret"}
        with pytest.raises(X.XPostError, match="HTTP 401") as info:
            X.XPoster(bad, budget)("hi", PNG)
        assert "not-the-secret" not in str(info.value)
        assert budget.count == 2


# ------------------------------------------------------------------ poster_for


def test_poster_for_needs_credentials_and_budget(_data_dir):
    env = {"FLY_X_API_KEY": "k", "FLY_X_API_SECRET": "s", "FLY_X_ACCESS_TOKEN": "t",
           "FLY_X_ACCESS_SECRET": "ts"}
    assert X.poster_for(Cfg(), {}) == (None, "no_credentials")
    assert X.poster_for(Cfg(), {**env, "FLY_X_API_SECRET": ""}) == (None, "no_credentials")
    poster, why = X.poster_for(Cfg(), env, month="2026-09")
    assert isinstance(poster, X.XPoster) and why is None
    assert poster.budget.network == "testnet" and poster.budget.budget == 40
    assert not poster.budget.path.exists()  # reading the counter writes nothing
    poster, why = X.poster_for(Cfg(budget=0), env, month="2026-09")
    assert poster is None and why.startswith("x_budget: 0 of 0 posts used in 2026-09")
    X.MonthlyBudget("testnet", 1, month="2026-09").charge()
    poster, why = X.poster_for(Cfg(budget=1), env, month="2026-09")
    assert poster is None and why.startswith("x_budget: 1 of 1")
    assert X.poster_for(Cfg(budget=2), env, month="2026-09")[1] is None
    # a config without the field means the spec's default of 40
    poster, _ = X.poster_for(type("C", (), {"network": "testnet"})(), env, month="2026-09")
    assert poster.budget.budget == 40
    assert paths.network_dir("testnet") / "x-posts-2026-09.json" == poster.budget.path
