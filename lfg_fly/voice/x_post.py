"""Posting to X (spec §4.5): OAuth 1.0a with the fly's own dev app, and the monthly budget.

`poster_for(cfg)` gives `publish` its `poster` when the four `FLY_X_*` credentials are
set (`post.x_credentials`) and this month's budget (`FLY_X_MONTHLY_BUDGET`, default 40)
still has room; otherwise it gives the reason the post goes to the outbox instead.

The poster does two calls, both signed with OAuth 1.0a HMAC-SHA1 (RFC 5849; the
signature base string covers the query and `oauth_*` parameters only, since neither a
multipart nor a JSON body is signed, §3.4.1.3.1):

1. `POST https://api.x.com/2/media/upload` (multipart: `media` = the card PNG,
   `media_category=tweet_image`) → the media id (`data.id`; the v1.1 shape
   `media_id_string` is read too).
2. `POST https://api.x.com/2/tweets` (JSON `{"text", "media": {"media_ids": [id]}}`) →
   the post id (`data.id`).

These are X API v2's documented shapes at build time; the live API has not been
exercised from this box (the spec's "posts read right" is a rehearsal criterion), so the
endpoints are module constants and a refusal falls back to the outbox in `publish`.

The budget is a per-network, per-month counter file, `FLY_DATA_DIR/<network>/
x-posts-<YYYY-MM>.json`, charged BEFORE the upload so a crash mid-post counts as a post,
never the reverse. Credentials live in the poster object only: no result, log line,
exception message or file ever carries them. HTTP is stdlib `urllib` (synchronous, like
`publish` itself: the loop's day is one post, and nothing else is in flight); the
transport is injectable so tests never touch the network.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

from lfg_fly import paths
from lfg_fly.voice.post import Poster, x_credentials

log = logging.getLogger(__name__)

MEDIA_UPLOAD_URL = "https://api.x.com/2/media/upload"
TWEETS_URL = "https://api.x.com/2/tweets"
MEDIA_CATEGORY = "tweet_image"
TIMEOUT = 30.0
USER_AGENT = "lfg-fly (spec 4.5)"
BUDGET_FILE = "x-posts-{month}.json"
DEFAULT_BUDGET = 40

# (method, url, headers, body) -> (status, body bytes)
Transport = Callable[[str, str, Mapping[str, str], bytes], tuple[int, bytes]]


class XPostError(RuntimeError):
    """X refused, answered nonsense or the wire failed. Never carries a credential."""


class BudgetExhausted(XPostError):
    """This month's FLY_X_MONTHLY_BUDGET is used up; the post goes to the outbox."""


# ---------------------------------------------------------------- OAuth 1.0a (RFC 5849)


def _enc(value: Any) -> str:
    """RFC 5849 §3.6 percent-encoding: everything but ALPHA / DIGIT / "-" / "." / "_" / "~"."""
    return quote(str(value), safe="")


def _base_url(url: str) -> str:
    """Scheme and host lower-cased, default ports dropped, no query or fragment (§3.4.1.2)."""
    u = urlsplit(url)
    scheme, host = u.scheme.lower(), (u.hostname or "").lower()
    port = u.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    return f"{scheme}://{host}{u.path or '/'}"


def oauth1_signature(method: str, url: str, params: Sequence[tuple[str, str]],
                     consumer_secret: str, token_secret: str) -> str:
    """base64(HMAC-SHA1(enc(consumer_secret)&enc(token_secret), base string)) over the
    method, the base URL and the normalized parameters (§3.4.1, §3.4.2)."""
    pairs = sorted((_enc(k), _enc(v)) for k, v in params)
    normalized = "&".join(f"{k}={v}" for k, v in pairs)
    base = "&".join((method.upper(), _enc(_base_url(url)), _enc(normalized)))
    key = f"{_enc(consumer_secret)}&{_enc(token_secret)}".encode()
    return base64.b64encode(hmac.new(key, base.encode(), hashlib.sha1).digest()).decode()


def oauth1_header(method: str, url: str, creds: Mapping[str, str], *,
                  form: Sequence[tuple[str, str]] | None = None, nonce: str | None = None,
                  timestamp: int | None = None) -> str:
    """The `Authorization: OAuth ...` header for one request (§3.5.1).

    `creds` is `post.x_credentials`' dict. The signed parameters are the `oauth_*` set,
    the URL's query and, for an `application/x-www-form-urlencoded` body only, `form`;
    a multipart or JSON body contributes nothing (§3.4.1.3.1).
    """
    oauth = {
        "oauth_consumer_key": creds["api_key"],
        "oauth_nonce": nonce or secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time()) if timestamp is None else timestamp),
        "oauth_token": creds["access_token"],
        "oauth_version": "1.0",
    }
    params: list[tuple[str, str]] = list(oauth.items())
    params += parse_qsl(urlsplit(url).query, keep_blank_values=True)
    params += list(form or ())
    sig = oauth1_signature(method, url, params, creds["api_secret"], creds["access_secret"])
    fields = {**oauth, "oauth_signature": sig}
    return "OAuth " + ", ".join(f'{k}="{_enc(v)}"' for k, v in sorted(fields.items()))


# ---------------------------------------------------------------- the wire


def multipart(fields: Sequence[tuple[str, str]], file_field: str, filename: str,
              content_type: str, data: bytes) -> tuple[str, bytes]:
    """A `multipart/form-data` body: the text fields, then one file part. Returns
    (Content-Type header value, body)."""
    boundary = "----lfg-fly-" + secrets.token_hex(12)
    out = bytearray()
    for name, value in fields:
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n").encode()
    out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").encode()
    out += data
    out += f"\r\n--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(out)


def urllib_transport(method: str, url: str, headers: Mapping[str, str],
                     body: bytes) -> tuple[int, bytes]:
    """The default transport: one blocking HTTP request; an HTTP error is an answer, not
    an exception, so the caller sees X's own message."""
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # noqa: S310 — https constants
            return int(resp.status), resp.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()


# ---------------------------------------------------------------- the monthly budget


class MonthlyBudget:
    """`FLY_X_MONTHLY_BUDGET` (spec §4.5), one counter file per network and UTC month.

    `FLY_DATA_DIR/<network>/x-posts-<YYYY-MM>.json` holds `{network, month, posts: [...]}`;
    `count` is the number of posts charged this month. Like the spend cap, `check` never
    writes and `charge` refuses past the budget (atomic tmp + rename).
    """

    def __init__(self, network: str, budget: int, month: str | None = None):
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
            raise ValueError("budget must be a non-negative int")
        self.network = network
        self.budget = budget
        self.month = month or datetime.now(timezone.utc).strftime("%Y-%m")
        self.path = paths.network_dir(network) / BUDGET_FILE.format(month=self.month)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"network": self.network, "month": self.month, "posts": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("network") != self.network \
                or data.get("month") != self.month:
            raise XPostError(f"{self.path} is not this network's counter for {self.month}")
        if not isinstance(data.get("posts"), list):
            raise XPostError(f"{self.path} is corrupt: posts")
        return data

    @property
    def count(self) -> int:
        return len(self._load()["posts"])

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.count)

    def check(self) -> None:
        count = self.count
        if count >= self.budget:
            raise BudgetExhausted(f"x_budget: {count} of {self.budget} posts used in "
                                  f"{self.month} on {self.network} (FLY_X_MONTHLY_BUDGET)")

    def charge(self, **entry: Any) -> None:
        """Count one post (before it is made). `entry` is what to remember about it; never
        the text itself and never a credential."""
        self.check()
        data = self._load()
        data["posts"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              **entry})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
        try:
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            raise


# ---------------------------------------------------------------- the poster


def _media_id(answer: dict) -> str:
    data = answer.get("data")
    if isinstance(data, dict) and data.get("id"):
        return str(data["id"])
    for key in ("media_id_string", "media_id"):  # the v1.1 shape, should X still answer it
        if answer.get(key):
            return str(answer[key])
    raise XPostError("media upload: X's answer names no media id")


class XPoster:
    """`publish`'s `Poster`: `(text, png) -> {"id", "media_id", "url"}`.

    Charges the month's budget, uploads the card, creates the post. The credentials are
    held here and appear in the OAuth header only; `repr` shows the budget, never them.
    """

    def __init__(self, creds: Mapping[str, str], budget: MonthlyBudget, *,
                 transport: Transport | None = None):
        missing = [k for k in ("api_key", "api_secret", "access_token", "access_secret")
                   if not creds.get(k)]
        if missing:
            raise ValueError(f"X credentials incomplete: {missing}")
        self._creds = {k: str(creds[k]) for k in
                       ("api_key", "api_secret", "access_token", "access_secret")}
        self.budget = budget
        self._transport = transport

    def __repr__(self) -> str:
        return (f"XPoster(network={self.budget.network}, month={self.budget.month}, "
                f"budget={self.budget.budget})")

    __str__ = __repr__

    def _call(self, what: str, url: str, content_type: str, body: bytes) -> dict:
        transport = self._transport if self._transport is not None else urllib_transport
        headers = {"Authorization": oauth1_header("POST", url, self._creds),
                   "Content-Type": content_type, "User-Agent": USER_AGENT}
        try:
            status, raw = transport("POST", url, headers, body)
        except XPostError:
            raise
        except Exception as e:  # noqa: BLE001 — the wire; the message names the failure only
            raise XPostError(f"{what}: {type(e).__name__}: {e}") from e
        snippet = raw.decode("utf-8", "replace")[:300]
        if status // 100 != 2:
            raise XPostError(f"{what}: X answered HTTP {status}: {snippet}")
        try:
            data = json.loads(raw)
        except ValueError:
            raise XPostError(f"{what}: X's answer is not JSON: {snippet}") from None
        return data if isinstance(data, dict) else {}

    def __call__(self, text: str, png: bytes) -> dict:
        self.budget.charge(chars=len(text), png_bytes=len(png))
        content_type, body = multipart(
            [("media_category", MEDIA_CATEGORY), ("media_type", "image/png")],
            "media", "card.png", "image/png", png)
        media_id = _media_id(self._call("media upload", MEDIA_UPLOAD_URL, content_type, body))
        payload = json.dumps({"text": text, "media": {"media_ids": [media_id]}}).encode()
        answer = self._call("post", TWEETS_URL, "application/json", payload)
        post_id = str((answer.get("data") or {}).get("id") or "") \
            if isinstance(answer.get("data"), dict) else ""
        if not post_id:
            raise XPostError("post: X's answer names no post id")
        log.info("posted to X: %s (media %s)", post_id, media_id)
        return {"id": post_id, "media_id": media_id, "url": f"https://x.com/i/status/{post_id}"}


def poster_for(cfg: Any, env: Mapping[str, str] | None = None, *,
               transport: Transport | None = None,
               month: str | None = None) -> tuple[Poster | None, str | None]:
    """What `publish` should do today: `(poster, None)` when the `FLY_X_*` credentials are
    set and the month's budget has room, else `(None, reason)` with the reason the post
    goes to the outbox ("no_credentials", or "x_budget: ..."). Reads the counter only."""
    creds = x_credentials(env)
    if creds is None:
        return None, "no_credentials"
    budget = MonthlyBudget(cfg.network, int(getattr(cfg, "x_monthly_budget", DEFAULT_BUDGET)),
                           month)
    try:
        budget.check()
    except BudgetExhausted as e:
        log.warning("%s; the post goes to the outbox", e)
        return None, str(e)
    return XPoster(creds, budget, transport=transport), None


__all__ = [
    "MEDIA_UPLOAD_URL", "TWEETS_URL", "BudgetExhausted", "MonthlyBudget", "XPostError",
    "XPoster", "multipart", "oauth1_header", "oauth1_signature", "poster_for",
    "urllib_transport",
]
