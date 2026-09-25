"""A fake of the LFG public API as the fly uses it (docs/lfg-api.md), for tests.

Shared by the client, loop, setup and reconcile tests. pytest puts ``tests/`` on
``sys.path`` in its default import mode, so a test module binds the fixture with::

    import fake_lfg as FL
    fake_lfg = FL.fake_lfg      # yields (base_url, state); a plain assignment keeps F811 quiet

``FL.ProofSigner(wallet, account=...)`` and ``FL.FakeLedger()`` stand in for
``body.signer.Signer`` / ``body.chain.Ledger`` in ``LfgClient.sign_in`` with a real
xrpl ``Wallet``, so the fake's signature check is exercised for real.

Design notes:

- The venv has no pytest-asyncio, so the server runs in a background thread with
  its own event loop (``FakeLfgServer``); tests drive the client with
  ``asyncio.run``. For tests that run their own loop, ``make_app(state)`` builds
  the aiohttp application directly.
- All state is one mutable ``FakeLfgState``. Tests configure it before the calls
  and inspect it after; every request is logged in ``state.requests``.
- Sign-in verifies the proof the way LFG does (docs/lfg-api.md §1): the closed
  field allowlist, Destination/Amount/Fee/Sequence/LastLedgerSequence/SourceTag,
  the memos decoded exactly, the signing key (master, or the account's registered
  RegularKey in ``state.regular_keys``), and the signature via
  ``xrpl.core.keypairs.is_valid_message`` over ``encode_for_signing``.
- Sign requests (§8) are rows in ``state.sign_requests``; ``POST .../result`` with a
  hash resolves a row and runs its side effect: a mint session advances
  awaiting_payment -> offer_ready on its payment and -> done on its accept, the
  Closet becomes claimable, the BRIX line becomes set, a pending offer is delivered.
  ``state.sign_result_stalls`` / ``state.sign_result_always`` force 202/503 first.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from xrpl.core import keypairs
from xrpl.core.binarycodec import encode_for_signing
from xrpl.core.keypairs import derive_classic_address, is_valid_message
from xrpl.utils import str_to_hex
from xrpl.wallet import Wallet

SOURCE_TAG = 2606160021
PROOF_DESTINATION = "rrrrrrrrrrrrrrrrrNAMEtxvNvQ"
PROOF_AMOUNT = "1"
MAX_PROOF_FEE_DROPS = 10_000
PROOF_LLS_WINDOW = 900
SIGNIN_TTL = 300
SIGN_TTL = 900
MINT_PRICE_XRP = "10"
BULK_MINT_MAX = 10
TF_FULLY_CANONICAL = 0x80000000
MEMO_FORMAT_HEX = str_to_hex("text/plain")
NONCE_MEMO_TYPE = "lfg/nonce"

SLOTS = ("Background", "Back", "Body", "Clothing", "Mouth", "Eyebrows", "Eyes", "Head", "Accessory")
NON_BODY_SLOTS = tuple(s for s in SLOTS if s != "Body")

# A 1x1 transparent PNG, what /api/layer serves on a hit.
PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)

_HEX_RE = re.compile(r"[0-9A-Fa-f]+")
_TX_HASH_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
_PROOF_ALLOWED = {
    "TransactionType", "Account", "Destination", "Amount", "Fee", "Sequence",
    "LastLedgerSequence", "SourceTag", "Memos", "SigningPubKey", "TxnSignature", "Flags",
    "NetworkID",
}
_RESPONSE_ARTIFACTS = {
    "hash", "ctid", "date", "ledger_index", "inLedger", "validated", "meta", "close_time_iso",
    "status",
}


class ProofError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------ sign-in doubles


class FakeLedger:
    """Only what `LfgClient.sign_in` needs from body/chain.Ledger."""

    def __init__(self, index: int = 1000):
        self.index = index
        self.calls = 0

    async def validated_ledger_index(self) -> int:
        self.calls += 1
        return self.index


class ProofSigner:
    """A body/signer.Signer stand-in that really signs the proof with an xrpl Wallet.

    `account` is the wallet whose RegularKey this key is (the default is the key's
    own address, i.e. a master-key proof). `tamper=True` breaks the signature after
    signing. Every call is recorded in `seen` as (tx_json, source_tag).
    """

    def __init__(self, wallet: Wallet | None = None, account: str | None = None,
                 tamper: bool = False):
        self.wallet = wallet or Wallet.create()
        self.account = account or self.wallet.address
        self.address = self.wallet.address
        self.public_key = self.wallet.public_key
        self.tamper = tamper
        self.seen: list[tuple[dict, int]] = []

    def sign_proof(self, tx_json: dict, source_tag: int) -> dict:
        self.seen.append((dict(tx_json), source_tag))
        tx = dict(tx_json)
        tx["SigningPubKey"] = self.wallet.public_key
        tx["TxnSignature"] = keypairs.sign(bytes.fromhex(encode_for_signing(tx)),
                                           self.wallet.private_key)
        if self.tamper:
            tx["LastLedgerSequence"] += 1  # the signature no longer covers the tx
        return tx


# ------------------------------------------------------------------ memos


def build_memos(initiator: str, platform: str, action: str) -> list[dict]:
    """LFG's `build_memos_json`: hex-encoded, lower-case, MemoFormat text/plain."""
    return [
        {"Memo": {"MemoType": str_to_hex(k), "MemoData": str_to_hex(v),
                  "MemoFormat": MEMO_FORMAT_HEX}}
        for k, v in (("initiator", initiator), ("platform", platform), ("action", action))
    ]


def decode_memos(value: Any) -> dict[str, str] | None:
    if not isinstance(value, list):
        return None
    out: dict[str, str] = {}
    try:
        for entry in value:
            body = entry.get("Memo") if isinstance(entry, dict) else None
            if not isinstance(body, dict):
                return None
            key = bytes.fromhex(str(body.get("MemoType"))).decode()
            data = bytes.fromhex(str(body.get("MemoData"))).decode()
            if key in out:
                return None
            out[key] = data
    except (ValueError, UnicodeDecodeError, TypeError):
        return None
    return out


# ------------------------------------------------------------------ state


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def _pad_attributes(traits: dict[str, str] | list[dict], body: str | None) -> list[dict]:
    if isinstance(traits, list):
        traits = {a["trait_type"]: a["value"] for a in traits}
    got = dict(traits)
    if body and "Body" not in got:
        got["Body"] = body
    return [{"trait_type": s, "value": got.get(s, "None")} for s in SLOTS]


@dataclass
class FakeLfgState:
    """Everything the fake serves and remembers. Mutate freely from a test."""

    SLOTS = SLOTS

    # gates and identity
    agent_enabled: bool = True
    economy_enabled: bool = True
    network: str = "testnet"
    validated_ledger: int = 1000  # created_ledger of proof rows
    proof_lls_window: int = PROOF_LLS_WINDOW
    signin_rate_limit: int | None = None  # None = off; N = 429 after N starts
    signing_account: str = field(default_factory=lambda: Wallet.create().address)
    brix_issuer: str = field(default_factory=lambda: Wallet.create().address)
    brix_currency: str = "4252495800000000000000000000000000000000"
    # keys: account -> its validated RegularKey address; accounts with master disabled
    regular_keys: dict[str, str] = field(default_factory=dict)
    master_disabled: set[str] = field(default_factory=set)
    key_lookup_ok: bool = True  # False -> 503 regular_key_unverified for RegularKey proofs
    key_revoked: bool = False  # True -> every bearer request 401 key_revoked
    # sessions
    tokens: dict[str, str] = field(default_factory=dict)  # token -> wallet (live ones)
    issued_tokens: list[str] = field(default_factory=list)  # every token ever issued
    sessions: list[dict] = field(default_factory=list)  # issued tokens' claims
    logouts: int = 0
    logout_error: tuple[int, dict] | None = None
    proofs: list[dict] = field(default_factory=list)  # accepted proof tx_json
    proof_rows: list[dict] = field(default_factory=list)
    proof_errors: list[str] = field(default_factory=list)
    signin_starts: int = 0
    # wardrobe
    characters: list[dict] = field(default_factory=list)
    closet_assets: list[dict] = field(default_factory=list)  # [{slot, value, count}]
    closet_token: dict = field(
        default_factory=lambda: {"status": "active", "nft_id": "C" * 64}
    )
    closet_error: tuple[int, dict] | None = None
    listed: set[tuple[str, str]] = field(default_factory=set)  # Closet Market asks
    swap_fee: dict | None = field(
        default_factory=lambda: {"pay_with": "BRIX", "amount": "5", "per_nft": True}
    )
    z_order: dict = field(
        default_factory=lambda: {
            "layers": {s: i for i, s in enumerate(SLOTS)},
            "z_overrides": [],
        }
    )
    trait_tokens: list[dict] = field(default_factory=list)
    # layers: None = every (body, trait, value) resolves; else only the members
    layers: set[tuple[str, str, str]] | None = None
    # equip / harvest
    equip_outcome: str = "done"  # done | failed_reverted | failed_uncertain | hang
    equip_running_polls: int = 1  # GETs that report "running" before the outcome
    equip_sessions: dict[str, dict] = field(default_factory=dict)
    harvest_outcome: str = "done"  # done | failed | hang
    harvest_running_polls: int = 1
    harvest_sessions: dict[str, dict] = field(default_factory=dict)
    # sign requests
    sign_requests: dict[str, dict] = field(default_factory=dict)
    sign_ttl: float = SIGN_TTL
    sign_result_stalls: list[int] = field(default_factory=list)  # per-call 202/503 first
    sign_result_always: int | None = None  # 202 or 503 forever
    # mint
    mint_sessions: dict[str, dict] = field(default_factory=dict)
    mint_queue: list[dict] = field(default_factory=list)  # nft_id, body_type, traits, ...
    mint_refusal: tuple[int, dict] | None = None
    mint_pay_with: str = "XRP"
    mint_price_xrp: str = MINT_PRICE_XRP
    collection_headroom: int = 100
    bulk_jobs: dict[str, dict] = field(default_factory=dict)
    minted: int = 0
    # offers
    pending_offers: list[dict] = field(default_factory=list)
    # BRIX
    brix: dict = field(
        default_factory=lambda: {
            "claimable": 0, "unlisted_last_epoch": 0, "accrued_total": 0,
            "claimed_total": 0, "open_claim": None, "last_epoch": None,
        }
    )
    brix_trustline_set: bool = False
    brix_claims_enabled: bool = True
    brix_claim_outcome: str = "confirmed"  # confirmed | submitted | failed
    brix_claims: dict[str, dict] = field(default_factory=dict)
    # rarity
    rarity_supply: dict = field(
        default_factory=lambda: {"network": "testnet", "as_of": 0, "n_live": 0, "counts": {}}
    )
    rarity: dict[str, dict] = field(default_factory=dict)  # body -> slots payload
    # health
    health: dict = field(
        default_factory=lambda: {
            "ok": True, "active_sessions": 0,
            "detail": {"mint": 0, "swap": 0, "economy": 0, "market": 0},
            "oldest_session_age": 0,
        }
    )
    # log of (method, path_qs)
    requests: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.characters:
            self.add_character(
                "A" * 64, body="male",
                traits={"Background": "Blue", "Clothing": "Hoodie", "Eyes": "Laser",
                        "Head": "Crown"},
            )

    # ------------------------------------------------------------ helpers

    def add_character(self, nft_id: str, *, body: str = "male", traits=None,
                      mutable: bool = True, blank: bool = False, edition: int | None = None
                      ) -> str:
        self.characters.append(
            {
                "nft_id": nft_id, "edition": edition or len(self.characters) + 1, "body": body,
                "mutable": mutable, "blank": blank,
                "image_url": f"https://img.test/{nft_id[:8]}.png",
                "video_url": None,
                "attributes": _pad_attributes(traits or {}, None if blank else body),
            }
        )
        return nft_id

    def character(self, nft_id: str) -> dict | None:
        return next((c for c in self.characters if c["nft_id"] == nft_id), None)

    def look(self, nft_id: str) -> dict[str, str]:
        ch = self.character(nft_id)
        assert ch is not None, nft_id
        return {a["trait_type"]: a["value"] for a in ch["attributes"]}

    def add_closet(self, slot: str, value: str, count: int = 1) -> None:
        for a in self.closet_assets:
            if a["slot"] == slot and a["value"] == value:
                a["count"] += count
                return
        self.closet_assets.append({"slot": slot, "value": value, "count": count})

    def closet_count(self, slot: str, value: str) -> int:
        return next(
            (a["count"] for a in self.closet_assets if a["slot"] == slot and a["value"] == value),
            0,
        )

    def _debit_closet(self, slot: str, value: str) -> None:
        for a in self.closet_assets:
            if a["slot"] == slot and a["value"] == value:
                a["count"] -= 1
                if a["count"] <= 0:
                    self.closet_assets.remove(a)
                return

    def new_sign_request(self, txjson: dict, *, wallet: str, purpose: str = "tx",
                         on_signed: tuple | None = None) -> dict:
        """A §8 row; `on_signed` = (kind, key) is the side effect run when a hash lands."""
        now = time.time()
        row = {
            "id": _new_id("wc-"), "wallet": wallet, "purpose": purpose, "txjson": txjson,
            "state": "pending", "created_at": now, "expires_at": now + self.sign_ttl,
            "txid": None, "_on_signed": on_signed,
        }
        self.sign_requests[row["id"]] = row
        return row

    def nft_for(self, ch: dict) -> dict:
        """`swap_meta.normalize_nft`: the /api/nfts shape of an economy character."""
        return {
            "nft_id": ch["nft_id"], "name": f"LFG #{ch['edition']}", "number": ch["edition"],
            "season": 1, "image": ch["image_url"], "video": ch["video_url"], "burn_count": 0,
            "gender": ch["body"], "attributes": list(ch["attributes"]), "blank": ch["blank"],
            "mutable": ch["mutable"], "uri_hex": str_to_hex(f"ipfs://{ch['nft_id'][:16]}"),
        }

    def _resolves(self, body: str, slot: str, value: str) -> bool:
        return self.layers is None or (body, slot, value) in self.layers


# ------------------------------------------------------------------ proof


def _verify_proof(state: FakeLfgState, tx_json: Any, *, nonce: str, created_ledger: int
                  ) -> tuple[str, str]:
    """LFG's `verify_proof` + `_check_signing_key`; returns (account, signer address)."""
    if not isinstance(tx_json, dict):
        raise ProofError("shape")
    if tx_json.get("TransactionType") != "Payment":
        raise ProofError("type")
    if "DeliverMax" in tx_json:
        if "Amount" not in tx_json:
            tx_json = {**tx_json, "Amount": tx_json["DeliverMax"]}
        elif tx_json["DeliverMax"] != tx_json["Amount"]:
            raise ProofError("amount")
    tx_json = {k: v for k, v in tx_json.items() if k not in _RESPONSE_ARTIFACTS | {"DeliverMax"}}
    if set(tx_json) - _PROOF_ALLOWED:
        raise ProofError("extra_field")
    if tx_json.get("Destination") != PROOF_DESTINATION:
        raise ProofError("destination")
    if tx_json.get("Amount") != PROOF_AMOUNT:
        raise ProofError("amount")
    fee = tx_json.get("Fee")
    if not (isinstance(fee, str) and fee.isdigit() and 0 < int(fee) <= MAX_PROOF_FEE_DROPS):
        raise ProofError("fee")
    seq = tx_json.get("Sequence")
    if not (isinstance(seq, int) and not isinstance(seq, bool) and seq > 0):
        raise ProofError("sequence")
    lls = tx_json.get("LastLedgerSequence")
    if not (isinstance(lls, int) and not isinstance(lls, bool) and lls > 0):
        raise ProofError("last_ledger")
    if lls > created_ledger + state.proof_lls_window:
        raise ProofError("last_ledger")
    if tx_json.get("SourceTag") != SOURCE_TAG:
        raise ProofError("source_tag")
    if "Flags" in tx_json and tx_json["Flags"] not in (0, TF_FULLY_CANONICAL):
        raise ProofError("flags")
    if "NetworkID" in tx_json:
        raise ProofError("network_id")
    decoded = decode_memos(tx_json.get("Memos")) or {}
    expected = dict(decode_memos(build_memos("user", "agent", "signin")) or {})
    expected[NONCE_MEMO_TYPE] = nonce
    if decoded.get("action") != "signin":
        raise ProofError("action")
    if decoded.get(NONCE_MEMO_TYPE) != nonce:
        raise ProofError("nonce")
    if decoded != expected:
        raise ProofError("memos")
    account, pub, sig = tx_json.get("Account"), tx_json.get("SigningPubKey"), \
        tx_json.get("TxnSignature")
    if not all(isinstance(x, str) and x for x in (account, pub, sig)):
        raise ProofError("shape")
    if account == PROOF_DESTINATION:
        raise ProofError("account")
    if not (_HEX_RE.fullmatch(pub) and _HEX_RE.fullmatch(sig)):
        raise ProofError("shape")
    pub, sig = pub.upper(), sig.upper()
    try:
        signer = derive_classic_address(pub)
    except Exception as e:
        raise ProofError("pubkey") from e
    # the key rule
    if signer == account:
        if account in state.master_disabled:
            raise ProofError("master_disabled")
    else:
        if not state.key_lookup_ok:
            raise ProofError("regular_key_unverified")
        if state.regular_keys.get(account) != signer:
            raise ProofError("pubkey_account")
    unsigned = {k: v for k, v in tx_json.items() if k != "TxnSignature"}
    unsigned["SigningPubKey"] = pub
    try:
        ok = is_valid_message(bytes.fromhex(encode_for_signing(unsigned)), bytes.fromhex(sig),
                              pub)
    except Exception as e:
        raise ProofError("signature") from e
    if not ok:
        raise ProofError("signature")
    return account, signer


# ------------------------------------------------------------------ app


def _json(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _err(status: int, error: str, code: str | None = None, **extra) -> web.Response:
    body: dict[str, Any] = {"error": error}
    if code:
        body["code"] = code
    body.update(extra)
    return _json(body, status)


def make_app(state: FakeLfgState) -> web.Application:
    app = web.Application(middlewares=[_log_middleware(state)])
    r = app.router

    async def body_of(request: web.Request) -> dict:
        if not request.can_read_body:
            return {}
        try:
            data = await request.json()
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    # ---- auth -------------------------------------------------------------

    def auth(request: web.Request) -> str | web.Response:
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        wallet = state.tokens.get(token)
        if not wallet:
            return _err(401, "unauthorized")
        if state.key_revoked:
            state.tokens.pop(token, None)
            return _err(401, "signing key revoked", "key_revoked")
        return wallet

    def wallet_handler(fn):
        async def handler(request: web.Request) -> web.Response:
            who = auth(request)
            if isinstance(who, web.Response):
                return who
            return await fn(request, who)

        return handler

    # ---- §1 sign-in -------------------------------------------------------

    async def signin_start(request):
        body = await body_of(request)
        if body.get("provider") != "agent":
            return _err(400, "unsupported provider in this fake")
        if not state.agent_enabled:
            return _err(503, "agent sign-in is not enabled", "agent_disabled")
        state.signin_starts += 1
        if state.signin_rate_limit is not None and state.signin_starts > state.signin_rate_limit:
            return _err(429, "too many sign-in attempts", "rate_limited")
        nonce = secrets.token_hex(32)
        now = time.time()
        row = {
            "id": _new_id("wc-"), "wallet": "", "purpose": "signin", "nonce": nonce,
            "state": "pending", "created_at": now, "expires_at": now + SIGNIN_TTL,
            "created_ledger": state.validated_ledger, "provider": "agent", "txjson": None,
            "txid": None, "_on_signed": None,
        }
        state.sign_requests[row["id"]] = row
        state.proof_rows.append(row)
        memos = build_memos("user", "agent", "signin") + [
            {"Memo": {"MemoType": str_to_hex(NONCE_MEMO_TYPE), "MemoData": str_to_hex(nonce)}}
        ]
        return _json({
            "sign_id": row["id"], "nonce": nonce, "source_tag": SOURCE_TAG, "memos": memos,
            "expires_at": row["expires_at"], "provider": "agent",
        })

    async def signin_proof(request):
        body = await body_of(request)
        row = state.sign_requests.get(str(body.get("sign_id") or ""))
        if row is None or row.get("purpose") != "signin":
            return _err(404, "not found")
        if row["state"] == "expired":
            return _err(410, "sign-in expired", "proof_expired")
        if row["state"] != "pending":
            return _err(409, "already used", "proof_replayed")
        if row["expires_at"] < time.time():
            row["state"] = "expired"
            return _err(410, "sign-in expired", "proof_expired")
        try:
            account, signer = _verify_proof(
                state, body.get("tx_json"), nonce=row["nonce"],
                created_ledger=row["created_ledger"],
            )
        except ProofError as e:
            state.proof_errors.append(e.reason)
            if e.reason == "regular_key_unverified":
                return _err(503, "could not check the signing key on-ledger; try again",
                            "regular_key_unverified")
            return _err(400, "bad proof", "bad_proof")
        row["state"] = "consumed"
        row["wallet"] = account
        state.proofs.append(body["tx_json"])
        token = "tok-" + secrets.token_hex(16)
        state.issued_tokens.append(token)  # so a test can scan the disk for it after logout
        claims = {
            "id": account, "name": account[:8], "platform": "web", "provider": "agent",
            "exp": time.time() + 21600, "jti": uuid.uuid4().hex, "wallet": account,
        }
        if signer != account:
            claims["key"] = "regular"
            claims["signer"] = signer
        state.tokens[token] = account
        state.sessions.append(claims)
        return _json({
            "state": "signed", "wallet": account, "session_token": token,
            "user": {"id": account, "username": claims["name"]},
        })

    @wallet_handler
    async def logout(request, wallet):
        if state.logout_error:
            return _json(state.logout_error[1], state.logout_error[0])
        token = request.headers["Authorization"][7:]
        state.tokens.pop(token, None)
        state.logouts += 1
        return _json({"ok": True})

    @wallet_handler
    async def me(request, wallet):
        return _json({"id": wallet, "username": wallet[:8], "wallet": wallet})

    # ---- §2 reads ---------------------------------------------------------

    @wallet_handler
    async def nfts(request, wallet):
        return _json({
            "nfts": [state.nft_for(c) for c in state.characters],
            "swappable_traits": list(NON_BODY_SLOTS), "swap_fee": state.swap_fee,
            "swap_matrix": {},
        })

    @wallet_handler
    async def economy(request, wallet):
        if not state.economy_enabled:
            return _err(403, "economy is disabled", "economy_disabled")
        return _json({
            "characters": [dict(c) for c in state.characters],
            "closet": {"assets": [dict(a) for a in state.closet_assets],
                       "token": dict(state.closet_token)},
            "trait_order": list(SLOTS), "z_order": state.z_order, "slots": list(NON_BODY_SLOTS),
            "trait_tokens": list(state.trait_tokens),
        })

    async def rarity_supply(request):
        return _json(state.rarity_supply)

    async def rarity(request):
        body = request.query.get("body")
        if not body:
            return _err(400, "bad params")
        payload = state.rarity.get(body) or {"slots": {}, "stale_slots": [], "generated_at": 0}
        return _json({"body": body, **payload})

    async def layer(request):
        q = request.query
        body, trait, value = q.get("body"), q.get("trait"), q.get("value")
        if not (body and trait and value):
            return _err(400, "bad layer params")
        if state._resolves(body, trait, value):
            return web.Response(body=PNG_1x1, content_type="image/png")
        return web.Response(status=404, body=b"", content_type="image/png")

    # ---- §3 equip ---------------------------------------------------------

    def _running(sessions: dict) -> bool:
        return any(s["state"] == "running" for s in sessions.values())

    @wallet_handler
    async def equip(request, wallet):
        body = await body_of(request)
        nft_id = body.get("nft_id")
        changes = body.get("changes")
        if not isinstance(nft_id, str) or not nft_id:
            return _err(400, "missing or invalid field: 'nft_id'")
        if not isinstance(changes, list) or not changes or len(changes) > 8:
            return _err(400, "missing or invalid field: 'changes'")
        slots = [c.get("slot") for c in changes]
        if len(set(slots)) != len(slots):
            return _err(400, "duplicate slot in changes")
        if _running(state.equip_sessions) or _running(state.harvest_sessions):
            return _err(409, "an economy action is already in progress")
        ch = state.character(nft_id)
        if ch is None:
            return _err(400, "nft is not owned by this wallet")
        if ch["blank"]:
            return _err(409, "blank character", "blank_character")
        if not ch["mutable"]:
            return _err(400, "cannot equip: character is not mutable")
        debit: dict[tuple[str, str], int] = {}
        for c in changes:
            slot, value = c.get("slot"), c.get("value")
            if slot not in NON_BODY_SLOTS or not isinstance(value, str):
                return _err(400, f"bad change {c!r}")
            if (slot, value) in state.listed:
                return _err(400, f"{value} is listed in your Closet market")
            debit[(slot, value)] = debit.get((slot, value), 0) + 1
            if state.closet_count(slot, value) < debit[(slot, value)]:
                return _err(400, f"the Closet does not hold {slot}={value}")
            if value != "None" and not state._resolves(ch["body"], slot, value):
                return _err(400, f"'{value}' does not fit a {ch['body']} body")
        sid = _new_id("eq-")
        session = {
            "id": sid, "kind": "equip", "state": "running", "error": None, "displaced": [],
            "resolution": None, "platform": "web",
            "_nft_id": nft_id, "_changes": [dict(c) for c in changes],
            "_polls": state.equip_running_polls, "_outcome": state.equip_outcome,
        }
        state.equip_sessions[sid] = session
        return _json(_public(session))

    def _settle_equip(s: dict) -> None:
        outcome = s["_outcome"]
        if outcome == "hang":
            return
        if s["_polls"] > 0:
            s["_polls"] -= 1
            return
        if outcome == "done":
            ch = state.character(s["_nft_id"])
            assert ch is not None
            for c in s["_changes"]:
                slot, value = c["slot"], c["value"]
                for a in ch["attributes"]:
                    if a["trait_type"] == slot:
                        s["displaced"].append({"slot": slot, "value": a["value"]})
                        state.add_closet(slot, a["value"])
                        a["value"] = value
                state._debit_closet(slot, value)
            s["state"], s["resolution"] = "done", "committed"
        elif outcome == "failed_reverted":
            s["state"], s["resolution"], s["error"] = "failed", "reverted", "modify failed"
        elif outcome == "failed_uncertain":
            s["state"], s["resolution"], s["error"] = "failed", "uncertain", "sync indeterminate"
        else:
            raise ValueError(f"unknown equip_outcome {outcome!r}")

    @wallet_handler
    async def equip_status(request, wallet):
        s = state.equip_sessions.get(request.match_info["sid"])
        if s is None:
            return _err(404, "not found")
        if s["state"] == "running":
            _settle_equip(s)
        return _json(_public(s))

    # ---- §4 harvest -------------------------------------------------------

    @wallet_handler
    async def harvest(request, wallet):
        body = await body_of(request)
        nft_id = body.get("nft_id")
        if not isinstance(nft_id, str) or not nft_id:
            return _err(400, "missing or invalid field: 'nft_id'")
        if state.closet_token.get("status") != "active":
            return _err(400, "Create and claim your Closet first.")
        ch = state.character(nft_id)
        if ch is None:
            return _err(400, "nft is not owned by this wallet")
        if ch["blank"]:
            return _err(409, "blank character", "blank_character")
        if not ch["mutable"]:
            return _err(400, "cannot harvest: character is not mutable")
        if _running(state.equip_sessions):
            return _err(409, "an economy action is already in progress")
        sid = _new_id("hv-")
        session = {
            "id": sid, "kind": "harvest", "state": "running", "error": None, "moved_assets": [],
            "accept": None, "accept_push": None, "new_nft_id": None, "platform": "web",
            "_nft_id": nft_id, "_polls": state.harvest_running_polls,
            "_outcome": state.harvest_outcome,
        }
        state.harvest_sessions[sid] = session
        return _json(_public(session))

    def _settle_harvest(s: dict) -> None:
        outcome = s["_outcome"]
        if outcome == "hang":
            return
        if s["_polls"] > 0:
            s["_polls"] -= 1
            return
        if outcome == "done":
            ch = state.character(s["_nft_id"])
            assert ch is not None
            for a in ch["attributes"]:
                s["moved_assets"].append([a["trait_type"], a["value"]])
                state.add_closet(a["trait_type"], a["value"])
                a["value"] = "None"
            ch["blank"] = True
            s["state"] = "done"
        elif outcome == "failed":
            s["state"], s["error"] = "failed", "modify failed"
        else:
            raise ValueError(f"unknown harvest_outcome {outcome!r}")

    @wallet_handler
    async def harvest_status(request, wallet):
        s = state.harvest_sessions.get(request.match_info["sid"])
        if s is None:
            return _err(404, "not found")
        if s["state"] == "running":
            _settle_harvest(s)
        return _json(_public(s))

    # ---- §5 closet --------------------------------------------------------

    @wallet_handler
    async def closet(request, wallet):
        if state.closet_error:
            return _json(state.closet_error[1], state.closet_error[0])
        tok = state.closet_token
        if tok.get("status") == "active":
            return _json({"status": "active", "nft_id": tok["nft_id"], "accept": None,
                          "accept_push": None})
        if tok.get("status") == "pending_accept" and tok.get("_accept_row"):
            row = state.sign_requests[tok["_accept_row"]]
            if row["state"] == "signed":
                tok.clear()
                tok.update({"status": "active", "nft_id": row["_nft_id"]})
                return _json({"status": "active", "nft_id": tok["nft_id"], "accept": None,
                              "accept_push": None})
            if row["state"] == "pending":
                return _json({"status": "pending_accept", "nft_id": row["_nft_id"],
                              "accept": f"lfg-wc://{row['id']}", "accept_push": None})
        nft_id = tok.get("nft_id") or "C" * 64
        offer = _new_id("").upper()[:64].ljust(64, "0")
        row = state.new_sign_request(
            {"TransactionType": "NFTokenAcceptOffer", "Account": wallet,
             "NFTokenSellOffer": offer, "SourceTag": SOURCE_TAG,
             "Memos": build_memos("user", "agent", "accept-offer")},
            wallet=wallet, on_signed=("closet", nft_id),
        )
        row["_nft_id"] = nft_id
        tok.clear()
        tok.update({"status": "pending_accept", "nft_id": nft_id, "_accept_row": row["id"]})
        return _json({"status": "pending_accept", "nft_id": nft_id,
                      "accept": f"lfg-wc://{row['id']}", "accept_push": None})

    # ---- §6 BRIX ----------------------------------------------------------

    @wallet_handler
    async def brix(request, wallet):
        return _json({"wallet": wallet, **state.brix})

    @wallet_handler
    async def brix_claim(request, wallet):
        if not state.brix_claims_enabled:
            return _err(503, "claims are disabled", "claims_disabled")
        if not state.brix_trustline_set:
            return _err(409, "set the BRIX trustline first", "trustline_required")
        if state.brix.get("open_claim"):
            return _err(409, "a claim is in flight", "claim_in_flight")
        amount = state.brix.get("claimable") or 0
        if not amount:
            return _err(400, "nothing to claim", "nothing_to_claim")
        cid = _new_id("claim-")
        outcome = state.brix_claim_outcome
        claim = {"claim_id": cid, "state": outcome, "amount": amount,
                 "tx_hash": secrets.token_hex(32).upper() if outcome != "failed" else None}
        state.brix_claims[cid] = claim
        if outcome == "confirmed":
            state.brix["claimed_total"] = state.brix.get("claimed_total", 0) + amount
            state.brix["claimable"] = 0
        elif outcome == "submitted":
            state.brix["open_claim"] = {"claim_id": cid, "state": outcome,
                                        "tx_hash": claim["tx_hash"]}
        return _json(claim)

    @wallet_handler
    async def brix_claim_status(request, wallet):
        claim = state.brix_claims.get(request.match_info["cid"])
        if claim is None:
            return _err(404, "not found")
        return _json(claim)

    @wallet_handler
    async def brix_trustline(request, wallet):
        if state.brix_trustline_set:
            return _json({"state": "already_set"})
        row = state.new_sign_request(
            {"TransactionType": "TrustSet", "Account": wallet, "Flags": 131072,
             "LimitAmount": {"currency": state.brix_currency, "issuer": state.brix_issuer,
                             "value": "1000000000"},
             "SourceTag": SOURCE_TAG, "Memos": build_memos("user", "agent", "trustset")},
            wallet=wallet, on_signed=("trustline", None),
        )
        return _json({"state": "pending", "uuid": row["id"], "xumm_url": f"lfg-wc://{row['id']}",
                      "qr_png": None, "expires_at": row["expires_at"]})

    @wallet_handler
    async def brix_trustline_status(request, wallet):
        row = state.sign_requests.get(request.match_info["uuid"])
        if row is None or row["wallet"] != wallet:
            return _err(404, "not found")
        st = row["state"]
        if st == "signed":
            return _json({"state": "signed", "tx_hash": row["txid"]})
        if st == "rejected":
            return _json({"state": "rejected", "code": "user_rejected"})
        if st == "failed":
            return _json({"state": "rejected", "code": "failed"})
        if st == "expired" or row["expires_at"] < time.time():
            return _json({"state": "expired"})
        return _json({"state": "pending"})

    # ---- §7 mint ----------------------------------------------------------

    def _next_minted(wallet: str) -> dict:
        spec = state.mint_queue.pop(0) if state.mint_queue else {}
        state.minted += 1
        number = spec.get("nft_number") or 1000 + state.minted
        nft_id = spec.get("nft_id") or f"{state.minted:064X}"
        return {
            "nft_number": number, "nft_id": nft_id,
            "image_url": spec.get("image_url") or f"https://img.test/{number}.png",
            "video_url": spec.get("video_url"), "body_type": spec.get("body_type") or "male",
            "traits": spec.get("traits") or {"Head": "Cap"},
        }

    def _payment_row(wallet: str, quantity: int, on_signed: tuple) -> dict:
        drops = str(int(state.mint_price_xrp) * 1_000_000 * quantity)
        return state.new_sign_request(
            {"TransactionType": "Payment", "Account": wallet,
             "Destination": state.signing_account, "Amount": drops, "SourceTag": SOURCE_TAG,
             "Memos": build_memos("user", "agent", "mint")},
            wallet=wallet, on_signed=on_signed,
        )

    def _active_mint(wallet: str) -> dict | None:
        for s in state.mint_sessions.values():
            if s["_wallet"] == wallet and s["state"] not in (
                "done", "failed", "payment_timeout", "cancelled"
            ):
                return s
        return None

    @wallet_handler
    async def mint(request, wallet):
        body = await body_of(request)
        if state.mint_refusal:
            return _json(state.mint_refusal[1], state.mint_refusal[0])
        if _active_mint(wallet) is not None:
            return _err(409, "mint already in progress")
        if state.collection_headroom <= 0:
            return _err(409, "collection full", "collection_full")
        sid = _new_id("mint-")
        session = {
            "id": sid, "platform": "web", "state": "awaiting_payment", "error": None,
            "reason": None, "sponsored": False, "sponsorship_reason": None,
            "pay_with": state.mint_pay_with, "pay_amount": state.mint_price_xrp,
            "payment_link": None, "payment_push": None, "qr_scanned": False,
            "accept_scanned": False, "accept_signed": False, "nft_number": None, "nft_id": None,
            "image_url": None, "video_url": None, "traits": None, "body_type": None,
            "accept_qr_url": None, "accept_deeplink": None, "accept_push": None,
            "_wallet": wallet, "ref": body.get("ref"),
        }
        row = _payment_row(wallet, 1, ("mint_payment", sid))
        session["payment_link"] = f"lfg-wc://{row['id']}"
        state.mint_sessions[sid] = session
        return _json(_public(session, drop=("ref",)))

    @wallet_handler
    async def mint_status(request, wallet):
        s = state.mint_sessions.get(request.match_info["sid"])
        if s is None or s["_wallet"] != wallet:
            return _err(404, "not found")
        return _json(_public(s, drop=("ref",)))

    @wallet_handler
    async def mint_active(request, wallet):
        s = _active_mint(wallet)
        if s is None:
            return _err(404, "no active mint")
        return _json(_public(s, drop=("ref",)))

    @wallet_handler
    async def bulk_mint(request, wallet):
        body = await body_of(request)
        if state.mint_refusal:
            return _json(state.mint_refusal[1], state.mint_refusal[0])
        try:
            requested = int(body.get("quantity"))
        except (TypeError, ValueError):
            return _err(400, "missing or invalid field: 'quantity'")
        if requested < 1:
            return _err(400, "missing or invalid field: 'quantity'")
        qty = min(requested, BULK_MINT_MAX, state.collection_headroom)
        if qty < 1:
            return _err(409, "collection full", "collection_full")
        jid = _new_id("bulk-")
        job = {
            "id": jid, "platform": "web", "state": "awaiting_payment", "error": None,
            "requested_qty": requested, "quantity": qty, "pay_with": state.mint_pay_with,
            "pay_amount": str(int(state.mint_price_xrp) * qty), "payment_link": None,
            "network": state.network, "persist_failed": False,
            "units": [
                {"index": i, "state": "pending", "nft_number": None, "nft_id": None,
                 "image_url": None, "offer_id": None, "error": None, "traits": None,
                 "body_type": None, "accepted": False}
                for i in range(qty)
            ],
            "minted": 0, "offered": 0, "_wallet": wallet,
        }
        row = _payment_row(wallet, qty, ("bulk_payment", jid))
        job["payment_link"] = f"lfg-wc://{row['id']}"
        state.bulk_jobs[jid] = job
        return _json(_public(job))

    @wallet_handler
    async def bulk_status(request, wallet):
        j = state.bulk_jobs.get(request.match_info["jid"])
        if j is None or j["_wallet"] != wallet:
            return _err(404, "not found")
        return _json(_public(j))

    @wallet_handler
    async def bulk_unit_accept(request, wallet):
        j = state.bulk_jobs.get(request.match_info["jid"])
        if j is None or j["_wallet"] != wallet:
            return _err(404, "not found")
        try:
            unit = j["units"][int(request.match_info["index"])]
        except (ValueError, IndexError):
            return _err(404, "no such unit")
        if unit["state"] != "offered" or not unit["offer_id"]:
            return _err(409, "unit has no offer to accept", "not_offered")
        row = state.new_sign_request(
            {"TransactionType": "NFTokenAcceptOffer", "Account": wallet,
             "NFTokenSellOffer": unit["offer_id"], "SourceTag": SOURCE_TAG,
             "Memos": build_memos("user", "agent", "accept-offer")},
            wallet=wallet, on_signed=("bulk_accept", (j["id"], unit["index"])),
        )
        return _json({"qr": None, "link": f"lfg-wc://{row['id']}", "push": None})

    # ---- §8 sign requests -------------------------------------------------

    @wallet_handler
    async def sign_request(request, wallet):
        row = state.sign_requests.get(request.match_info["sid"])
        if row is None or row.get("purpose") != "tx" or row.get("wallet") != wallet:
            return _err(404, "not found")
        if row["state"] == "pending" and row["expires_at"] < time.time():
            row["state"] = "expired"
        return _json({"id": row["id"], "state": row["state"], "txjson": row["txjson"],
                      "expires_at": row["expires_at"], "txid": row["txid"]})

    @wallet_handler
    async def sign_result(request, wallet):
        row = state.sign_requests.get(request.match_info["sid"])
        if row is None or row.get("purpose") != "tx":
            return _err(404, "not found")
        if row.get("wallet") != wallet:
            return _err(403, "not your request", "not_your_request")
        body = await body_of(request)
        tx_hash = body.get("hash")
        if tx_hash is not None:
            if not isinstance(tx_hash, str) or not _TX_HASH_RE.match(tx_hash):
                return _err(400, "hash must be 64 hex characters", "bad_request")
            tx_hash = tx_hash.upper()
        elif not (body.get("rejected") or body.get("error")):
            return _err(400, "one of hash/rejected/error is required", "bad_request")
        if row["state"] == "pending" and row["expires_at"] + 60 < time.time():
            row["state"] = "expired"
        if row["state"] == "expired":
            return _err(410, "request expired", "expired")
        if row["state"] != "pending":
            if tx_hash is not None and row.get("txid") == tx_hash:
                return _json({"state": row["state"], "txid": row["txid"]})
            return _err(409, "already resolved", "already_resolved", state=row["state"])
        if tx_hash is None:
            if body.get("rejected"):
                row["state"] = "rejected"
                return _json({"state": "rejected"})
            row["state"] = "failed"
            row["error"] = str(body.get("error"))[:200]
            return _json({"state": "failed"})
        stall = state.sign_result_always
        if stall is None and state.sign_result_stalls:
            stall = state.sign_result_stalls.pop(0)
        if stall == 202:
            return _json({"state": "pending", "code": "tx_not_found"}, 202)
        if stall == 503:
            return _err(503, "could not reach the ledger", "ledger_unavailable")
        row["state"], row["txid"] = "signed", tx_hash
        _on_signed(row, wallet)
        return _json({"state": "signed", "txid": tx_hash})

    def _on_signed(row: dict, wallet: str) -> None:
        hook = row.get("_on_signed")
        if not hook:
            return
        kind, key = hook
        if kind == "mint_payment":
            s = state.mint_sessions[key]
            s.update(_next_minted(wallet))
            offer = secrets.token_hex(32).upper()
            acc = state.new_sign_request(
                {"TransactionType": "NFTokenAcceptOffer", "Account": wallet,
                 "NFTokenSellOffer": offer, "SourceTag": SOURCE_TAG,
                 "Memos": build_memos("user", "agent", "accept-offer")},
                wallet=wallet, on_signed=("mint_accept", key),
            )
            s["accept_deeplink"] = f"lfg-wc://{acc['id']}"
            s["state"] = "offer_ready"
            state.collection_headroom -= 1
        elif kind == "mint_accept":
            s = state.mint_sessions[key]
            s["accept_signed"], s["state"] = True, "done"
            state.add_character(s["nft_id"], body=s["body_type"], traits=s["traits"],
                                edition=s["nft_number"])
        elif kind == "bulk_payment":
            j = state.bulk_jobs[key]
            for u in j["units"]:
                m = _next_minted(wallet)
                u.update({"state": "offered", "nft_number": m["nft_number"],
                          "nft_id": m["nft_id"], "image_url": m["image_url"],
                          "offer_id": secrets.token_hex(32).upper(), "traits": m["traits"],
                          "body_type": m["body_type"]})
            j["minted"] = j["offered"] = len(j["units"])
            j["state"] = "done"
            state.collection_headroom -= len(j["units"])
        elif kind == "bulk_accept":
            jid, index = key
            u = state.bulk_jobs[jid]["units"][index]
            u["accepted"] = True
            state.add_character(u["nft_id"], body=u["body_type"], traits=u["traits"],
                                edition=u["nft_number"])
        elif kind == "closet":
            pass  # promoted on the next POST /api/closet, as LFG does
        elif kind == "trustline":
            state.brix_trustline_set = True
        elif kind == "offer":
            offer = next((o for o in state.pending_offers if o["offer_index"] == key), None)
            if offer is not None:
                state.pending_offers.remove(offer)
                ch = offer.get("_character")
                if ch:
                    state.add_character(ch["nft_id"], body=ch.get("body", "male"),
                                        traits=ch.get("traits") or {},
                                        mutable=ch.get("mutable", True),
                                        blank=ch.get("blank", False))
        else:
            raise ValueError(f"unknown sign hook {kind!r}")

    # ---- §9 pending offers ------------------------------------------------

    @wallet_handler
    async def pending_offers(request, wallet):
        return _json({"offers": [_public(o) for o in state.pending_offers]})

    @wallet_handler
    async def accept_offer(request, wallet):
        body = await body_of(request)
        idx = body.get("offer_index")
        offer = next((o for o in state.pending_offers if o["offer_index"] == idx), None)
        if offer is None:
            return _err(410, "offer is gone", "offer_gone")
        row = state.new_sign_request(
            {"TransactionType": "NFTokenAcceptOffer", "Account": wallet, "NFTokenSellOffer": idx,
             "SourceTag": SOURCE_TAG, "Memos": build_memos("user", "agent", "accept-offer")},
            wallet=wallet, on_signed=("offer", idx),
        )
        return _json({"qr": None, "link": f"lfg-wc://{row['id']}", "push": None})

    # ---- §10 health -------------------------------------------------------

    async def health(request):
        return _json({**state.health, "active_sessions": len(state.tokens)})

    r.add_post("/api/web/signin", signin_start)
    r.add_post("/api/web/signin/proof", signin_proof)
    r.add_post("/api/logout", logout)
    r.add_get("/api/me", me)
    r.add_get("/api/nfts", nfts)
    r.add_get("/api/economy", economy)
    r.add_get("/api/rarity/supply", rarity_supply)
    r.add_get("/api/rarity", rarity)
    r.add_get("/api/layer", layer)
    r.add_post("/api/equip", equip)
    r.add_get("/api/equip/{sid}", equip_status)
    r.add_post("/api/harvest", harvest)
    r.add_get("/api/harvest/{sid}", harvest_status)
    r.add_post("/api/closet", closet)
    r.add_get("/api/brix", brix)
    r.add_post("/api/brix/claim", brix_claim)
    r.add_get("/api/brix/claim/{cid}", brix_claim_status)
    r.add_post("/api/brix/trustline", brix_trustline)
    r.add_get("/api/brix/trustline/{uuid}", brix_trustline_status)
    r.add_post("/api/mint", mint)
    r.add_get("/api/mint/active", mint_active)
    r.add_post("/api/mint/bulk", bulk_mint)
    r.add_get("/api/mint/bulk/{jid}", bulk_status)
    r.add_post("/api/mint/bulk/{jid}/units/{index}/accept", bulk_unit_accept)
    r.add_get("/api/mint/{sid}", mint_status)
    r.add_get("/api/offers/pending", pending_offers)
    r.add_post("/api/offers/accept", accept_offer)
    r.add_get("/api/sign/{sid}", sign_request)
    r.add_post("/api/sign/{sid}/result", sign_result)
    r.add_get("/api/health", health)
    return app


def _public(obj: dict, drop: tuple[str, ...] = ()) -> dict:
    """Strip the fake's private (underscore) bookkeeping from a served object."""
    return {k: v for k, v in obj.items() if not k.startswith("_") and k not in drop}


def _log_middleware(state: FakeLfgState):
    @web.middleware
    async def mw(request: web.Request, handler):
        state.requests.append((request.method, request.path_qs))
        return await handler(request)

    return mw


# ------------------------------------------------------------------ server


class FakeLfgServer:
    """The fake app on 127.0.0.1:<random>, served from a daemon thread."""

    def __init__(self, state: FakeLfgState | None = None):
        self.state = state or FakeLfgState()
        self.base_url: str = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None

    def start(self) -> FakeLfgServer:
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def serve() -> None:
                runner = web.AppRunner(make_app(self.state))
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                host, port = runner.addresses[0][:2]
                self.base_url = f"http://{host}:{port}"
                self._runner = runner
                ready.set()

            try:
                loop.run_until_complete(serve())
            except BaseException as e:  # noqa: BLE001  (surfaced to the starting thread)
                failure.append(e)
                ready.set()
                return
            loop.run_forever()
            loop.run_until_complete(self._runner.cleanup())
            loop.close()

        self._thread = threading.Thread(target=run, name="fake-lfg", daemon=True)
        self._thread.start()
        ready.wait(10)
        if failure:
            raise failure[0]
        if not self.base_url:
            raise RuntimeError("fake LFG did not start")
        return self

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(10)

    def __enter__(self) -> FakeLfgServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


@pytest.fixture
def fake_lfg() -> Iterator[tuple[str, FakeLfgState]]:
    """(base_url, state) of a fresh fake LFG for one test."""
    with FakeLfgServer() as server:
        yield server.base_url, server.state
