"""The LFG HTTP client (docs/lfg-api.md; spec §4.1 step 1, §4.4 "what a live session can do").

`LfgClient` speaks LFG's public API as an ordinary agent user: it signs in with a
RegularKey proof (docs/lfg-api.md §1), holds the bearer token in memory only, and logs
out in `__aexit__` no matter how the block ended (spec §4.1 step 1). Every non-2xx
answer raises `LfgError(status, code, body)`. The client never signs anything itself:
the proof goes through `Signer.sign_proof`, and every other signature is a §8 sign
request the caller resolves with `sign_result`.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

PROOF_DESTINATION = "rrrrrrrrrrrrrrrrrNAMEtxvNvQ"
PROOF_AMOUNT = "1"
PROOF_FEE = "12"
PROOF_SEQUENCE = 1
PROOF_LLS_AHEAD = 20
SIGN_LINK_SCHEME = "lfg-wc://"
_SIGN_ID_RE = re.compile(r"^wc-[0-9a-f]{32}$")
_RETRY_STATUSES = (202, 503)


class LfgError(RuntimeError):
    """A non-2xx answer from LFG: `status`, the body's `code` (else its `error`), `body`."""

    def __init__(self, status: int, code: str | None, body: dict | str):
        self.status = status
        self.code = code
        self.body = body
        detail = code or (body if isinstance(body, str) else "")
        super().__init__(f"LFG {status}" + (f" {detail}" if detail else ""))


def _code_of(body: Any) -> str | None:
    if isinstance(body, dict):
        code = body.get("code") or body.get("error")
        return str(code) if code else None
    return None


class LfgClient:
    """One agent session against `api_base` (contract §body/client.py).

    `token` and `wallet` are set by `sign_in`. The token lives in this object only:
    it is never written, logged or repr'd. `_clock`/`_sleep` are `time.time` and
    `asyncio.sleep`, exposed so tests can drive the retry and poll loops quickly.
    """

    SIGN_RESULT_BACKOFF = (1.0, 2.0, 4.0, 8.0)

    def __init__(self, api_base: str, session: aiohttp.ClientSession | None = None,
                 timeout: float = 30.0):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.token: str | None = None
        self.wallet: str | None = None
        self.logout_error: BaseException | None = None
        self._session = session
        self._own_session = session is None
        self._layers: dict[tuple[str, str, str], bool] = {}
        self._clock: Callable[[], float] = time.time
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def __repr__(self) -> str:
        who = f"signed in as {self.wallet}" if self.token else "signed out"
        return f"<LfgClient {self.api_base} {who}>"

    # ------------------------------------------------------------ lifecycle

    async def __aenter__(self) -> LfgClient:
        self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Log out if signed in, always; the token is dropped whatever happens (§4.1)."""
        try:
            if self.token is not None:
                try:
                    await self.logout()
                except (LfgError, aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                    self.logout_error = e
        finally:
            self.token = None
            if self._own_session and self._session is not None:
                await self._session.close()
                self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            self._own_session = True
        return self._session

    # ------------------------------------------------------------ transport

    async def _request(self, method: str, path: str, *, json: Any = None,
                       params: dict | None = None, auth: bool = True) -> tuple[int, Any]:
        """One HTTP exchange: (status, decoded body). Raises nothing on a non-2xx."""
        headers = {}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        session = self._ensure_session()
        async with session.request(method, self.api_base + path, json=json, params=params,
                                   headers=headers) as resp:
            if "json" in resp.headers.get("Content-Type", ""):
                body = await resp.json(content_type=None)
            else:
                body = await resp.text()
            return resp.status, body

    async def _call(self, method: str, path: str, *, json: Any = None,
                    params: dict | None = None, auth: bool = True) -> Any:
        status, body = await self._request(method, path, json=json, params=params, auth=auth)
        if status // 100 != 2:
            raise LfgError(status, _code_of(body), body)
        return body

    # ------------------------------------------------------------ §1 sign-in

    async def sign_in(self, signer, ledger) -> str:
        """Agent sign-in (docs/lfg-api.md §1): start, build the proof, have the signer
        sign it, redeem it for a bearer token. Returns the token (also `self.token`).

        The proof is a never-submitted Payment: Sequence 1 (LFG does not check it),
        Fee "12", LastLedgerSequence = validated + 20 (well inside LFG's 900 window),
        the start response's SourceTag and Memos verbatim (spec §4.3 row 1).
        """
        start = await self._call("POST", "/api/web/signin", json={"provider": "agent"},
                                 auth=False)
        source_tag = int(start["source_tag"])
        validated = await ledger.validated_ledger_index()
        tx = {
            "TransactionType": "Payment",
            "Account": signer.account,
            "Destination": PROOF_DESTINATION,
            "Amount": PROOF_AMOUNT,
            "Fee": PROOF_FEE,
            "Sequence": PROOF_SEQUENCE,
            "LastLedgerSequence": int(validated) + PROOF_LLS_AHEAD,
            "SourceTag": source_tag,
            "Memos": start["memos"],
        }
        signed = signer.sign_proof(tx, source_tag)
        done = await self._call("POST", "/api/web/signin/proof", auth=False,
                                json={"sign_id": start["sign_id"], "tx_json": signed})
        self.token = str(done["session_token"])
        self.wallet = str(done["wallet"])
        return self.token

    async def logout(self) -> None:
        """`POST /api/logout`; the token is forgotten even if the call fails."""
        if self.token is None:
            return
        try:
            await self._call("POST", "/api/logout")
        finally:
            self.token = None

    async def me(self) -> dict:
        return await self._call("GET", "/api/me")

    # ------------------------------------------------------------ §2, §10 reads

    async def health(self) -> dict:
        return await self._call("GET", "/api/health", auth=False)

    async def nfts(self) -> dict:
        return await self._call("GET", "/api/nfts")

    async def economy(self) -> dict:
        return await self._call("GET", "/api/economy")

    async def rarity_supply(self) -> dict:
        return await self._call("GET", "/api/rarity/supply", auth=False)

    async def rarity(self, body: str) -> dict:
        return await self._call("GET", "/api/rarity", params={"body": body}, auth=False)

    async def layer_resolves(self, body: str, slot: str, value: str) -> bool:
        """`GET /api/layer` 200 → True, 404 → False (spec §4.1 step 4); memoized per client."""
        key = (body, slot, value)
        if key in self._layers:
            return self._layers[key]
        session = self._ensure_session()
        params = {"body": body, "trait": slot, "value": value, "thumb": "1"}
        async with session.get(self.api_base + "/api/layer", params=params) as resp:
            if resp.status == 200:
                result = True
            elif resp.status == 404:
                result = False
            else:
                body_text = await resp.text()
                raise LfgError(resp.status, _code_of(_maybe_json(body_text)),
                               _maybe_json(body_text))
        self._layers[key] = result
        return result

    # ------------------------------------------------------------ §3, §4 builder

    async def equip(self, nft_id: str, changes: list[dict]) -> dict:
        return await self._call("POST", "/api/equip", json={"nft_id": nft_id, "changes": changes})

    async def equip_status(self, sid: str) -> dict:
        return await self._call("GET", f"/api/equip/{sid}")

    async def harvest(self, nft_id: str) -> dict:
        return await self._call("POST", "/api/harvest", json={"nft_id": nft_id})

    async def harvest_status(self, sid: str) -> dict:
        return await self._call("GET", f"/api/harvest/{sid}")

    # ------------------------------------------------------------ §5 closet

    async def closet(self) -> dict:
        return await self._call("POST", "/api/closet")

    # ------------------------------------------------------------ §6 BRIX

    async def brix(self) -> dict:
        return await self._call("GET", "/api/brix")

    async def brix_claim(self) -> dict:
        return await self._call("POST", "/api/brix/claim")

    async def brix_claim_status(self, cid: str) -> dict:
        return await self._call("GET", f"/api/brix/claim/{cid}")

    async def brix_trustline(self) -> dict:
        return await self._call("POST", "/api/brix/trustline")

    async def brix_trustline_status(self, uuid: str) -> dict:
        return await self._call("GET", f"/api/brix/trustline/{uuid}")

    # ------------------------------------------------------------ §7 mint

    async def mint(self, ref: str | None = None) -> dict:
        return await self._call("POST", "/api/mint", json={"ref": ref} if ref else {})

    async def mint_status(self, sid: str) -> dict:
        return await self._call("GET", f"/api/mint/{sid}")

    async def mint_active(self) -> dict:
        return await self._call("GET", "/api/mint/active")

    async def bulk_mint(self, quantity: int) -> dict:
        return await self._call("POST", "/api/mint/bulk", json={"quantity": int(quantity)})

    async def bulk_status(self, jid: str) -> dict:
        return await self._call("GET", f"/api/mint/bulk/{jid}")

    async def bulk_unit_accept(self, jid: str, index: int) -> dict:
        return await self._call("POST", f"/api/mint/bulk/{jid}/units/{int(index)}/accept")

    # ------------------------------------------------------------ §9 offers

    async def pending_offers(self) -> dict:
        return await self._call("GET", "/api/offers/pending")

    async def accept_offer(self, offer_index: str) -> dict:
        return await self._call("POST", "/api/offers/accept", json={"offer_index": offer_index})

    # ------------------------------------------------------------ §8 sign requests

    async def sign_request(self, sid: str) -> dict:
        return await self._call("GET", f"/api/sign/{sid}")

    async def sign_result(self, sid: str, *, tx_hash: str | None = None, rejected: bool = False,
                          error: str | None = None) -> dict:
        """`POST /api/sign/{sid}/result` with exactly one of hash / rejected / error.

        A hash is only believed once LFG finds the transaction validated, so 202
        `tx_not_found` and 503 `ledger_unavailable` are retried with backoff until the
        row's `expires_at`; past it the last such answer is raised as `LfgError`.
        """
        given = [x for x in (tx_hash is not None, rejected, error is not None) if x]
        if len(given) != 1:
            raise ValueError("sign_result takes exactly one of tx_hash, rejected, error")
        if tx_hash is not None:
            payload: dict[str, Any] = {"hash": tx_hash}
        elif rejected:
            payload = {"rejected": True}
        else:
            payload = {"error": str(error)}
        path = f"/api/sign/{sid}/result"
        expires_at: float | None = None
        attempt = 0
        while True:
            status, body = await self._request("POST", path, json=payload)
            if status // 100 == 2 and status not in _RETRY_STATUSES:
                return body
            if status not in _RETRY_STATUSES:
                raise LfgError(status, _code_of(body), body)
            if expires_at is None:
                expires_at = float((await self.sign_request(sid))["expires_at"])
            now = self._clock()
            if now >= expires_at:
                raise LfgError(status, _code_of(body), body)
            step = self.SIGN_RESULT_BACKOFF[min(attempt, len(self.SIGN_RESULT_BACKOFF) - 1)]
            await self._sleep(min(step, expires_at - now))
            attempt += 1

    async def wait(self, getter: Callable[[str], Awaitable[dict]], sid: str, terminal: set[str],
                   timeout: float, every: float = 3.0) -> dict:
        """Poll `getter(sid)` every `every` seconds until its `state` is in `terminal`.

        Raises `TimeoutError` (naming the last state) once `timeout` seconds have passed;
        the final poll lands on the deadline itself.
        """
        deadline = self._clock() + timeout
        while True:
            status = await getter(sid)
            state = status.get("state")
            if state in terminal:
                return status
            now = self._clock()
            if now >= deadline:
                raise TimeoutError(f"{sid}: still {state!r} after {timeout}s")
            await self._sleep(max(0.0, min(every, deadline - now)))

    @staticmethod
    def sign_id(link: str) -> str:
        """`"lfg-wc://wc-<32 hex>"` → `"wc-<32 hex>"`; anything else raises ValueError."""
        if not isinstance(link, str) or not link.startswith(SIGN_LINK_SCHEME):
            raise ValueError(f"not an lfg-wc:// link: {link!r}")
        sid = link[len(SIGN_LINK_SCHEME):]
        if not _SIGN_ID_RE.match(sid):
            raise ValueError(f"not a sign request id: {sid!r}")
        return sid


def _maybe_json(text: str) -> dict | str:
    import json

    try:
        data = json.loads(text)
    except ValueError:
        return text
    return data if isinstance(data, dict) else text
