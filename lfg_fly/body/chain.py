"""The fly's view of the XRP Ledger: chain identity, reads, autofill, simulate, submit.

Spec §4.4 "Chain identity": `server_info.network_id` (mainnet 0, testnet 1)
plus the hash of ledger 32570, asked of full-history public endpoints;
endpoints answering `lgrNotFound` or unreachable are skipped, at least one must
return the expected hash, and any mismatch refuses. The contract is
`docs/plans/2026-09-25-body-interfaces.md` § `lfg_fly/body/chain.py`.

Raw aiohttp JSON-RPC (`POST {"method", "params": [{...}]}` → `{"result": {...}}`)
rather than xrpl-py's client, so the fake in `tests/fake_rpc.py` is a plain
dict of responders. `request` accepts either a dict with a `method` key or an
xrpl-py `Request` model. Nothing here ever sees a seed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp
from xrpl.core import binarycodec
from xrpl.core.keypairs.helpers import sha512_first_half

log = logging.getLogger(__name__)

NETWORK_IDS = {"mainnet": 0, "testnet": 1}
IDENTITY_LEDGER = 32570
LLS_WINDOW = 20  # LastLedgerSequence = validated + 20 (contract; LFG allows up to +900)
DEFAULT_FEE_DROPS = 12
MAX_FEE_DROPS = 10_000  # the proof's ceiling (§4.3 row 1); a spike above it waits, never pays
NFTS_PAGE = 400
OFFERS_PAGE = 500

# Error answers that say "this endpoint can't help right now", not "your request is wrong":
# `request` moves on to the next URL instead of returning them.
SKIP_ERRORS = frozenset(
    {
        "unknownCmd",
        "notSupported",
        "noNetwork",
        "noCurrent",
        "noClosed",
        "notSynced",
        "tooBusy",
        "slowDown",
        "amendmentBlocked",
        "lgrNotFound",
        "internal",
    }
)


class ChainError(RuntimeError):
    """Base of everything this module raises."""


class ChainIdentityError(ChainError):
    """§4.4: the endpoints do not prove the expected network; the fly refuses to run."""


class LedgerUnavailable(ChainError):
    """No configured endpoint answered the request."""


class RequestError(ChainError):
    """An endpoint answered with an error result the caller did not expect."""

    def __init__(self, method: str, result: dict):
        self.method = method
        self.result = result
        self.error = str(result.get("error"))
        super().__init__(f"{method}: {self.error}: {result.get('error_message', '')}".rstrip(": "))


class AutofillError(ChainError):
    """The transaction cannot be completed (no Account, or the account is not on this ledger)."""


class SimulateFailed(ChainError):
    """The `simulate` pre-flight did not come back `tesSUCCESS`."""

    def __init__(self, message: str, engine_result: str | None = None, result: dict | None = None):
        super().__init__(message)
        self.engine_result = engine_result
        self.result = result


class SubmitFailed(ChainError):
    """The transaction was refused, or did not validate by its LastLedgerSequence."""

    def __init__(
        self,
        message: str,
        engine_result: str | None = None,
        hash: str | None = None,  # noqa: A002 — the field name the ledger uses
        result: dict | None = None,
    ):
        super().__init__(message)
        self.engine_result = engine_result
        self.hash = hash
        self.result = result


@dataclass(frozen=True)
class ChainIdentity:
    """What the §4.4 check established, and which endpoint proved it."""

    network_id: int
    ledger_32570: str
    endpoint: str


def transaction_hash(tx_blob: str) -> str:
    """The id of a signed blob: SHA-512Half(0x54584E00 || blob), upper-case hex."""
    return sha512_first_half(bytes.fromhex("54584E00" + tx_blob)).hex().upper()


def decode_uri(uri_hex: str) -> str:
    """An NFToken URI field (hex) as text."""
    return bytes.fromhex(uri_hex).decode("utf-8")


def _encode(req: Any) -> tuple[str, dict]:
    """(method, params) from a dict with a `method` key or an xrpl-py Request model."""
    if isinstance(req, dict):
        params = dict(req)
    elif hasattr(req, "to_dict"):
        params = dict(req.to_dict())
    else:
        raise TypeError(f"not a request: {type(req).__name__}")
    method = params.pop("method", None)
    if not method:
        raise ValueError("request has no method")
    return str(method), params


class Ledger:
    """A list of JSON-RPC endpoints for one network, tried in order.

    `timeout` bounds each HTTP call; `poll_interval` is the pause between
    `tx` polls in `submit_and_wait` (0 in tests); `max_outages` is how many
    consecutive polls may find no endpoint before the wait gives up.
    """

    def __init__(
        self,
        urls: Sequence[str],
        network: str,
        timeout: float = 20.0,
        *,
        poll_interval: float = 1.0,
        max_outages: int = 5,
        session: aiohttp.ClientSession | None = None,
    ):
        if network not in NETWORK_IDS:
            raise ValueError(f"unknown network {network!r}; expected one of {sorted(NETWORK_IDS)}")
        if not urls:
            raise ValueError("at least one RPC url is required")
        self.urls = tuple(urls)
        self.network = network
        self.network_id = NETWORK_IDS[network]
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_outages = max_outages
        self._session = session
        self._owned = session is None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- transport ---------------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        stale = self._session is None or (
            self._owned and (self._session.closed or self._loop is not loop)
        )
        if stale:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            self._owned = True
            self._loop = loop
        assert self._session is not None
        return self._session

    async def close(self) -> None:
        if self._owned and self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None if self._owned else self._session

    async def __aenter__(self) -> Ledger:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _ask(self, url: str, method: str, params: dict) -> dict | None:
        """One endpoint's `result`, or None when it is unreachable or answers garbage."""
        payload = {"method": method, "params": [params]}
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    log.debug("%s %s: HTTP %s", url, method, resp.status)
                    return None
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, ValueError) as e:
            log.debug("%s %s: %s", url, method, type(e).__name__)
            return None
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, dict):
            log.debug("%s %s: no result object", url, method)
            return None
        return result

    async def request(self, req: Any, *, skip: frozenset[str] = SKIP_ERRORS) -> dict:
        """The first endpoint's answer, including semantic error results (e.g. actNotFound).

        Endpoints that are unreachable, answer non-JSON, or answer an error in
        `skip` (unknownCmd, tooBusy, lgrNotFound, ...) are passed over.
        """
        method, params = _encode(req)
        reasons: list[str] = []
        for url in self.urls:
            result = await self._ask(url, method, params)
            if result is None:
                reasons.append(f"{url}: unreachable")
                continue
            error = result.get("error")
            if error in skip:
                reasons.append(f"{url}: {error}")
                continue
            return result
        raise LedgerUnavailable(f"no endpoint answered {method}: " + "; ".join(reasons))

    # -- §4.4 chain identity -----------------------------------------------------------------

    async def identity_check(self, expected_hash: str | None) -> ChainIdentity:
        """§4.4: prove every endpoint is on this network and one of them holds ledger 32570.

        Per URL: `server_info.network_id` must equal the network's id (0 / 1);
        a different id refuses at once. Then `ledger(32570)`: `lgrNotFound` and
        unreachable endpoints are skipped. Every URL is asked before deciding,
        so a mismatching hash anywhere refuses even when another endpoint
        agrees with the pin; the first agreeing endpoint is the one reported.
        No pinned hash (a testnet reset not yet re-pinned) refuses too.
        """
        if not expected_hash:
            raise ChainIdentityError(
                f"no ledger-{IDENTITY_LEDGER} hash pinned for {self.network}: refusing to run"
            )
        expected = expected_hash.upper()
        matches: list[str] = []
        skipped: list[str] = []
        for url in self.urls:
            info = await self._ask(url, "server_info", {})
            if info is None or info.get("error"):
                skipped.append(f"{url}: {'unreachable' if info is None else info.get('error')}")
                continue
            network_id = (info.get("info") or {}).get("network_id")
            if network_id != self.network_id:
                raise ChainIdentityError(
                    f"{url}: server_info.network_id is {network_id!r}, "
                    f"expected {self.network_id} ({self.network})"
                )
            ledger = await self._ask(url, "ledger", {"ledger_index": IDENTITY_LEDGER})
            if ledger is None or ledger.get("error"):
                skipped.append(
                    f"{url}: {'unreachable' if ledger is None else ledger.get('error')}"
                )
                continue
            found = ledger.get("ledger_hash") or (ledger.get("ledger") or {}).get("ledger_hash")
            if not isinstance(found, str) or not found:
                skipped.append(f"{url}: no ledger_hash")
                continue
            if found.upper() != expected:
                raise ChainIdentityError(
                    f"{url}: ledger {IDENTITY_LEDGER} hash is {found.upper()}, "
                    f"expected {expected} ({self.network})"
                )
            matches.append(url)
        if not matches:
            raise ChainIdentityError(
                f"no endpoint returned ledger {IDENTITY_LEDGER} with hash {expected}: "
                + "; ".join(skipped)
            )
        log.info("chain identity ok: %s network_id=%d via %s", self.network, self.network_id,
                 matches[0])
        return ChainIdentity(network_id=self.network_id, ledger_32570=expected, endpoint=matches[0])

    # -- reads -------------------------------------------------------------------------------

    async def account_info(self, address: str) -> dict | None:
        """`account_info` on the validated ledger; None when the account does not exist."""
        r = await self.request(
            {"method": "account_info", "account": address, "ledger_index": "validated"}
        )
        if r.get("error") == "actNotFound":
            return None
        if r.get("error"):
            raise RequestError("account_info", r)
        return r

    async def account_nfts(self, address: str) -> list[dict]:
        """Every NFToken the account holds (follows `marker`); [] for an unknown account."""
        out: list[dict] = []
        marker: Any = None
        while True:
            params: dict[str, Any] = {
                "method": "account_nfts",
                "account": address,
                "ledger_index": "validated",
                "limit": NFTS_PAGE,
            }
            if marker is not None:
                params["marker"] = marker
            r = await self.request(params)
            if r.get("error") == "actNotFound":
                return []
            if r.get("error"):
                raise RequestError("account_nfts", r)
            out.extend(r.get("account_nfts") or [])
            marker = r.get("marker")
            if marker is None:
                return out

    async def nft_info(self, nft_id: str) -> dict | None:
        """clio's `nft_info`; None when no endpoint serves it or the token is unknown."""
        for url in self.urls:
            r = await self._ask(url, "nft_info", {"nft_id": nft_id})
            if r is None:
                continue
            error = r.get("error")
            if error in SKIP_ERRORS:
                continue  # a rippled node: unknownCmd
            if error == "objectNotFound":
                return None
            if error:
                raise RequestError("nft_info", r)
            return r
        return None

    async def nft_uri(self, nft_id: str, owner: str) -> str | None:
        """§4.2 step 2: the token's current URI from the ledger, never LFG's index.

        clio's `nft_info` first; a rippled-only endpoint list falls back to
        scanning `account_nfts(owner)`. None when the token is burned, absent
        or has no URI.
        """
        info = await self.nft_info(nft_id)
        if info is not None and not info.get("is_burned"):
            uri = info.get("uri")
            return decode_uri(uri) if uri else None
        want = nft_id.upper()
        for entry in await self.account_nfts(owner):
            if str(entry.get("NFTokenID", "")).upper() == want:
                uri = entry.get("URI")
                return decode_uri(uri) if uri else None
        return None

    async def sell_offers(self, nft_id: str) -> list[dict]:
        """`nft_sell_offers` (follows `marker`); [] when there are none (objectNotFound)."""
        out: list[dict] = []
        marker: Any = None
        while True:
            params: dict[str, Any] = {
                "method": "nft_sell_offers",
                "nft_id": nft_id,
                "ledger_index": "validated",
                "limit": OFFERS_PAGE,
            }
            if marker is not None:
                params["marker"] = marker
            r = await self.request(params)
            if r.get("error") == "objectNotFound":
                return out
            if r.get("error"):
                raise RequestError("nft_sell_offers", r)
            out.extend(r.get("offers") or [])
            marker = r.get("marker")
            if marker is None:
                return out

    async def tx(self, tx_hash: str) -> dict | None:
        """`tx` by hash; None when no endpoint knows it yet (txnNotFound)."""
        r = await self.request({"method": "tx", "transaction": tx_hash, "binary": False})
        if r.get("error") == "txnNotFound":
            return None
        if r.get("error"):
            raise RequestError("tx", r)
        r.setdefault("hash", tx_hash.upper())
        return r

    async def validated_ledger_index(self) -> int:
        r = await self.request({"method": "ledger", "ledger_index": "validated"})
        if r.get("error"):
            raise RequestError("ledger", r)
        return int(r["ledger_index"])

    # -- writes ------------------------------------------------------------------------------

    async def _fee_drops(self) -> int:
        """The open-ledger fee, floored at 12 drops, capped at MAX_FEE_DROPS; 12 without `fee`."""
        try:
            r = await self.request({"method": "fee"})
            drops = int((r.get("drops") or {})["open_ledger_fee"])
        except (LedgerUnavailable, KeyError, TypeError, ValueError):
            return DEFAULT_FEE_DROPS
        return max(DEFAULT_FEE_DROPS, min(drops, MAX_FEE_DROPS))

    async def autofill(self, tx: dict) -> dict:
        """A copy of `tx` with Fee, Sequence and LastLedgerSequence filled where missing.

        Fee from the server's open-ledger fee (or "12"); Sequence from
        `account_info`; LastLedgerSequence = validated + 20. Never NetworkID:
        network ids 0 and 1 must not carry it. Fields already present are kept.
        """
        out = dict(tx)
        account = out.get("Account")
        if not isinstance(account, str) or not account:
            raise AutofillError("transaction has no Account")
        if "Fee" not in out:
            out["Fee"] = str(await self._fee_drops())
        if "Sequence" not in out:
            info = await self.account_info(account)
            if info is None:
                raise AutofillError(f"account {account} is not on the {self.network} ledger")
            out["Sequence"] = int(info["account_data"]["Sequence"])
        if "LastLedgerSequence" not in out:
            out["LastLedgerSequence"] = await self.validated_ledger_index() + LLS_WINDOW
        return out

    async def simulate(self, tx_json: dict) -> dict:
        """§4.3 pre-flight: the `simulate` RPC; anything but `tesSUCCESS` raises SimulateFailed."""
        try:
            r = await self.request({"method": "simulate", "tx_json": tx_json, "binary": False})
        except LedgerUnavailable as e:
            raise SimulateFailed(f"simulate unavailable: {e}") from e
        if r.get("error"):
            raise SimulateFailed(
                f"simulate refused: {r.get('error')}: {r.get('error_message', '')}".rstrip(": "),
                engine_result=str(r.get("error")),
                result=r,
            )
        engine_result = r.get("engine_result")
        if engine_result != "tesSUCCESS":
            raise SimulateFailed(
                f"simulate: {engine_result}: {r.get('engine_result_message', '')}".rstrip(": "),
                engine_result=engine_result,
                result=r,
            )
        return r

    async def submit_and_wait(self, tx_blob: str) -> dict:
        """Submit a signed blob and wait for its validated result.

        The hash is computed locally before anything is sent, so a lost
        `submit` answer still leads to polling rather than a blind re-submit
        (§4.2 "never blindly re-submit"). A `tem`/`tef` preliminary result
        raises at once. Polling `tx` continues until the transaction is
        validated or the validated ledger passes its LastLedgerSequence
        (required: an unbounded wait is refused). A validated result other
        than `tesSUCCESS` raises SubmitFailed with that engine result.
        """
        try:
            decoded = binarycodec.decode(tx_blob)
        except Exception as e:  # noqa: BLE001 — binarycodec raises several types
            raise SubmitFailed(f"tx_blob does not decode: {e}") from e
        lls = decoded.get("LastLedgerSequence")
        if not isinstance(lls, int):
            raise SubmitFailed("tx_blob has no LastLedgerSequence: refusing an unbounded submit")
        tx_hash = transaction_hash(tx_blob)
        prelim: str | None = None
        try:
            r = await self.request({"method": "submit", "tx_blob": tx_blob})
        except LedgerUnavailable as e:
            # The blob may have reached the network before the answer was lost: poll, never resend.
            log.warning("submit %s: %s; polling until ledger %d", tx_hash, e, lls)
        else:
            if r.get("error"):
                raise SubmitFailed(
                    f"submit refused: {r.get('error')}: {r.get('error_message', '')}".rstrip(": "),
                    engine_result=str(r.get("error")),
                    hash=tx_hash,
                    result=r,
                )
            prelim = r.get("engine_result")
            if isinstance(prelim, str) and prelim[:3] in ("tem", "tef"):
                detail = r.get("engine_result_message", "")
                raise SubmitFailed(
                    f"submit {tx_hash}: {prelim}: {detail}".rstrip(": "),
                    engine_result=prelim,
                    hash=tx_hash,
                    result=r,
                )
        return await self._wait_validated(tx_hash, lls, prelim)

    def _finish(self, tx_hash: str, r: dict) -> dict:
        meta = r.get("meta") if isinstance(r.get("meta"), dict) else r.get("metaData")
        code = (meta or {}).get("TransactionResult")
        if code != "tesSUCCESS":
            raise SubmitFailed(
                f"{tx_hash} validated with {code}", engine_result=code, hash=tx_hash, result=r
            )
        return r

    async def _wait_validated(self, tx_hash: str, lls: int, prelim: str | None) -> dict:
        outages = 0
        while True:
            try:
                r = await self.tx(tx_hash)
                if r is not None and r.get("validated"):
                    return self._finish(tx_hash, r)
                current = await self.validated_ledger_index()
            except LedgerUnavailable:
                outages += 1
                if outages >= self.max_outages:
                    raise
                await asyncio.sleep(self.poll_interval)
                continue
            outages = 0
            if current > lls:
                # one last look: the endpoint that answered `tx` may have lagged the one
                # that answered `ledger`
                r = await self.tx(tx_hash)
                if r is not None and r.get("validated"):
                    return self._finish(tx_hash, r)
                raise SubmitFailed(
                    f"{tx_hash} not validated by LastLedgerSequence {lls} "
                    f"(validated ledger {current}); preliminary result {prelim}",
                    engine_result=prelim,
                    hash=tx_hash,
                )
            await asyncio.sleep(self.poll_interval)
