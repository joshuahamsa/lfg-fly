"""A fake XRPL JSON-RPC endpoint for the chain tests (spec §4.4, §7 "chain identity").

`FakeRpc` is an aiohttp `TestServer` answering `POST /` the way rippled and
clio do: `{"result": {..., "status": "success"}}`, or an error result
`{"error": "lgrNotFound", "error_code", "error_message", "status": "error"}`.
The shapes mirror the live public endpoints (probed 2026-09-25): `server_info`
carries `info.network_id`; `ledger` puts `ledger_hash` at the top level and
inside `ledger`; rippled answers `nft_info` with `unknownCmd` while clio
serves it; `nft_sell_offers` without offers is `objectNotFound`.

Every method is a responder `params -> result dict` in `responders`, so a test
can swap any one of them. Helpers cover the §4.4 cases: `prune()` makes the
endpoint answer `lgrNotFound` for ledger 32570, `wrong_hash()` makes it return
a different hash, `unreachable_url()` gives an address nobody listens on, and
`hang` / `http_status` / `garbage` make the endpoint misbehave at the HTTP layer.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Callable
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer
from xrpl.core import binarycodec
from xrpl.core.keypairs.helpers import sha512_first_half

MAINNET_HASH = "4109C6F2045FC7EFF4CDE8F9905D19C28820D86304080FF886B299F0206E42B5"
TESTNET_HASH = "18D82E2D616C76A960DB76D514ED39BCC377381A037DFC0F5AF4F0E22E14FC53"
OTHER_HASH = "00" * 32

Responder = Callable[[dict], dict]


class RpcError(Exception):
    """Raised by a responder to answer with an XRPL error result."""

    def __init__(self, error: str, message: str = "", **extra: Any):
        super().__init__(error)
        self.error = error
        self.message = message or error
        self.extra = extra


def unreachable_url() -> str:
    """A loopback URL with nothing listening: connections are refused at once."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/"


def tx_hash(tx_blob: str) -> str:
    """The transaction id of a signed blob: SHA-512Half(0x54584E00 || blob)."""
    return sha512_first_half(bytes.fromhex("54584E00" + tx_blob)).hex().upper()


class FakeRpc:
    """One fake endpoint. Use as `async with FakeRpc(...) as rpc:` and read `rpc.url`."""

    def __init__(
        self,
        *,
        network_id: int = 1,
        hash_32570: str = TESTNET_HASH,
        validated: int = 1_000,
        clio: bool = False,
        open_ledger_fee: str = "10",
    ):
        self.network_id = network_id
        self.hash_32570: str | None = hash_32570
        self.validated = validated
        self.clio = clio
        self.open_ledger_fee = open_ledger_fee
        # ledger state
        self.accounts: dict[str, dict] = {}  # address -> account_data
        self.nfts: dict[str, list[dict]] = {}  # owner -> account_nfts entries
        self.nft_infos: dict[str, dict] = {}  # nft_id -> nft_info result (clio only)
        self.offers: dict[str, list[dict]] = {}  # nft_id -> sell offers
        self.txs: dict[str, dict] = {}  # hash -> tx result (validated or not)
        # submit behaviour
        self.submit_result = "tesSUCCESS"  # preliminary engine_result
        self.final_result = "tesSUCCESS"  # meta.TransactionResult once validated
        self.polls_until_validated: int | None = 1  # None: never validates
        self.advance_per_ledger_query = 0  # validated ledger index growth per `ledger` call
        self.simulate_result = "tesSUCCESS"
        # HTTP-layer misbehaviour
        self.hang: float = 0.0
        self.http_status: int | None = None
        self.garbage = False
        self.page_size = 400
        self.calls: list[tuple[str, dict]] = []
        self.responders: dict[str, Responder] = {
            "server_info": self._server_info,
            "ledger": self._ledger,
            "fee": self._fee,
            "account_info": self._account_info,
            "account_nfts": self._account_nfts,
            "nft_info": self._nft_info,
            "nft_sell_offers": self._nft_sell_offers,
            "tx": self._tx,
            "submit": self._submit,
            "simulate": self._simulate,
        }
        self._server: TestServer | None = None

    # -- lifecycle -----------------------------------------------------------------

    async def __aenter__(self) -> FakeRpc:
        app = web.Application()
        app.router.add_post("/", self._handle)
        self._server = TestServer(app)
        await self._server.start_server()
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        await self._server.close()

    @property
    def url(self) -> str:
        assert self._server is not None, "use `async with FakeRpc() as rpc`"
        return str(self._server.make_url("/"))

    # -- §4.4 helpers ------------------------------------------------------------------

    def prune(self) -> FakeRpc:
        """Answer `lgrNotFound` for ledger 32570, like the public testnet rippled nodes."""
        self.hash_32570 = None
        return self

    def wrong_hash(self, h: str = OTHER_HASH) -> FakeRpc:
        self.hash_32570 = h
        return self

    def answer(self, method: str, responder: Responder) -> FakeRpc:
        self.responders[method] = responder
        return self

    def answer_error(self, method: str, error: str, message: str = "") -> FakeRpc:
        def _fail(_params: dict) -> dict:
            raise RpcError(error, message)

        return self.answer(method, _fail)

    def calls_for(self, method: str) -> list[dict]:
        return [p for m, p in self.calls if m == method]

    def add_account(self, address: str, sequence: int = 1, balance: str = "100000000") -> None:
        self.accounts[address] = {
            "Account": address,
            "Balance": balance,
            "Sequence": sequence,
            "OwnerCount": 0,
            "Flags": 0,
            "LedgerEntryType": "AccountRoot",
        }

    def add_nft(self, owner: str, nft_id: str, uri: str | None, issuer: str = "rIssuer") -> None:
        entry: dict[str, Any] = {
            "Flags": 8,
            "Issuer": issuer,
            "NFTokenID": nft_id,
            "NFTokenTaxon": 0,
            "nft_serial": len(self.nfts.get(owner, [])),
        }
        if uri is not None:
            entry["URI"] = uri.encode().hex().upper()
        self.nfts.setdefault(owner, []).append(entry)
        self.nft_infos[nft_id] = {
            "nft_id": nft_id,
            "ledger_index": self.validated,
            "owner": owner,
            "is_burned": False,
            "flags": 8,
            "transfer_fee": 0,
            "issuer": issuer,
            "nft_taxon": 0,
            "nft_serial": entry["nft_serial"],
            "uri": entry.get("URI", ""),
        }

    def validate_pending(self) -> None:
        for t in self.txs.values():
            self._validate(t)

    # -- the HTTP handler ----------------------------------------------------------------

    async def _handle(self, request: web.Request) -> web.Response:
        if self.hang:
            await asyncio.sleep(self.hang)
        if self.http_status is not None:
            return web.Response(status=self.http_status, text="unavailable")
        if self.garbage:
            return web.Response(text="<html>not json</html>")
        body = await request.json()
        method = body.get("method")
        params = (body.get("params") or [{}])[0]
        self.calls.append((method, params))
        responder = self.responders.get(method)
        try:
            if responder is None:
                raise RpcError("unknownCmd", "Unknown method.")
            result = dict(responder(params))
            result.setdefault("status", "success")
        except RpcError as e:
            result = {
                "error": e.error,
                "error_code": 0,
                "error_message": e.message,
                "request": {"command": method, **params},
                "status": "error",
                **e.extra,
            }
        return web.json_response({"result": result})

    # -- default responders --------------------------------------------------------------

    def _server_info(self, _params: dict) -> dict:
        return {
            "info": {
                "network_id": self.network_id,
                "complete_ledgers": f"32570-{self.validated}",
                "validated_ledger": {"seq": self.validated, "hash": "AA" * 32},
            }
        }

    def _ledger(self, params: dict) -> dict:
        index = params.get("ledger_index", "validated")
        if index == "validated" or index == "current" or index == "closed":
            self.validated += self.advance_per_ledger_query
            seq, h = self.validated, "CC" * 32
        elif index == 32570:
            if self.hash_32570 is None:
                raise RpcError("lgrNotFound", "ledgerNotFound")
            seq, h = 32570, self.hash_32570
        else:
            raise RpcError("lgrNotFound", "ledgerNotFound")
        return {
            "ledger_hash": h,
            "ledger_index": seq,
            "validated": True,
            "ledger": {"ledger_hash": h, "ledger_index": str(seq), "closed": True},
        }

    def _fee(self, _params: dict) -> dict:
        return {
            "drops": {
                "base_fee": "10",
                "median_fee": "5000",
                "minimum_fee": "10",
                "open_ledger_fee": self.open_ledger_fee,
            },
            "ledger_current_index": self.validated + 1,
        }

    def _account_info(self, params: dict) -> dict:
        data = self.accounts.get(params.get("account", ""))
        if data is None:
            raise RpcError("actNotFound", "Account not found.", account=params.get("account"))
        return {
            "account_data": dict(data),
            "ledger_index": self.validated,
            "validated": True,
        }

    def _account_nfts(self, params: dict) -> dict:
        owner = params.get("account", "")
        if owner not in self.accounts and owner not in self.nfts:
            raise RpcError("actNotFound", "Account not found.")
        entries = self.nfts.get(owner, [])
        start = int(params.get("marker", 0) or 0)
        limit = min(int(params.get("limit", self.page_size) or self.page_size), self.page_size)
        page = entries[start : start + limit]
        result: dict[str, Any] = {
            "account": owner,
            "account_nfts": page,
            "ledger_index": self.validated,
            "validated": True,
        }
        if start + limit < len(entries):
            result["marker"] = start + limit
        return result

    def _nft_info(self, params: dict) -> dict:
        if not self.clio:
            raise RpcError("unknownCmd", "Unknown method.")
        info = self.nft_infos.get(params.get("nft_id", ""))
        if info is None:
            raise RpcError("objectNotFound", "NFT not found")
        return dict(info)

    def _nft_sell_offers(self, params: dict) -> dict:
        offers = self.offers.get(params.get("nft_id", ""))
        if not offers:
            raise RpcError("objectNotFound", "The requested object was not found.")
        return {"nft_id": params["nft_id"], "offers": list(offers)}

    def _tx(self, params: dict) -> dict:
        h = params.get("transaction", "").upper()
        t = self.txs.get(h)
        if t is None:
            raise RpcError("txnNotFound", "Transaction not found.")
        t["_polls"] = t.get("_polls", 0) + 1
        if (
            not t["validated"]
            and self.polls_until_validated is not None
            and t["_polls"] >= self.polls_until_validated
        ):
            self._validate(t)
        return {k: v for k, v in t.items() if not k.startswith("_")}

    def _validate(self, t: dict) -> None:
        t["validated"] = True
        t["ledger_index"] = self.validated
        t["meta"] = {"TransactionIndex": 0, "TransactionResult": self.final_result}

    def _submit(self, params: dict) -> dict:
        blob = params.get("tx_blob")
        if not blob:
            raise RpcError("invalidParams", "Missing field 'tx_blob'.")
        try:
            tx_json = binarycodec.decode(blob)
        except Exception as e:  # noqa: BLE001 — the fake mirrors rippled's error
            raise RpcError("invalidTransaction", "Transaction is not valid.") from e
        h = tx_hash(blob)
        prelim = self.submit_result
        if not prelim.startswith(("tem", "tef")):
            self.txs.setdefault(h, {**tx_json, "hash": h, "validated": False})
            # like rippled: an applied transaction consumes the account's sequence, so two
            # otherwise identical payments never share a hash
            acct = self.accounts.get(str(tx_json.get("Account", "")))
            seq = tx_json.get("Sequence")
            if acct is not None and isinstance(seq, int) and seq >= acct.get("Sequence", 0):
                acct["Sequence"] = seq + 1
        return {
            "engine_result": prelim,
            "engine_result_code": 0 if prelim == "tesSUCCESS" else -1,
            "engine_result_message": prelim,
            "tx_blob": blob,
            "tx_json": {**tx_json, "hash": h},
            "accepted": True,
            "applied": prelim.startswith(("tes", "tec")),
            "broadcast": not prelim.startswith(("tem", "tef")),
            "kept": True,
            "queued": False,
            "validated_ledger_index": self.validated,
        }

    def _simulate(self, params: dict) -> dict:
        tx_json = params.get("tx_json")
        if not isinstance(tx_json, dict):
            raise RpcError("invalidParams", "Missing field 'tx_json'.")
        if tx_json.get("Account") not in self.accounts:
            raise RpcError("srcActNotFound", "Source account not found.")
        r = self.simulate_result
        return {
            "engine_result": r,
            "engine_result_code": 0 if r == "tesSUCCESS" else -1,
            "engine_result_message": r,
            "tx_json": dict(tx_json),
            "meta": {"TransactionResult": r},
            "ledger_index": self.validated + 1,
        }
