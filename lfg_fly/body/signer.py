"""The signer (spec §4.3): the fly's RegularKey and the policy every signature passes.

The signer holds only the RegularKey seed of the fly's wallet (§4.4: the master key stays
in Xaman, or in testnet's wallet.json which this module never reads). It signs exactly four
things, each under one row of the §4.3 table:

- Payment (proof), `sign_proof`: the canonical sign-in proof; never submitted.
- Payment (mint), `sign_and_submit(tx, Purpose.mint_payment(...))`: XRP drops equal to
  pay_amount x quantity, Destination = LFG's pinned signing account, no SendMax / Paths /
  DestinationTag, within the setup-wide cap (FLY_SETUP_MAX_XRP, `SpendLedger`).
- NFTokenAcceptOffer, `sign_and_submit(tx, Purpose.accept_offer(...))`: NFTokenSellOffer only
  (no NFTokenBuyOffer / NFTokenBrokerFee); the offer, read on-ledger, has Amount "0",
  Destination = the fly, and an owner that is the NFTokenID's issuer or a FLY_DONOR_SOURCES
  wallet.
- TrustSet, `sign_and_submit(tx, Purpose.trustset(...))`: LimitAmount is the BRIX pair and
  Flags stay within {tfSetNoRipple, tfFullyCanonicalSig}.

Anything else, any other TransactionType included, is refused with `PolicyError`.

`sign_and_submit` runs autofill -> policy -> `simulate` pre-flight -> sign -> submit against a
duck-typed ledger (`autofill`, `simulate`, `submit_and_wait`, `sell_offers`), so this module
never imports `lfg_fly.body.chain`. The chain-identity check (§4.4) is the caller's, once per
process, before the ledger is handed here. A RegularKey signs with its own key pair while
`Account` stays the wallet, at the binary-codec level (`encode_for_signing`, `keypairs.sign`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from xrpl.core import addresscodec, keypairs
from xrpl.core.binarycodec import encode, encode_for_signing

from lfg_fly import paths

# --- constants (docs/lfg-api.md "Config values"; xrpl flags)
PROOF_DESTINATION = "rrrrrrrrrrrrrrrrrNAMEtxvNvQ"
PROOF_AMOUNT = "1"
MAX_PROOF_FEE_DROPS = 10_000
MAX_FEE_DROPS = 10_000  # the same generous bound for the txs the fly does submit
NONCE_MEMO_TYPE = "lfg/nonce"
PROOF_MEMOS = {"initiator": "user", "platform": "agent", "action": "signin"}
DROPS_PER_XRP = 1_000_000
TF_FULLY_CANONICAL_SIG = 0x80000000
TF_SET_NO_RIPPLE = 0x00020000
LSF_SELL_NFTOKEN = 0x00000001
TX_HASH_PREFIX = bytes.fromhex("54584E00")
SPEND_FILE = "setup-spend.json"

PROOF_FIELDS = frozenset({
    "TransactionType", "Account", "Destination", "Amount", "Fee", "Sequence",
    "LastLedgerSequence", "SourceTag", "Memos",
})
PROOF_SIGNED_FIELDS = PROOF_FIELDS | {"SigningPubKey", "TxnSignature"}

# Fields every submitted tx may carry (autofill adds Fee/Sequence/LastLedgerSequence).
COMMON_FIELDS = frozenset({
    "TransactionType", "Account", "Fee", "Sequence", "LastLedgerSequence", "SourceTag", "Memos",
    "Flags", "SigningPubKey",
})
AUTOFILL_FIELDS = frozenset({"Fee", "Sequence", "LastLedgerSequence"})
ROW_FIELDS = {
    "mint_payment": frozenset({"Destination", "Amount"}),
    "accept_offer": frozenset({"NFTokenSellOffer"}),
    "trustset": frozenset({"LimitAmount"}),
}
ROW_TYPE = {"mint_payment": "Payment", "accept_offer": "NFTokenAcceptOffer", "trustset": "TrustSet"}
ROW_FLAGS = {
    "mint_payment": TF_FULLY_CANONICAL_SIG,
    "accept_offer": TF_FULLY_CANONICAL_SIG,
    "trustset": TF_FULLY_CANONICAL_SIG | TF_SET_NO_RIPPLE,
}
# Named so a refusal says what was smuggled in, not just "unexpected field".
NAMED_REFUSALS = {
    "mint_payment": ("SendMax", "Paths", "DestinationTag", "DeliverMin", "DeliverMax", "InvoiceID"),
    "accept_offer": ("NFTokenBuyOffer", "NFTokenBrokerFee"),
    "trustset": ("QualityIn", "QualityOut"),
}
KINDS = ("proof", "mint_payment", "accept_offer", "trustset")

_HEX = re.compile(r"[0-9A-Fa-f]+")
_HEX64 = re.compile(r"[0-9A-Fa-f]{64}")
_DIGITS = re.compile(r"[0-9]+")


class PolicyError(RuntimeError):
    """The signer refuses to sign (spec §4.3)."""


class PreflightError(RuntimeError):
    """`simulate` reported anything but tesSUCCESS without the ledger raising itself."""


# ---------------------------------------------------------------- helpers


def xrp_to_drops(xrp: Decimal | int | str) -> int:
    """Exact XRP -> drops; refuses fractions of a drop, negatives and non-finite values."""
    try:
        value = Decimal(str(xrp)) if not isinstance(xrp, Decimal) else xrp
    except InvalidOperation as e:
        raise ValueError("not a decimal XRP amount") from e
    if not value.is_finite() or value < 0:
        raise ValueError("XRP amount must be a finite, non-negative number")
    drops = value * DROPS_PER_XRP
    if drops != drops.to_integral_value():
        raise ValueError("XRP amount is not a whole number of drops")
    return int(drops)


def issuer_of(nft_id: str) -> str:
    """The issuer encoded in an NFTokenID: bytes 4..24 of the 32-byte id (spec §4.3 row 3)."""
    if not isinstance(nft_id, str) or not _HEX64.fullmatch(nft_id):
        raise ValueError("NFTokenID must be 64 hex characters")
    return addresscodec.encode_classic_address(bytes.fromhex(nft_id)[4:24])


def tx_hash(tx_blob: str) -> str:
    """The ledger's hash of a signed blob: SHA-512Half over the TXN prefix + blob."""
    digest = hashlib.sha512(TX_HASH_PREFIX + bytes.fromhex(tx_blob)).digest()
    return digest[:32].hex().upper()


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_drops(v: Any) -> bool:
    return isinstance(v, str) and bool(_DIGITS.fullmatch(v))


def _memo_dict(memos: Any) -> dict[str, str]:
    """Decode an LFG Memos array to {type: data}; raises PolicyError on any malformed entry."""
    if not isinstance(memos, list) or not memos:
        raise PolicyError("Memos must be a non-empty list")
    out: dict[str, str] = {}
    for entry in memos:
        body = entry.get("Memo") if isinstance(entry, dict) else None
        if not isinstance(body, dict) or set(body) - {"MemoType", "MemoData", "MemoFormat"}:
            raise PolicyError("Memos: malformed memo entry")
        for k in ("MemoType", "MemoData"):
            if k not in body or not isinstance(body[k], str) or not _HEX.fullmatch(body[k]):
                raise PolicyError(f"Memos: malformed memo {k}")
        if "MemoFormat" in body and not (
            isinstance(body["MemoFormat"], str) and _HEX.fullmatch(body["MemoFormat"])
        ):
            raise PolicyError("Memos: malformed memo MemoFormat")
        try:
            key = bytes.fromhex(body["MemoType"]).decode()
            val = bytes.fromhex(body["MemoData"]).decode()
        except (ValueError, UnicodeDecodeError) as e:
            raise PolicyError("Memos: malformed memo, not UTF-8") from e
        if key in out:
            raise PolicyError(f"Memos: duplicate memo {key!r}")
        out[key] = val
    return out


def _check_memos_shape(memos: Any) -> None:
    if memos is None:
        return
    if not isinstance(memos, list):
        raise PolicyError("Memos must be a list")
    if memos:
        _memo_dict(memos)


def _check_fee(tx: dict, cap: int) -> None:
    fee = tx.get("Fee")
    if not _is_drops(fee) or not 0 < int(fee) <= cap:
        raise PolicyError(f"Fee must be a digit string in 1..{cap} drops, got {fee!r}")


def _check_positive_int(tx: dict, key: str) -> None:
    v = tx.get(key)
    if not _is_int(v) or v <= 0:
        raise PolicyError(f"{key} must be a positive integer, got {v!r}")


# ---------------------------------------------------------------- Purpose


@dataclass(frozen=True)
class Purpose:
    """Why a signature is wanted: which §4.3 row applies, and that row's pinned parameters.

    Use the constructors: `proof()`, `mint_payment(pay_amount_xrp, quantity, destination)`,
    `accept_offer(nft_id, allowed_owners=())`, `trustset(currency, issuer)`.
    """

    kind: str
    pay_amount_xrp: Decimal | None = None  # mint_payment: the session's pay_amount (XRP)
    quantity: int = 1  # mint_payment: 1, or the bulk job's quantity
    destination: str | None = None  # mint_payment: LFG's signing account for this network
    pay_with: str = "XRP"  # mint_payment: the session's pay_with; anything but XRP is refused
    nft_id: str | None = None  # accept_offer: the NFT the offer sells
    allowed_owners: frozenset[str] = field(default_factory=frozenset)  # accept_offer: extras
    currency: str | None = None  # trustset: the BRIX currency code / hex
    issuer: str | None = None  # trustset: the BRIX issuer

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown purpose {self.kind!r}; one of {KINDS}")
        if self.kind == "mint_payment":
            if self.pay_amount_xrp is None or self.destination is None:
                raise ValueError("mint_payment needs pay_amount_xrp and destination")
            try:
                amount = Decimal(str(self.pay_amount_xrp))
            except InvalidOperation as e:
                raise ValueError("pay_amount_xrp is not a decimal") from e
            if not amount.is_finite():
                raise ValueError("pay_amount_xrp must be finite")
            object.__setattr__(self, "pay_amount_xrp", amount)
            if not _is_int(self.quantity) or self.quantity < 1:
                raise ValueError("quantity must be a positive integer")
        elif self.kind == "accept_offer":
            if not isinstance(self.nft_id, str):
                raise ValueError("accept_offer needs nft_id")
            issuer_of(self.nft_id)  # ValueError when malformed
        elif self.kind == "trustset":
            if not isinstance(self.currency, str) or not isinstance(self.issuer, str):
                raise ValueError("trustset needs currency and issuer")
        if not isinstance(self.allowed_owners, frozenset):
            object.__setattr__(self, "allowed_owners", frozenset(self.allowed_owners))

    @classmethod
    def proof(cls) -> Purpose:
        return cls("proof")

    @classmethod
    def mint_payment(cls, pay_amount_xrp: Decimal | str | int, quantity: int, destination: str,
                     pay_with: str = "XRP") -> Purpose:
        return cls("mint_payment", pay_amount_xrp=pay_amount_xrp,  # type: ignore[arg-type]
                   quantity=quantity, destination=destination, pay_with=pay_with)

    @classmethod
    def accept_offer(cls, nft_id: str, allowed_owners: Iterable[str] = ()) -> Purpose:
        return cls("accept_offer", nft_id=nft_id, allowed_owners=frozenset(allowed_owners))

    @classmethod
    def trustset(cls, currency: str, issuer: str) -> Purpose:
        return cls("trustset", currency=currency, issuer=issuer)


# ---------------------------------------------------------------- SpendLedger


class SpendLedger:
    """The setup-wide XRP cap, FLY_SETUP_MAX_XRP (spec §4.3 Payment (mint) row; §4.4 caps).

    Persisted at FLY_DATA_DIR/<network>/setup-spend.json so the cap survives process restarts.
    It counts mint payments' Amount (the price); network fees (about 12 drops a tx) are not
    counted, so an operator's whole-XRP cap admits exactly the mints it names.
    """

    def __init__(self, network: str, max_xrp: float | Decimal | str | int,
                 path: Path | None = None):
        self.network = network
        self.cap_drops = xrp_to_drops(Decimal(str(max_xrp)))
        self.path = path if path is not None else paths.network_dir(network) / SPEND_FILE

    def _load(self) -> dict:
        if not self.path.exists():
            return {"network": self.network, "spent_drops": 0, "charges": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("network") != self.network:
            raise PolicyError(
                f"{self.path} belongs to network {data.get('network')!r}, not {self.network!r}"
            )
        if not _is_int(data.get("spent_drops")) or data["spent_drops"] < 0:
            raise PolicyError(f"{self.path} is corrupt: spent_drops")
        return data

    @property
    def spent_drops(self) -> int:
        return self._load()["spent_drops"]

    @property
    def remaining_drops(self) -> int:
        return max(0, self.cap_drops - self.spent_drops)

    def check(self, drops: int) -> None:
        """Refuse when `drops` more would exceed the cap. Never writes."""
        if not _is_int(drops) or drops < 0:
            raise ValueError("drops must be a non-negative integer")
        spent = self.spent_drops
        if spent + drops > self.cap_drops:
            raise PolicyError(
                f"setup spend cap: {spent} + {drops} drops exceeds FLY_SETUP_MAX_XRP "
                f"({self.cap_drops} drops) on {self.network}"
            )

    def charge(self, drops: int) -> None:
        """Record `drops` spent; raises PolicyError past the cap (atomic tmp + rename)."""
        self.check(drops)
        data = self._load()
        data["spent_drops"] += drops
        data["charges"].append(
            {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "drops": drops}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------- Signer


class Signer:
    """The fly's RegularKey (spec §4.3, §4.4).

    `seed` is the RegularKey's family seed; `account` is the wallet it is the RegularKey of.
    The seed is derived once, kept only as a private key in memory, and never appears in a
    repr, a log or an exception. `address` (== `account`) is the fly's wallet; `key_address`
    is the key's own address, which must differ (the signer never holds the master key).
    """

    def __init__(self, seed: str, account: str, ledger: Any, config: Any,
                 spend_ledger: SpendLedger | None = None):
        if not isinstance(account, str) or not addresscodec.is_valid_classic_address(account):
            raise ValueError("account must be the fly's wallet, a classic r-address")
        if not isinstance(seed, str):
            raise ValueError("seed must be a family seed string")
        try:
            public, private = keypairs.derive_keypair(seed)
        except Exception:  # never echo the seed: xrpl's own message is dropped too
            raise ValueError("seed is not a valid XRPL family seed") from None
        self.account = account
        self.address = account
        self.public_key = public.upper()
        self.key_address = keypairs.derive_classic_address(self.public_key)
        if self.key_address == account:
            raise PolicyError("the signer never holds the master key (spec §4.4): "
                              "this seed is the wallet's own")
        self._private_key = private
        self._ledger = ledger
        self._config = config
        self.spend = spend_ledger if spend_ledger is not None else SpendLedger(
            config.network, getattr(config, "setup_max_xrp", 0.0)
        )

    def __repr__(self) -> str:
        return f"Signer(account={self.account}, key={self.key_address})"

    __str__ = __repr__

    # -- row 1 ---------------------------------------------------------------

    def sign_proof(self, tx_json: dict, source_tag: int) -> dict:
        """Sign the sign-in proof (spec §4.3 row 1; docs/lfg-api.md §1).

        The tx must be exactly {TransactionType Payment, Account (the wallet), Destination
        rrrrrrrrrrrrrrrrrNAMEtxvNvQ, Amount "1", Fee (1..10000 drops), Sequence > 0,
        LastLedgerSequence > 0, SourceTag == source_tag, Memos (LFG's, verbatim: initiator=user,
        platform=agent, action=signin, lfg/nonce)}. Returns a new dict with SigningPubKey
        (upper-case hex) and TxnSignature over encode_for_signing of the rest. Never submitted;
        this method never touches the ledger.
        """
        if not isinstance(tx_json, dict):
            raise PolicyError("proof tx_json must be a dict")
        if not _is_int(source_tag):
            raise PolicyError("source_tag must be an int")
        missing = PROOF_FIELDS - set(tx_json)
        extra = set(tx_json) - PROOF_FIELDS
        if missing:
            raise PolicyError(f"proof is missing {sorted(missing)}")
        if extra:
            raise PolicyError(f"proof carries unexpected field(s) {sorted(extra)}")
        if tx_json["TransactionType"] != "Payment":
            raise PolicyError("proof TransactionType must be Payment")
        if tx_json["Account"] != self.account:
            raise PolicyError("proof Account must be the fly's wallet")
        if tx_json["Destination"] != PROOF_DESTINATION:
            raise PolicyError(f"proof Destination must be {PROOF_DESTINATION}")
        if tx_json["Amount"] != PROOF_AMOUNT:
            raise PolicyError('proof Amount must be the string "1"')
        _check_fee(tx_json, MAX_PROOF_FEE_DROPS)
        _check_positive_int(tx_json, "Sequence")
        _check_positive_int(tx_json, "LastLedgerSequence")
        if not _is_int(tx_json["SourceTag"]) or tx_json["SourceTag"] != source_tag:
            raise PolicyError("proof SourceTag must equal the sign-in start's source_tag")
        decoded = _memo_dict(tx_json["Memos"])
        for key, want in PROOF_MEMOS.items():
            if decoded.get(key) != want:
                raise PolicyError(f"proof memo {key} must be {want!r}")
        nonce = decoded.get(NONCE_MEMO_TYPE)
        if not nonce:
            raise PolicyError("proof memos carry no lfg/nonce")
        if set(decoded) != set(PROOF_MEMOS) | {NONCE_MEMO_TYPE}:
            raise PolicyError("proof memos must be exactly initiator/platform/action/lfg/nonce")
        return self._sign(copy.deepcopy(tx_json))

    # -- rows 2-4 ------------------------------------------------------------

    def policy(self, tx: dict, purpose: Purpose, *, offers: Sequence[dict] | None = None) -> None:
        """Refuse (PolicyError) unless `tx` is exactly what `purpose`'s §4.3 row permits.

        Pure: reads `tx`, `purpose`, the config's pins and `offers`, the list the caller read
        on-ledger with `ledger.sell_offers(purpose.nft_id)` (required for accept_offer).
        """
        if not isinstance(tx, dict):
            raise PolicyError("tx_json must be a dict")
        if not isinstance(purpose, Purpose):
            raise PolicyError("purpose must be a Purpose")
        if purpose.kind == "proof":
            raise PolicyError("the sign-in proof is never submitted; use sign_proof")
        want = ROW_TYPE[purpose.kind]
        tt = tx.get("TransactionType")
        if tt != want:
            raise PolicyError(f"TransactionType {tt!r} refused under {purpose.kind} (only {want})")
        for name in NAMED_REFUSALS[purpose.kind]:
            if name in tx:
                raise PolicyError(f"{want} with {name} refused")
        self._check_common(tx, purpose.kind)
        if purpose.kind == "mint_payment":
            self._policy_mint(tx, purpose)
        elif purpose.kind == "accept_offer":
            self._policy_accept(tx, purpose, offers)
        else:
            self._policy_trustset(tx, purpose)

    def _check_common(self, tx: dict, kind: str) -> None:
        if tx.get("Account") != self.account:
            raise PolicyError("Account must be the fly's wallet")
        if "TxnSignature" in tx:
            raise PolicyError("TxnSignature present: the signer signs unsigned transactions")
        if "SigningPubKey" in tx and tx["SigningPubKey"] not in ("", self.public_key):
            raise PolicyError("SigningPubKey must be empty or the fly's RegularKey")
        extra = set(tx) - COMMON_FIELDS - ROW_FIELDS[kind]
        if extra:
            raise PolicyError(
                f"{tx['TransactionType']} carries unexpected field(s) {sorted(extra)}"
            )
        _check_fee(tx, MAX_FEE_DROPS)
        _check_positive_int(tx, "Sequence")
        _check_positive_int(tx, "LastLedgerSequence")
        if "SourceTag" in tx and (not _is_int(tx["SourceTag"]) or tx["SourceTag"] < 0):
            raise PolicyError("SourceTag must be a non-negative int")
        _check_memos_shape(tx.get("Memos"))
        if "Flags" in tx:
            flags = tx["Flags"]
            if not _is_int(flags) or flags < 0 or flags & ~ROW_FLAGS[kind]:
                raise PolicyError(f"Flags {flags!r} refused for {tx['TransactionType']}")

    def _policy_mint(self, tx: dict, purpose: Purpose) -> None:
        """§4.3 Payment (mint): XRP drops only, price x quantity, LFG's pinned account, the cap."""
        if purpose.pay_with != "XRP":
            raise PolicyError(f"mint pay_with {purpose.pay_with!r}: the fly pays XRP only")
        pinned = getattr(self._config, "signing_account", None)
        if not pinned:
            raise PolicyError("mint refused: LFG's signing account is not pinned "
                              "(FLY_LFG_SIGNING_ACCOUNT)")
        if purpose.destination != pinned:
            raise PolicyError("mint destination differs from the pinned LFG signing account")
        dest = tx.get("Destination")
        if dest != purpose.destination or dest == self.account:
            raise PolicyError("mint Destination must be LFG's signing account")
        amount = tx.get("Amount")
        if not _is_drops(amount):
            raise PolicyError("mint Amount must be XRP drops (a digit string)")
        try:
            expected = xrp_to_drops(purpose.pay_amount_xrp * purpose.quantity)  # type: ignore
        except ValueError as e:
            raise PolicyError(f"mint price is not a whole number of drops: {e}") from e
        if expected <= 0 or int(amount) != expected:
            raise PolicyError(f"mint Amount {amount} != pay_amount x quantity = {expected} drops")
        self.spend.check(expected)

    def _policy_accept(self, tx: dict, purpose: Purpose, offers: Sequence[dict] | None) -> None:
        """§4.3 NFTokenAcceptOffer: a sell offer read on-ledger, free, to the fly, from the
        NFTokenID's issuer or a donor source."""
        index = tx.get("NFTokenSellOffer")
        if not isinstance(index, str) or not _HEX64.fullmatch(index):
            raise PolicyError("NFTokenSellOffer must be a 64-hex offer index")
        if offers is None:
            raise PolicyError("accept refused: the offer was not read on-ledger")
        match = [o for o in offers if isinstance(o, dict)
                 and str(o.get("nft_offer_index", "")).upper() == index.upper()]
        if not match:
            raise PolicyError(f"offer {index} is not on the ledger for {purpose.nft_id}")
        offer = match[0]
        if offer.get("amount") != "0":
            raise PolicyError(f"offer Amount must be \"0\", got {offer.get('amount')!r}")
        if offer.get("destination") != self.account:
            raise PolicyError("offer Destination must be the fly's wallet")
        flags = offer.get("flags", LSF_SELL_NFTOKEN)
        if not _is_int(flags) or not flags & LSF_SELL_NFTOKEN:
            raise PolicyError("offer is not a sell offer")
        allowed = {issuer_of(purpose.nft_id)} | set(purpose.allowed_owners)  # type: ignore
        allowed |= set(getattr(self._config, "donor_sources", ()) or ())
        owner = offer.get("owner")
        if not isinstance(owner, str) or owner not in allowed:
            raise PolicyError(
                f"offer owner {owner!r} is neither the NFT's issuer nor a donor source"
            )

    def _policy_trustset(self, tx: dict, purpose: Purpose) -> None:
        """§4.3 TrustSet: the BRIX pair and nothing else."""
        limit = tx.get("LimitAmount")
        if not isinstance(limit, dict) or set(limit) - {"currency", "issuer", "value"}:
            raise PolicyError("LimitAmount must be {currency, issuer, value}")
        if not _same_currency(limit.get("currency"), purpose.currency):
            raise PolicyError(
                f"LimitAmount currency {limit.get('currency')!r} is not the BRIX pair"
            )
        issuer = limit.get("issuer")
        if issuer != purpose.issuer or issuer == self.account:
            raise PolicyError(f"LimitAmount issuer {issuer!r} is not the BRIX pair")
        value = limit.get("value")
        try:
            parsed = Decimal(value) if isinstance(value, str) else None
        except InvalidOperation:
            parsed = None
        if parsed is None or not parsed.is_finite() or parsed < 0:
            raise PolicyError(f"LimitAmount value {value!r} must be a non-negative decimal string")

    # -- the flow -------------------------------------------------------------

    async def sign_and_submit(self, tx_json: dict, purpose: Purpose) -> dict:
        """autofill -> policy -> simulate -> sign -> submit_and_wait (spec §4.3).

        Returns {"hash": <64 upper hex>, "result": <the validated result>}. The spend cap is
        charged just before a mint payment is submitted, so a crash mid-submit counts as spent.
        """
        if not isinstance(purpose, Purpose):
            raise PolicyError("purpose must be a Purpose")
        if purpose.kind == "proof":
            raise PolicyError("the sign-in proof is never submitted; use sign_proof")
        if not isinstance(tx_json, dict):
            raise PolicyError("tx_json must be a dict")
        tx = await self._ledger.autofill(copy.deepcopy(tx_json))
        _check_autofill(tx_json, tx)
        offers = None
        if purpose.kind == "accept_offer":
            offers = await self._ledger.sell_offers(purpose.nft_id)
        self.policy(tx, purpose, offers=offers)
        unsigned = {k: v for k, v in tx.items() if k not in ("SigningPubKey", "TxnSignature")}
        sim = await self._ledger.simulate(dict(unsigned))
        if isinstance(sim, dict):
            engine = sim.get("engine_result")
            if engine is not None and engine != "tesSUCCESS":
                raise PreflightError(f"simulate: {engine} {sim.get('engine_result_message', '')}")
        signed = self._sign(unsigned)
        blob = encode(signed)
        if purpose.kind == "mint_payment":
            self.spend.charge(int(tx["Amount"]))
        result = await self._ledger.submit_and_wait(blob)
        digest = tx_hash(blob)
        got = result.get("hash") if isinstance(result, dict) else None
        if isinstance(got, str) and got.upper() != digest:
            raise RuntimeError(f"ledger reported hash {got} for a blob hashing to {digest}")
        return {"hash": digest, "result": result}

    def _sign(self, tx: dict) -> dict:
        tx["SigningPubKey"] = self.public_key
        try:
            message = bytes.fromhex(encode_for_signing(tx))
        except Exception as e:
            raise PolicyError(f"transaction does not encode: {e}") from e
        tx["TxnSignature"] = keypairs.sign(message, self._private_key)
        return tx


def _same_currency(a: Any, b: Any) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    if len(a) == 40 and len(b) == 40 and _HEX.fullmatch(a) and _HEX.fullmatch(b):
        return a.upper() == b.upper()
    return a == b


def _check_autofill(before: dict, after: dict) -> None:
    """Autofill may only set Fee / Sequence / LastLedgerSequence (and an empty SigningPubKey
    placeholder); any other difference is tampering by the endpoint and is refused."""
    if not isinstance(after, dict):
        raise PolicyError("autofill returned no transaction")
    for k, v in before.items():
        if k in AUTOFILL_FIELDS:
            continue
        if k not in after or after[k] != v:
            raise PolicyError(f"autofill changed {k}")
    added = set(after) - set(before) - AUTOFILL_FIELDS
    if added == {"SigningPubKey"} and after["SigningPubKey"] == "":
        added = set()
    if added:
        raise PolicyError(f"autofill added {sorted(added)}")
