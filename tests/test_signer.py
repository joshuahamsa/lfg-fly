"""The signer (spec §4.3 policy table, every row permitted and refused; §4.4 RegularKey only).

The ledger is a duck (autofill / simulate / submit_and_wait / sell_offers) so nothing here
imports lfg_fly.body.chain; the config is a plain dataclass with the fields the signer reads.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

import pytest
from xrpl.constants import CryptoAlgorithm
from xrpl.core import keypairs
from xrpl.core.binarycodec import decode, encode_for_signing
from xrpl.models.transactions.transaction import Transaction
from xrpl.utils import parse_nftoken_id, str_to_hex

from lfg_fly.body import signer as S

run = asyncio.run

PROOF_DEST = "rrrrrrrrrrrrrrrrrNAMEtxvNvQ"
SOURCE_TAG = 2606160021
NONCE = "ab" * 32
BRIX_HEX = "4252495800000000000000000000000000000000"
# xrpl-py's own documented example: issuer rJoxBSzpXhPtAuqFmqxQtGKjA13jUJWthE
NFT_ID = "000B0539C35B55AA096BA6D87A6E6C965A6534150DC56E5E12C5D09E0000000C"
NFT_ISSUER = "rJoxBSzpXhPtAuqFmqxQtGKjA13jUJWthE"
OFFER_IDX = "AA" * 32
TF_FULLY_CANONICAL = 0x80000000
TF_SET_NO_RIPPLE = 0x00020000


def seed_for(tag: str, algorithm: CryptoAlgorithm = CryptoAlgorithm.ED25519) -> str:
    """A deterministic, unfunded family seed for a test role (16 bytes of entropy, hex)."""
    return keypairs.generate_seed(hashlib.sha256(tag.encode()).hexdigest()[:32], algorithm)


def addr_of(seed: str) -> str:
    return keypairs.derive_classic_address(keypairs.derive_keypair(seed)[0])


REGULAR_SEED = seed_for("regular-key")
MASTER = addr_of(seed_for("fly-master-wallet"))  # the fly's wallet; its master key is elsewhere
LFG_SIGNER = addr_of(seed_for("lfg-bot-wallet"))
BRIX_ISSUER = addr_of(seed_for("brix-issuer"))
DONOR = addr_of(seed_for("operator-donor"))
STRANGER = addr_of(seed_for("a-stranger"))


# ---------------------------------------------------------------- fakes


@dataclass
class FakeConfig:
    network: str = "testnet"
    setup_max_xrp: float = 100.0
    donor_sources: tuple[str, ...] = ()
    signing_account: str | None = LFG_SIGNER


class FakeLedger:
    """The four ledger methods the signer uses, recording the order they are called in."""

    def __init__(self, offers: dict[str, list[dict]] | None = None):
        self.calls: list = []
        self.offers = offers or {}
        self.fee, self.sequence, self.lls = "12", 41, 5000
        self.simulate_exc: Exception | None = None
        self.simulate_result: dict = {"engine_result": "tesSUCCESS"}
        self.tamper = None
        self.blob: str | None = None
        self.simulated: dict | None = None

    async def autofill(self, tx: dict) -> dict:
        self.calls.append("autofill")
        out = dict(tx)
        out.setdefault("Fee", self.fee)
        out.setdefault("Sequence", self.sequence)
        out.setdefault("LastLedgerSequence", self.lls)
        if self.tamper:
            self.tamper(out)
        return out

    async def simulate(self, tx_json: dict) -> dict:
        self.calls.append("simulate")
        self.simulated = dict(tx_json)
        if self.simulate_exc:
            raise self.simulate_exc
        return dict(self.simulate_result)

    async def submit_and_wait(self, tx_blob: str) -> dict:
        self.calls.append("submit")
        self.blob = tx_blob
        return {"hash": S.tx_hash(tx_blob), "validated": True,
                "meta": {"TransactionResult": "tesSUCCESS"}}

    async def sell_offers(self, nft_id: str) -> list[dict]:
        self.calls.append(("sell_offers", nft_id))
        return [dict(o) for o in self.offers.get(nft_id, [])]


def memo(key: str, value: str, fmt: bool = True) -> dict:
    m = {"MemoType": str_to_hex(key), "MemoData": str_to_hex(value)}
    if fmt:
        m["MemoFormat"] = str_to_hex("text/plain")
    return {"Memo": m}


def lfg_memos(action: str, nonce: str | None = None, platform: str = "agent") -> list[dict]:
    out = [memo("initiator", "user"), memo("platform", platform), memo("action", action)]
    if nonce is not None:
        out.append(memo("lfg/nonce", nonce, fmt=False))
    return out


def proof_tx(account: str = MASTER, **over) -> dict:
    tx = {
        "TransactionType": "Payment", "Account": account, "Destination": PROOF_DEST,
        "Amount": "1", "Fee": "12", "Sequence": 1, "LastLedgerSequence": 12_345,
        "SourceTag": SOURCE_TAG, "Memos": lfg_memos("signin", NONCE),
    }
    for k, v in over.items():
        if v is None:
            tx.pop(k, None)
        else:
            tx[k] = v
    return tx


def filled(tx: dict, **over) -> dict:
    """What the ledger's autofill hands the policy (fills only what the tx lacks)."""
    out = {"Fee": "12", "Sequence": 41, "LastLedgerSequence": 5000, **tx}
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out


def mint_tx(amount: str = "30000000", dest: str = LFG_SIGNER, **over) -> dict:
    tx = {"TransactionType": "Payment", "Account": MASTER, "Destination": dest,
          "Amount": amount, "SourceTag": SOURCE_TAG, "Memos": lfg_memos("mint")}
    tx.update(over)
    return tx


def accept_tx(**over) -> dict:
    tx = {"TransactionType": "NFTokenAcceptOffer", "Account": MASTER,
          "NFTokenSellOffer": OFFER_IDX, "SourceTag": SOURCE_TAG,
          "Memos": lfg_memos("accept-offer")}
    tx.update(over)
    return tx


def ledger_offer(owner: str = NFT_ISSUER, **over) -> dict:
    o = {"amount": "0", "flags": 1, "nft_offer_index": OFFER_IDX, "owner": owner,
         "destination": MASTER}
    for k, v in over.items():
        if v is None:
            o.pop(k, None)
        else:
            o[k] = v
    return o


def trustset_tx(**over) -> dict:
    tx = {"TransactionType": "TrustSet", "Account": MASTER, "Flags": TF_SET_NO_RIPPLE,
          "LimitAmount": {"currency": BRIX_HEX, "issuer": BRIX_ISSUER, "value": "1000000000"},
          "SourceTag": SOURCE_TAG, "Memos": lfg_memos("trustset")}
    tx.update(over)
    return tx


MINT = S.Purpose.mint_payment(Decimal("10"), 3, LFG_SIGNER)
ACCEPT = S.Purpose.accept_offer(NFT_ID)
TRUST = S.Purpose.trustset(BRIX_HEX, BRIX_ISSUER)


@pytest.fixture
def ledger():
    return FakeLedger(offers={NFT_ID: [ledger_offer()]})


@pytest.fixture
def cfg():
    return FakeConfig()


@pytest.fixture
def signer(ledger, cfg):
    return S.Signer(REGULAR_SEED, MASTER, ledger, cfg)


# ---------------------------------------------------------------- construction (§4.4)


def test_signer_is_a_regular_key_for_the_wallet(signer):
    pub, _ = keypairs.derive_keypair(REGULAR_SEED)
    assert signer.account == MASTER
    assert signer.address == MASTER  # the fly's wallet, not the key's own address
    assert signer.public_key == pub.upper()
    assert signer.key_address == keypairs.derive_classic_address(pub)
    assert signer.key_address != MASTER


def test_signer_never_holds_the_master_key(ledger, cfg):
    with pytest.raises(S.PolicyError, match="master"):
        S.Signer(REGULAR_SEED, addr_of(REGULAR_SEED), ledger, cfg)


def test_signer_rejects_bad_inputs_without_echoing_the_seed(ledger, cfg):
    with pytest.raises(ValueError) as exc:
        S.Signer("sEdNotASeedAtAll", MASTER, ledger, cfg)
    assert "sEdNotASeedAtAll" not in str(exc.value)
    with pytest.raises(ValueError, match="account"):
        S.Signer(REGULAR_SEED, "not-an-address", ledger, cfg)
    with pytest.raises(ValueError, match="account"):
        S.Signer(REGULAR_SEED, ledger, cfg, cfg)  # positional slip: account is not a str


def test_repr_and_str_hide_the_seed(signer):
    text = repr(signer) + str(signer) + json.dumps(vars(signer), default=str)
    assert REGULAR_SEED not in text
    assert MASTER in repr(signer)


# ---------------------------------------------------------------- row 1: the proof


def verify_like_lfg(signed: dict, tx_in: dict) -> None:
    """What lfg_core/signing/proof.py verify_proof checks, minus the ledger key lookup."""
    assert set(signed) == S.PROOF_SIGNED_FIELDS
    for k, v in tx_in.items():
        assert signed[k] == v
    pub, sig = signed["SigningPubKey"], signed["TxnSignature"]
    assert pub == pub.upper() and sig == sig.upper()
    unsigned = {k: v for k, v in signed.items() if k != "TxnSignature"}
    blob = bytes.fromhex(encode_for_signing(unsigned))
    assert keypairs.is_valid_message(blob, bytes.fromhex(sig), pub)
    # and the same signature over a tampered tx fails, so the check above is not vacuous
    tampered = bytes.fromhex(encode_for_signing({**unsigned, "Amount": "2"}))
    assert not keypairs.is_valid_message(tampered, bytes.fromhex(sig), pub)


@pytest.mark.parametrize("algorithm", [CryptoAlgorithm.ED25519, CryptoAlgorithm.SECP256K1])
def test_sign_proof_verifies_under_lfg_rules(algorithm, ledger, cfg):
    seed = seed_for("regular-key", algorithm)
    signer = S.Signer(seed, MASTER, ledger, cfg)
    tx = proof_tx()
    before = json.dumps(tx, sort_keys=True)
    signed = signer.sign_proof(tx, SOURCE_TAG)
    verify_like_lfg(signed, tx)
    assert signed["SigningPubKey"] == signer.public_key
    assert signed["Memos"] == tx["Memos"] and signed["Memos"] is not tx["Memos"]
    assert json.dumps(tx, sort_keys=True) == before  # input untouched
    assert ledger.calls == []  # never autofilled, simulated or submitted
    # the signing account is the wallet, signed by a key that is not its master
    assert keypairs.derive_classic_address(signed["SigningPubKey"]) != signed["Account"]


def test_sign_proof_is_deterministic_for_ed25519(signer):
    a = signer.sign_proof(proof_tx(), SOURCE_TAG)
    b = signer.sign_proof(proof_tx(), SOURCE_TAG)
    assert a == b


@pytest.mark.parametrize("over, why", [
    ({"Destination": MASTER}, "Destination"),
    ({"Destination": LFG_SIGNER}, "Destination"),
    ({"Amount": "2"}, "Amount"),
    ({"Amount": 1}, "Amount"),
    ({"Amount": {"currency": "USD", "issuer": BRIX_ISSUER, "value": "1"}}, "Amount"),
    ({"Fee": "10001"}, "Fee"),
    ({"Fee": "0"}, "Fee"),
    ({"Fee": 12}, "Fee"),
    ({"Fee": None}, "Fee"),
    ({"LastLedgerSequence": None}, "LastLedgerSequence"),
    ({"LastLedgerSequence": 0}, "LastLedgerSequence"),
    ({"LastLedgerSequence": "12345"}, "LastLedgerSequence"),
    ({"Sequence": 0}, "Sequence"),
    ({"Sequence": True}, "Sequence"),
    ({"Sequence": None}, "Sequence"),
    ({"SourceTag": SOURCE_TAG + 1}, "SourceTag"),
    ({"SourceTag": None}, "SourceTag"),
    ({"Account": STRANGER}, "Account"),
    ({"Account": None}, "Account"),
    ({"TransactionType": "AccountSet"}, "TransactionType"),
    ({"Memos": None}, "Memos"),
    ({"Memos": []}, "Memos"),
    ({"Memos": lfg_memos("link", NONCE)}, "action"),
    ({"Memos": lfg_memos("signin")}, "nonce"),
    ({"Memos": lfg_memos("signin", NONCE, platform="webapp")}, "platform"),
    ({"Memos": lfg_memos("signin", NONCE) + [memo("campaign", "x")]}, "memo"),
    ({"Memos": lfg_memos("signin", NONCE) + [memo("lfg/nonce", "cd" * 32, False)]}, "memo"),
    ({"Memos": [{"Memo": {"MemoType": "zz", "MemoData": "00"}}]}, "memo"),
    ({"DestinationTag": 7}, "DestinationTag"),
    ({"SendMax": "1"}, "SendMax"),
    ({"Flags": 0}, "Flags"),
    ({"NetworkID": 1}, "NetworkID"),
    ({"SigningPubKey": ""}, "SigningPubKey"),
    ({"TxnSignature": "00"}, "TxnSignature"),
])
def test_sign_proof_refuses_anything_but_the_canonical_proof(signer, over, why):
    with pytest.raises(S.PolicyError, match=why):
        signer.sign_proof(proof_tx(**over), SOURCE_TAG)


def test_sign_proof_source_tag_must_match_the_start_response(signer):
    with pytest.raises(S.PolicyError, match="SourceTag"):
        signer.sign_proof(proof_tx(), SOURCE_TAG + 1)
    with pytest.raises(S.PolicyError):
        signer.sign_proof(proof_tx(), "2606160021")  # type: ignore[arg-type]


def test_sign_proof_refuses_non_dicts(signer):
    with pytest.raises(S.PolicyError):
        signer.sign_proof(["Payment"], SOURCE_TAG)  # type: ignore[arg-type]


def test_proof_purpose_is_never_submitted(signer, ledger):
    with pytest.raises(S.PolicyError, match="never submitted"):
        signer.policy(filled(proof_tx()), S.Purpose.proof())
    with pytest.raises(S.PolicyError, match="never submitted"):
        run(signer.sign_and_submit(proof_tx(), S.Purpose.proof()))
    assert ledger.calls == []


# ---------------------------------------------------------------- Purpose


def test_purpose_constructors_and_validation():
    p = S.Purpose.mint_payment("10", 3, LFG_SIGNER)
    assert p.kind == "mint_payment" and p.pay_amount_xrp == Decimal("10") and p.quantity == 3
    assert p.pay_with == "XRP" and p.destination == LFG_SIGNER
    a = S.Purpose.accept_offer(NFT_ID, [DONOR])
    assert a.kind == "accept_offer" and a.nft_id == NFT_ID and a.allowed_owners == {DONOR}
    assert isinstance(a.allowed_owners, frozenset)
    t = S.Purpose.trustset(BRIX_HEX, BRIX_ISSUER)
    assert (t.kind, t.currency, t.issuer) == ("trustset", BRIX_HEX, BRIX_ISSUER)
    assert S.Purpose.proof().kind == "proof"
    with pytest.raises(ValueError):
        S.Purpose("harvest")
    with pytest.raises(ValueError):
        S.Purpose("mint_payment")  # no amount / destination
    with pytest.raises(ValueError):
        S.Purpose.mint_payment("10", 0, LFG_SIGNER)
    with pytest.raises(ValueError):
        S.Purpose("accept_offer")
    with pytest.raises(ValueError):
        S.Purpose("trustset", currency=BRIX_HEX)
    with pytest.raises(ValueError):
        S.Purpose.mint_payment("ten", 1, LFG_SIGNER)


# ---------------------------------------------------------------- row 2: Payment (mint)


def test_mint_payment_permitted(signer):
    signer.policy(filled(mint_tx()), MINT)
    signer.policy(filled(mint_tx(Flags=TF_FULLY_CANONICAL)), MINT)
    signer.policy(filled(mint_tx(Flags=0)), MINT)
    signer.policy(filled(mint_tx(), SigningPubKey=""), MINT)  # xrpl-py style placeholder
    # one unit at the fractional price LFG could quote
    signer.policy(filled(mint_tx("12500000")), S.Purpose.mint_payment("12.5", 1, LFG_SIGNER))
    # no memos / source tag is still a permitted shape
    tx = mint_tx()
    del tx["Memos"], tx["SourceTag"]
    signer.policy(filled(tx), MINT)


@pytest.mark.parametrize("over, why", [
    ({"Amount": "30000001"}, "Amount"),
    ({"Amount": "29999999"}, "Amount"),
    ({"Amount": "0"}, "Amount"),
    ({"Amount": 30000000}, "Amount"),
    ({"Amount": {"currency": "LFGO", "issuer": LFG_SIGNER, "value": "30"}}, "drops"),
    ({"Destination": DONOR}, "Destination"),
    ({"Destination": MASTER}, "Destination"),
    ({"SendMax": "30000000"}, "SendMax"),
    ({"Paths": [[{"currency": "USD", "issuer": BRIX_ISSUER}]]}, "Paths"),
    ({"DestinationTag": 1}, "DestinationTag"),
    ({"DeliverMin": "1"}, "DeliverMin"),
    ({"InvoiceID": "00" * 32}, "InvoiceID"),
    ({"Flags": 0x00020000}, "Flags"),  # tfPartialPayment
    ({"Flags": 0x00010000}, "Flags"),  # tfNoRippleDirect
    ({"Account": STRANGER}, "Account"),
    ({"Fee": "10001"}, "Fee"),
    ({"Fee": "abc"}, "Fee"),
    ({"Fee": None}, "Fee"),
    ({"Sequence": None}, "Sequence"),
    ({"LastLedgerSequence": None}, "LastLedgerSequence"),
    ({"TxnSignature": "00"}, "TxnSignature"),
    ({"SigningPubKey": "ED" + "00" * 32}, "SigningPubKey"),
    ({"TransactionType": "NFTokenAcceptOffer", "NFTokenSellOffer": OFFER_IDX}, "TransactionType"),
    ({"Memos": "not a list"}, "Memos"),
    ({"Memos": [{"Memo": {"MemoType": "zz"}}]}, "Memos"),
    ({"NetworkID": 1}, "NetworkID"),
])
def test_mint_payment_refused(signer, over, why):
    with pytest.raises(S.PolicyError, match=why):
        signer.policy(filled(mint_tx(), **over), MINT)


def test_mint_payment_purpose_pins(signer, ledger):
    with pytest.raises(S.PolicyError, match="pay_with"):
        lfgo = S.Purpose.mint_payment("10", 3, LFG_SIGNER, pay_with="LFGO")
        signer.policy(filled(mint_tx()), lfgo)
    # the purpose's destination must agree with the config pin, and the tx with both
    with pytest.raises(S.PolicyError, match="pinned"):
        signer.policy(filled(mint_tx(dest=DONOR)), S.Purpose.mint_payment("10", 3, DONOR))
    unpinned = S.Signer(REGULAR_SEED, MASTER, ledger, FakeConfig(signing_account=None))
    with pytest.raises(S.PolicyError, match="pinned"):
        unpinned.policy(filled(mint_tx()), MINT)
    # a price that is not a whole number of drops can never match
    with pytest.raises(S.PolicyError, match="drops"):
        signer.policy(filled(mint_tx()), S.Purpose.mint_payment("0.0000001", 1, LFG_SIGNER))
    with pytest.raises(S.PolicyError, match="Amount"):
        signer.policy(filled(mint_tx()), S.Purpose.mint_payment("10", 2, LFG_SIGNER))


def test_mint_payment_spend_cap(ledger):
    cfg = FakeConfig(setup_max_xrp=30.0)
    signer = S.Signer(REGULAR_SEED, MASTER, ledger, cfg)
    signer.policy(filled(mint_tx("30000000")), MINT)  # exactly the cap is allowed
    with pytest.raises(S.PolicyError, match="FLY_SETUP_MAX_XRP"):
        signer.policy(filled(mint_tx("40000000")), S.Purpose.mint_payment("10", 4, LFG_SIGNER))
    zero = S.Signer(REGULAR_SEED, MASTER, ledger, FakeConfig(setup_max_xrp=0.0))
    with pytest.raises(S.PolicyError, match="FLY_SETUP_MAX_XRP"):
        zero.policy(filled(mint_tx("10000000")), S.Purpose.mint_payment("10", 1, LFG_SIGNER))
    # policy is pure: nothing was charged
    assert signer.spend.spent_drops == 0


# ---------------------------------------------------------------- row 3: NFTokenAcceptOffer


def test_accept_offer_permitted(signer):
    offers = [ledger_offer()]  # owner = the issuer encoded in the NFTokenID
    signer.policy(filled(accept_tx()), ACCEPT, offers=offers)
    signer.policy(filled(accept_tx(Flags=TF_FULLY_CANONICAL)), ACCEPT, offers=offers)
    bare = {"TransactionType": "NFTokenAcceptOffer", "Account": MASTER,
            "NFTokenSellOffer": OFFER_IDX.lower()}
    signer.policy(filled(bare), ACCEPT, offers=offers)  # index case does not matter
    # a wallet in FLY_DONOR_SOURCES
    donors = S.Signer(REGULAR_SEED, MASTER, FakeLedger(), FakeConfig(donor_sources=(DONOR,)))
    donors.policy(filled(accept_tx()), ACCEPT, offers=[ledger_offer(owner=DONOR)])
    # or one the purpose names
    signer.policy(filled(accept_tx()), S.Purpose.accept_offer(NFT_ID, {DONOR}),
                  offers=[ledger_offer(owner=DONOR)])
    # unrelated offers on the same NFT do not matter
    signer.policy(filled(accept_tx()), ACCEPT,
                  offers=[ledger_offer(owner=STRANGER, nft_offer_index="BB" * 32), ledger_offer()])


@pytest.mark.parametrize("over, why", [
    ({"NFTokenBuyOffer": "CC" * 32}, "NFTokenBuyOffer"),
    ({"NFTokenBuyOffer": "CC" * 32, "NFTokenBrokerFee": "1"}, "NFTokenBuyOffer"),
    ({"NFTokenBrokerFee": "1"}, "NFTokenBrokerFee"),
    ({"NFTokenSellOffer": None}, "NFTokenSellOffer"),
    ({"NFTokenSellOffer": "zz" * 32}, "NFTokenSellOffer"),
    ({"NFTokenSellOffer": "AA" * 31}, "NFTokenSellOffer"),
    ({"NFTokenSellOffer": "BB" * 32}, "on the ledger"),
    ({"Flags": 1}, "Flags"),
    ({"Account": STRANGER}, "Account"),
    ({"Fee": "20000"}, "Fee"),
    ({"TransactionType": "Payment", "Destination": LFG_SIGNER, "Amount": "1"}, "TransactionType"),
    ({"Destination": LFG_SIGNER}, "Destination"),
    ({"TxnSignature": "00"}, "TxnSignature"),
])
def test_accept_offer_refused_by_tx(signer, over, why):
    tx = accept_tx()
    for k, v in over.items():
        if v is None:
            tx.pop(k)
        else:
            tx[k] = v
    with pytest.raises(S.PolicyError, match=why):
        signer.policy(filled(tx), ACCEPT, offers=[ledger_offer()])


@pytest.mark.parametrize("offer_over, why", [
    ({"amount": "1"}, "Amount"),
    ({"amount": "0.000001"}, "Amount"),
    ({"amount": 0}, "Amount"),
    ({"amount": {"currency": "BRIX", "issuer": BRIX_ISSUER, "value": "0"}}, "Amount"),
    ({"amount": None}, "Amount"),
    ({"destination": STRANGER}, "Destination"),
    ({"destination": None}, "Destination"),
    ({"owner": STRANGER}, "owner"),
    ({"owner": LFG_SIGNER}, "owner"),
    ({"owner": None}, "owner"),
    ({"flags": 0}, "sell"),
])
def test_accept_offer_refused_by_the_ledger_offer(signer, offer_over, why):
    with pytest.raises(S.PolicyError, match=why):
        signer.policy(filled(accept_tx()), ACCEPT, offers=[ledger_offer(**offer_over)])


def test_accept_offer_needs_the_offer_read_on_ledger(signer):
    with pytest.raises(S.PolicyError, match="on-ledger"):
        signer.policy(filled(accept_tx()), ACCEPT)
    with pytest.raises(S.PolicyError, match="on the ledger"):
        signer.policy(filled(accept_tx()), ACCEPT, offers=[])
    # the purpose's own owner list cannot smuggle in a non-zero price or another destination
    with pytest.raises(S.PolicyError, match="Amount"):
        signer.policy(filled(accept_tx()), S.Purpose.accept_offer(NFT_ID, {STRANGER}),
                      offers=[ledger_offer(owner=STRANGER, amount="1")])


PolicyErrorOrValue = (S.PolicyError, ValueError)


def test_issuer_of_nftoken_id():
    assert S.issuer_of(NFT_ID) == NFT_ISSUER == parse_nftoken_id(NFT_ID)["issuer"]
    assert S.issuer_of(NFT_ID.lower()) == NFT_ISSUER
    for tag in ("a", "b", "c"):
        raw = hashlib.sha256(tag.encode()).hexdigest().upper()
        assert S.issuer_of(raw) == parse_nftoken_id(raw)["issuer"]
    with pytest.raises(PolicyErrorOrValue):
        S.issuer_of("00" * 31)
    with pytest.raises(PolicyErrorOrValue):
        S.issuer_of("zz" * 32)


# ---------------------------------------------------------------- row 4: TrustSet


def test_trustset_permitted(signer):
    signer.policy(filled(trustset_tx()), TRUST)
    signer.policy(filled(trustset_tx(Flags=TF_SET_NO_RIPPLE | TF_FULLY_CANONICAL)), TRUST)
    signer.policy(filled(trustset_tx(Flags=TF_FULLY_CANONICAL)), TRUST)
    tx = trustset_tx()
    del tx["Flags"]
    signer.policy(filled(tx), TRUST)
    # currency hex compares case-insensitively; a 3-letter code exactly
    lower = {"currency": BRIX_HEX.lower(), "issuer": BRIX_ISSUER, "value": "1000000000"}
    signer.policy(filled(trustset_tx(LimitAmount=lower)), TRUST)
    iso = {"currency": "BRX", "issuer": BRIX_ISSUER, "value": "5"}
    signer.policy(filled(trustset_tx(LimitAmount=iso)), S.Purpose.trustset("BRX", BRIX_ISSUER))


@pytest.mark.parametrize("over, why", [
    ({"LimitAmount": {"currency": "USD", "issuer": BRIX_ISSUER, "value": "1"}}, "currency"),
    ({"LimitAmount": {"currency": BRIX_HEX, "issuer": STRANGER, "value": "1"}}, "issuer"),
    ({"LimitAmount": {"currency": BRIX_HEX, "issuer": MASTER, "value": "1"}}, "issuer"),
    ({"LimitAmount": {"currency": BRIX_HEX, "issuer": BRIX_ISSUER, "value": "-1"}}, "value"),
    ({"LimitAmount": {"currency": BRIX_HEX, "issuer": BRIX_ISSUER, "value": "lots"}}, "value"),
    ({"LimitAmount": {"currency": BRIX_HEX, "issuer": BRIX_ISSUER}}, "value"),
    ({"LimitAmount": "1000000000"}, "LimitAmount"),
    ({"LimitAmount": None}, "LimitAmount"),
    ({"Flags": 0x00010000}, "Flags"),  # tfSetfAuth
    ({"Flags": 0x00100000}, "Flags"),  # tfSetFreeze
    ({"Flags": TF_SET_NO_RIPPLE | 0x00040000}, "Flags"),  # + tfClearNoRipple
    ({"Flags": "131072"}, "Flags"),
    ({"QualityIn": 1}, "QualityIn"),
    ({"QualityOut": 1}, "QualityOut"),
    ({"Account": STRANGER}, "Account"),
    ({"Fee": "10001"}, "Fee"),
    ({"TransactionType": "Payment", "Destination": LFG_SIGNER, "Amount": "1"}, "TransactionType"),
    ({"TxnSignature": "00"}, "TxnSignature"),
])
def test_trustset_refused(signer, over, why):
    tx = trustset_tx()
    for k, v in over.items():
        if v is None:
            tx.pop(k)
        else:
            tx[k] = v
    with pytest.raises(S.PolicyError, match=why):
        signer.policy(filled(tx), TRUST)


# ---------------------------------------------------------------- anything else


@pytest.mark.parametrize("tt, extra", [
    ("SetRegularKey", {"RegularKey": STRANGER}),
    ("AccountSet", {"SetFlag": 4}),
    ("NFTokenCreateOffer", {"NFTokenID": NFT_ID, "Amount": "0", "Flags": 1}),
    ("NFTokenCancelOffer", {"NFTokenOffers": [OFFER_IDX]}),
    ("NFTokenBurn", {"NFTokenID": NFT_ID}),
    ("NFTokenModify", {"NFTokenID": NFT_ID}),
    ("OfferCreate", {"TakerGets": "1", "TakerPays": {"currency": "USD", "issuer": BRIX_ISSUER,
                                                     "value": "1"}}),
    ("SignerListSet", {"SignerQuorum": 0}),
    ("AccountDelete", {"Destination": STRANGER}),
    ("Payment", {"Destination": STRANGER, "Amount": "1"}),
])
@pytest.mark.parametrize("purpose", [MINT, ACCEPT, TRUST])
def test_other_transactions_are_refused_under_every_purpose(signer, tt, extra, purpose):
    tx = filled({"TransactionType": tt, "Account": MASTER, **extra})
    if tt == purpose_type(purpose):
        pytest.skip("that row's own type; covered by its refusal table")
    with pytest.raises(S.PolicyError, match="TransactionType"):
        signer.policy(tx, purpose, offers=[ledger_offer()])


def purpose_type(p: S.Purpose) -> str:
    return {"mint_payment": "Payment", "accept_offer": "NFTokenAcceptOffer",
            "trustset": "TrustSet"}[p.kind]


def test_policy_refuses_non_dicts_and_unknown_purposes(signer):
    with pytest.raises(S.PolicyError):
        signer.policy("Payment", MINT)  # type: ignore[arg-type]
    with pytest.raises(S.PolicyError):
        signer.policy(filled(mint_tx()), "mint_payment")  # type: ignore[arg-type]


# ---------------------------------------------------------------- sign_and_submit


def decoded_blob(ledger: FakeLedger) -> dict:
    assert ledger.blob is not None
    return decode(ledger.blob)


def assert_signed_by(signer: S.Signer, tx: dict) -> None:
    unsigned = {k: v for k, v in tx.items() if k != "TxnSignature"}
    assert tx["SigningPubKey"] == signer.public_key
    assert keypairs.is_valid_message(bytes.fromhex(encode_for_signing(unsigned)),
                                     bytes.fromhex(tx["TxnSignature"]), tx["SigningPubKey"])


def test_sign_and_submit_mint_payment_end_to_end(signer, ledger):
    out = run(signer.sign_and_submit(mint_tx(), MINT))
    assert ledger.calls == ["autofill", "simulate", "submit"]
    tx = decoded_blob(ledger)
    assert tx["Account"] == MASTER and tx["Destination"] == LFG_SIGNER
    assert tx["Amount"] == "30000000" and tx["Fee"] == "12"
    assert tx["Sequence"] == 41 and tx["LastLedgerSequence"] == 5000
    assert_signed_by(signer, tx)
    # simulate saw the autofilled, unsigned tx
    assert ledger.simulated["Fee"] == "12" and "TxnSignature" not in ledger.simulated
    assert out["hash"] == S.tx_hash(ledger.blob) == out["result"]["hash"]
    assert out["hash"] == Transaction.from_xrpl(tx).get_hash()
    assert len(out["hash"]) == 64 and out["hash"] == out["hash"].upper()
    # the cap was charged the mint price and persisted
    assert signer.spend.spent_drops == 30_000_000
    assert json.loads(signer.spend.path.read_text())["spent_drops"] == 30_000_000


def test_sign_and_submit_accept_offer_reads_the_offer_on_ledger(signer, ledger):
    out = run(signer.sign_and_submit(accept_tx(), ACCEPT))
    assert ledger.calls == ["autofill", ("sell_offers", NFT_ID), "simulate", "submit"]
    tx = decoded_blob(ledger)
    assert tx["TransactionType"] == "NFTokenAcceptOffer"
    assert tx["NFTokenSellOffer"] == OFFER_IDX
    assert_signed_by(signer, tx)
    assert out["hash"] == S.tx_hash(ledger.blob)
    assert signer.spend.spent_drops == 0  # only mint payments count against the cap


def test_sign_and_submit_trustset(signer, ledger):
    run(signer.sign_and_submit(trustset_tx(), TRUST))
    tx = decoded_blob(ledger)
    assert tx["TransactionType"] == "TrustSet" and tx["Flags"] == TF_SET_NO_RIPPLE
    assert tx["LimitAmount"]["issuer"] == BRIX_ISSUER
    assert_signed_by(signer, tx)
    assert signer.spend.spent_drops == 0


def test_sign_and_submit_stops_at_policy(signer, ledger):
    with pytest.raises(S.PolicyError):
        run(signer.sign_and_submit(mint_tx(SendMax="30000000"), MINT))
    assert ledger.calls == ["autofill"]
    ledger.calls.clear()
    with pytest.raises(S.PolicyError, match="owner"):
        ledger.offers[NFT_ID] = [ledger_offer(owner=STRANGER)]
        run(signer.sign_and_submit(accept_tx(), ACCEPT))
    assert ledger.calls == ["autofill", ("sell_offers", NFT_ID)]
    assert signer.spend.spent_drops == 0


def test_sign_and_submit_stops_at_simulate(signer, ledger):
    class SimulateFailed(RuntimeError):
        pass

    ledger.simulate_exc = SimulateFailed("tecUNFUNDED_PAYMENT")
    with pytest.raises(SimulateFailed):
        run(signer.sign_and_submit(mint_tx(), MINT))
    assert ledger.calls == ["autofill", "simulate"]
    assert signer.spend.spent_drops == 0
    ledger.calls.clear()
    ledger.simulate_exc = None
    ledger.simulate_result = {"engine_result": "tecPATH_DRY"}  # a ledger that forgot to raise
    with pytest.raises(S.PreflightError, match="tecPATH_DRY"):
        run(signer.sign_and_submit(mint_tx(), MINT))
    assert ledger.calls == ["autofill", "simulate"]


def test_sign_and_submit_refuses_an_autofill_that_changes_the_tx(signer, ledger):
    def tamper(tx):
        tx["Destination"] = STRANGER

    ledger.tamper = tamper
    with pytest.raises(S.PolicyError, match="autofill"):
        run(signer.sign_and_submit(mint_tx(), MINT))
    assert "submit" not in ledger.calls

    def add_field(tx):
        tx["DestinationTag"] = 5

    ledger.tamper = add_field
    with pytest.raises(S.PolicyError):
        run(signer.sign_and_submit(mint_tx(), MINT))
    assert "submit" not in ledger.calls

    def add_signing_key(tx):
        tx["SigningPubKey"] = "ED" + "11" * 32

    ledger.tamper = add_signing_key
    with pytest.raises(S.PolicyError):
        run(signer.sign_and_submit(mint_tx(), MINT))
    assert "submit" not in ledger.calls


def test_sign_and_submit_accepts_what_a_real_autofill_does(signer, ledger):
    def real_autofill(tx):
        tx["Fee"] = "15"  # overwrote the caller's Fee
        tx["LastLedgerSequence"] = 6000
        tx["SigningPubKey"] = ""  # xrpl-py's placeholder

    ledger.tamper = real_autofill
    run(signer.sign_and_submit(mint_tx(Fee="10"), MINT))
    tx = decoded_blob(ledger)
    assert tx["Fee"] == "15" and tx["LastLedgerSequence"] == 6000
    assert_signed_by(signer, tx)


def test_sign_and_submit_spend_cap_over_several_mints(ledger):
    cfg = FakeConfig(setup_max_xrp=25.0)
    signer = S.Signer(REGULAR_SEED, MASTER, ledger, cfg)
    one = S.Purpose.mint_payment("10", 1, LFG_SIGNER)
    run(signer.sign_and_submit(mint_tx("10000000"), one))
    run(signer.sign_and_submit(mint_tx("10000000"), one))
    with pytest.raises(S.PolicyError, match="FLY_SETUP_MAX_XRP"):
        run(signer.sign_and_submit(mint_tx("10000000"), one))
    assert ledger.calls.count("submit") == 2
    # a fresh signer (new process) reads the same cap state back from disk
    again = S.Signer(REGULAR_SEED, MASTER, ledger, cfg)
    assert again.spend.spent_drops == 20_000_000
    with pytest.raises(S.PolicyError, match="FLY_SETUP_MAX_XRP"):
        again.policy(filled(mint_tx("10000000")), one)


def test_sign_and_submit_rejects_bad_arguments(signer, ledger):
    with pytest.raises(S.PolicyError):
        run(signer.sign_and_submit("Payment", MINT))  # type: ignore[arg-type]
    with pytest.raises(S.PolicyError):
        run(signer.sign_and_submit(mint_tx(), "mint_payment"))  # type: ignore[arg-type]
    assert ledger.calls == []


# ---------------------------------------------------------------- SpendLedger


def test_spend_ledger_persists_under_the_network_dir(_data_dir):
    led = S.SpendLedger("testnet", 1.5)
    assert led.path == _data_dir / "testnet" / "setup-spend.json"
    assert led.cap_drops == 1_500_000 and led.spent_drops == 0 and led.remaining_drops == 1_500_000
    assert not led.path.exists()  # nothing written until a charge
    led.check(1_500_000)
    with pytest.raises(S.PolicyError, match="FLY_SETUP_MAX_XRP"):
        led.check(1_500_001)
    assert not led.path.exists()  # check never writes
    led.charge(1_000_000)
    assert led.spent_drops == 1_000_000 and led.remaining_drops == 500_000
    led.charge(500_000)
    with pytest.raises(S.PolicyError, match="FLY_SETUP_MAX_XRP"):
        led.charge(1)
    data = json.loads(led.path.read_text())
    assert data["network"] == "testnet" and data["spent_drops"] == 1_500_000
    assert [c["drops"] for c in data["charges"]] == [1_000_000, 500_000]
    assert S.SpendLedger("testnet", 1.5).spent_drops == 1_500_000
    # raising the cap later lets setup continue from where it stopped
    assert S.SpendLedger("testnet", 2).remaining_drops == 500_000


def test_spend_ledger_refuses_another_networks_file(_data_dir):
    led = S.SpendLedger("testnet", 5)
    led.charge(1)
    (_data_dir / "mainnet").mkdir(parents=True)
    (_data_dir / "mainnet" / "setup-spend.json").write_text(led.path.read_text())
    with pytest.raises(S.PolicyError, match="network"):
        S.SpendLedger("mainnet", 5).spent_drops  # noqa: B018


def test_spend_ledger_arguments():
    with pytest.raises(ValueError):
        S.SpendLedger("testnet", -1)
    with pytest.raises(ValueError):
        S.SpendLedger("devnet", 1)
    led = S.SpendLedger("testnet", 1)
    with pytest.raises(ValueError):
        led.charge(-1)
    with pytest.raises(ValueError):
        led.charge(1.5)  # type: ignore[arg-type]
    led.charge(0)
    assert led.spent_drops == 0


def test_spend_ledger_ties_a_charge_to_its_sign_request(_data_dir):
    """A charge remembers the sign request it paid (`ref`), so a resumed mint session can
    tell a paid request from an unpaid one and never pays twice (spec §4.2)."""
    paid, other = "wc-" + "a" * 32, "wc-" + "b" * 32
    led = S.SpendLedger("testnet", 5)
    led.charge(1_000_000, ref=paid)
    led.charge(1_000_000)  # a charge with no ref still counts against the cap
    assert led.charged(paid) and not led.charged(other)
    assert led.spent_drops == 2_000_000
    data = json.loads(led.path.read_text())
    assert [c.get("ref") for c in data["charges"]] == [paid, None]
    with pytest.raises(ValueError):
        led.charged("")
    # a spend file from before refs existed reads as "nothing charged under any ref"
    led.path.write_text(json.dumps({"network": "testnet", "spent_drops": 7,
                                    "charges": [{"at": "x", "drops": 7}]}))
    assert S.SpendLedger("testnet", 5).spent_drops == 7
    assert not S.SpendLedger("testnet", 5).charged(paid)


def test_sign_and_submit_charges_under_the_purposes_ref(signer, ledger):
    ref = "wc-" + "c" * 32
    purpose = S.Purpose.mint_payment(Decimal("10"), 3, LFG_SIGNER, ref=ref)
    assert purpose.ref == ref and MINT.ref is None
    run(signer.sign_and_submit(mint_tx(), purpose))
    assert signer.spend.charged(ref) and not signer.spend.charged("wc-" + "d" * 32)
    assert signer.spend.spent_drops == 30_000_000


def test_xrp_to_drops_is_exact():
    assert S.xrp_to_drops(Decimal("10")) == 10_000_000
    assert S.xrp_to_drops(Decimal("0.000001")) == 1
    assert S.xrp_to_drops(Decimal("12.5")) == 12_500_000
    assert S.xrp_to_drops(0) == 0
    with pytest.raises(ValueError):
        S.xrp_to_drops(Decimal("0.0000001"))
    with pytest.raises(ValueError):
        S.xrp_to_drops(Decimal("-1"))
    with pytest.raises(ValueError):
        S.xrp_to_drops(Decimal("NaN"))
