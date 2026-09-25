"""`fly setup`: the testnet rehearsal and the mainnet go-live steps (spec §5.3, §5 step 4, §4.3).

The steps, in the order the rehearsal runs them:

- `keygen`: a fresh RegularKey pair. On testnet it goes into `wallet.json`; on mainnet
  nothing is written, the address is for Xaman and the seed for `~/lfg-fly/.env`.
- `faucet` (testnet): create the master wallet through the public faucet and store its
  seed in `wallet.json`, or top the existing wallet up "as often as the spend cap needs".
- `regular-key` (testnet): `SetRegularKey`, the one transaction the MASTER seed signs
  (§4.3: everything else goes through the RegularKey signer). The seed is read from
  `wallet.json` for that call alone.
- `trustline`, `closet`, `mint`, `harvest`: LFG's public API as an agent user (docs/lfg-api.md),
  every signature through `Signer.sign_and_submit` under the matching §4.3 row
  (`trustset`, `accept_offer`, `mint_payment`), the spend cap (`FLY_SETUP_MAX_XRP`, §4.4)
  stopping the mint loop.
- `status`: wallet, balance, RegularKey, characters, Closet, spend.

`wallet.json` (`FLY_DATA_DIR/testnet/wallet.json`, chmod 600) is the state file of §5.3;
it is refused inside any git repository, the `fee_cover_rehearsal` pattern. The ledger and
LFG client are duck-typed here (`lfg_fly.body.chain.Ledger`, `lfg_fly.body.client.LfgClient`
in production); `cli_setup.py` wires them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import stat
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import aiohttp
from xrpl.core import addresscodec, keypairs
from xrpl.core.binarycodec import encode, encode_for_signing
from xrpl.wallet import Wallet

from lfg_fly import paths
from lfg_fly.body.chain import SimulateFailed, SubmitFailed
from lfg_fly.body.client import LfgClient, LfgError
from lfg_fly.body.config import FlyConfig
from lfg_fly.body.signer import (
    MAX_FEE_DROPS,
    PolicyError,
    PreflightError,
    Purpose,
    Signer,
    tx_hash,
    xrp_to_drops,
)

log = logging.getLogger(__name__)

FAUCET_URL = "https://faucet.altnet.rippletest.net/accounts"
WALLET_FIELDS = ("network", "address", "master_seed", "regular_seed", "regular_address",
                 "created_at")
# The BRIX pair (docs/lfg-api.md "Config values"): mainnet's issuer is a constant; testnet's is
# LFG's SEED address, which is also its signing account, unless FLY_BRIX_ISSUER says otherwise.
BRIX_CURRENCY_HEX = "4252495800000000000000000000000000000000"
MAINNET_BRIX_ISSUER = "rLfgoBriX5ZaMP32mtc7RUZJcjnisKh2Px"
BULK_MINT_MAX = 10  # LFG clamps a bulk job to this (docs/lfg-api.md §7)
CLOSET_RETRY_SECONDS = 5.0
DROPS_PER_XRP = Decimal(1_000_000)
SLOTS = ("Background", "Back", "Body", "Clothing", "Mouth", "Eyebrows", "Eyes", "Head",
         "Accessory")
REGULAR_KEY_FIELDS = frozenset({"TransactionType", "Account", "RegularKey", "Fee", "Sequence",
                                "LastLedgerSequence"})
MINT_PAID = frozenset({"offer_ready", "done", "failed", "payment_timeout", "cancelled"})
MINT_DEAD = frozenset({"failed", "payment_timeout", "cancelled"})
HARVEST_FINAL = frozenset({"done", "failed"})
TRUSTLINE_FINAL = frozenset({"signed", "rejected", "expired"})


class SetupError(RuntimeError):
    """A setup step cannot proceed; the message says which step to run or fix."""


class InsideRepoError(SetupError):
    """`wallet.json` would land inside a git repository (spec §5.3: refused)."""


class SpendCapReached(SetupError):
    """The next mint payment would exceed FLY_SETUP_MAX_XRP (spec §4.3, §4.4)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _xrp(drops: int | str) -> str:
    """Drops as a plain XRP decimal string ("100", "12.5")."""
    return str(Decimal(int(drops)) / DROPS_PER_XRP)


# ---------------------------------------------------------------- wallet.json (§5.3)


def inside_git_repo(path: str | Path) -> bool:
    """True when `path` or any ancestor holds a `.git` entry (a directory, or a worktree's
    file). The path need not exist yet."""
    p = Path(path).expanduser().resolve()
    return any((anc / ".git").exists() for anc in (p, *p.parents))


def _wallet_or_empty(network: str) -> dict:
    if paths.wallet_path(network).exists():
        return read_wallet(network)
    data = dict.fromkeys(WALLET_FIELDS)
    data["network"] = network
    return data


def read_wallet(network: str) -> dict:
    """The testnet wallet file: `{network, address, master_seed, regular_seed, regular_address,
    created_at}`. Refuses a missing file, a mode other than 0600, and a foreign stamp."""
    path = paths.wallet_path(network)
    if network != "testnet":
        raise SetupError(
            f"there is no wallet file on {network}: mainnet keeps its RegularKey seed in "
            "~/lfg-fly/.env (FLY_REGULAR_SEED) and its master key in Xaman (spec §4.4)"
        )
    if not path.exists():
        raise SetupError(f"{path} does not exist: run `fly setup faucet` and `fly setup keygen`")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise SetupError(f"{path} is mode {mode:o}; it holds seeds and must be chmod 600")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise SetupError(f"{path} is not valid JSON") from None
    if not isinstance(data, dict) or data.get("network") != network:
        stamped = data.get("network") if isinstance(data, dict) else None
        raise SetupError(f"{path} is stamped {stamped!r}, not {network!r}")
    return {k: data.get(k) for k in WALLET_FIELDS}


def write_wallet(data: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Write `wallet.json` atomically, owner-only (0600), never inside a git repository.

    The refusal comes before anything is created, so a misconfigured FLY_DATA_DIR leaves no
    trace in the repo. Only `WALLET_FIELDS` are kept.
    """
    network = data.get("network")
    if network not in paths.NETWORKS:
        raise SetupError(f"wallet data must be stamped with a network, not {network!r}")
    target = Path(path) if path is not None else paths.wallet_path(network)
    if inside_git_repo(target):
        raise InsideRepoError(
            f"{target} is inside a git repository: wallet.json never lives in a repo (spec §5.3); "
            "point FLY_DATA_DIR outside every checkout"
        )
    text = json.dumps({k: data.get(k) for k in WALLET_FIELDS}, indent=2) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)  # whatever the umask did
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    return target


# ---------------------------------------------------------------- keygen


def keygen(cfg: FlyConfig, *, force: bool = False) -> dict:
    """A new RegularKey pair (spec §5 step 2; §5.3 "a RegularKey is set by script").

    Testnet: merged into `wallet.json` (the master fields stay as they are, or None until
    `faucet` runs); a second keygen is refused without `force`, because a new key orphans the
    one set on-ledger. Mainnet: nothing is written; the caller shows the address (for Xaman)
    and the seed (for `~/lfg-fly/.env`) once.
    """
    key = Wallet.create()
    if cfg.network == "mainnet":
        return {"network": "mainnet", "regular_address": key.address, "regular_seed": key.seed}
    data = _wallet_or_empty("testnet")
    if data["regular_seed"] and not force:
        raise SetupError(
            f"wallet.json already holds a RegularKey ({data['regular_address']}); a new one "
            "would orphan the key set on-ledger. Pass --force to replace it, then re-run "
            "`fly setup regular-key`"
        )
    data.update(regular_seed=key.seed, regular_address=key.address,
                created_at=data["created_at"] or _now())
    path = write_wallet(data)
    log.info("RegularKey %s written to %s", key.address, path)
    return {"network": "testnet", "regular_address": key.address, "path": str(path)}


# ---------------------------------------------------------------- faucet (testnet)


async def _http_post_json(url: str, body: dict, timeout: float = 30.0) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(url, json=body) as resp:
            if resp.status // 100 != 2:
                raise SetupError(f"{url} answered HTTP {resp.status}")
            data = await resp.json(content_type=None)
    if not isinstance(data, dict):
        raise SetupError(f"{url} did not answer a JSON object")
    return data


async def _wait_funded(ledger: Any, address: str, sleep: Callable[[float], Awaitable[None]],
                       wait: float, every: float = 2.0) -> bool:
    for attempt in range(max(1, int(wait / every))):
        if attempt:
            await sleep(every)
        if await ledger.account_info(address) is not None:
            return True
    return False


async def faucet(cfg: FlyConfig, *, post: Callable[[str, dict], Awaitable[dict]] | None = None,
                 ledger: Any = None, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 wait: float = 60.0) -> dict:
    """Fund the testnet wallet from the public faucet (spec §5.3 "faucet-funded as often as
    the spend cap needs").

    Without a wallet yet: `POST {}` creates one, and its master seed is on disk (0600) before
    this returns. With one: `POST {"destination"}` tops it up. `post(url, body) -> dict` is
    the HTTP call, injected so tests never reach the network. With a `ledger`, waits until
    `account_info` sees the account (`funded`); otherwise `funded` is None.
    """
    if cfg.network != "testnet":
        raise SetupError("the faucet funds testnet only; a mainnet wallet is created in Xaman "
                         "(spec §5 step 1)")
    post = post or _http_post_json
    data = _wallet_or_empty("testnet")
    existing = data["address"]
    answer = await post(FAUCET_URL, {"destination": existing} if existing else {})
    account = answer.get("account") if isinstance(answer, dict) else None
    address = None
    if isinstance(account, dict):
        address = account.get("classicAddress") or account.get("address")
    if not isinstance(address, str) or not addresscodec.is_valid_classic_address(address):
        raise SetupError("the faucet did not return an account")
    created = existing is None
    if created:
        seed = answer.get("seed")
        if not isinstance(seed, str) or not seed:
            raise SetupError("the faucet did not return a seed for the new account")
        try:
            derived = Wallet.from_seed(seed).address
        except Exception:  # never echo the seed
            raise SetupError("the faucet's seed is not a valid family seed") from None
        if derived != address:
            raise SetupError("the faucet's seed does not derive the account it reports")
        data.update(address=address, master_seed=seed, created_at=data["created_at"] or _now())
        write_wallet(data)
        log.info("testnet wallet %s created; seed stored in %s", address,
                 paths.wallet_path("testnet"))
    elif address != existing:
        raise SetupError(f"the faucet funded {address}, not the wallet {existing}")
    funded = await _wait_funded(ledger, address, sleep, wait) if ledger is not None else None
    return {"network": "testnet", "address": address, "amount": answer.get("amount"),
            "created": created, "funded": funded}


# ---------------------------------------------------------------- regular-key (testnet)


def _check_regular_key_tx(tx: dict, account: str, regular: str) -> None:
    """The one master-signed transaction is exactly `SetRegularKey {Account, RegularKey}` plus
    the autofill fields (spec §4.3: anything else is refused)."""
    extra = set(tx) - REGULAR_KEY_FIELDS
    if extra:
        raise PolicyError(f"SetRegularKey carries unexpected field(s) {sorted(extra)}")
    if tx.get("TransactionType") != "SetRegularKey":
        raise PolicyError("the master key signs SetRegularKey and nothing else")
    if tx.get("Account") != account:
        raise PolicyError("SetRegularKey Account must be the fly's wallet")
    if tx.get("RegularKey") != regular:
        raise PolicyError("SetRegularKey RegularKey must be wallet.json's regular_address")
    fee = tx.get("Fee")
    if not isinstance(fee, str) or not fee.isdigit() or not 0 < int(fee) <= MAX_FEE_DROPS:
        raise PolicyError(f"Fee must be a digit string in 1..{MAX_FEE_DROPS} drops, got {fee!r}")
    for key in ("Sequence", "LastLedgerSequence"):
        v = tx.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise PolicyError(f"{key} must be a positive integer, got {v!r}")


def _sign_with_master(tx: dict, master_seed: str, account: str) -> str:
    """Sign `tx` with the master seed at the codec level and return the blob. The key material
    lives in this frame only."""
    try:
        public, private = keypairs.derive_keypair(master_seed)
    except Exception:
        raise SetupError("wallet.json's master_seed is not a valid family seed") from None
    if keypairs.derive_classic_address(public) != account:
        raise SetupError("wallet.json's master_seed does not belong to its address")
    signed = dict(tx)
    signed["SigningPubKey"] = public.upper()
    signed["TxnSignature"] = keypairs.sign(bytes.fromhex(encode_for_signing(signed)), private)
    return encode(signed)


async def set_regular_key(cfg: FlyConfig, ledger: Any, *,
                          sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                          verify_attempts: int = 5) -> dict:
    """`SetRegularKey` on the testnet wallet, signed by the master seed, once (spec §4.3, §5.3).

    Idempotent: when the validated AccountRoot already carries the key, nothing is sent.
    Otherwise autofill -> field check -> `simulate` pre-flight -> sign (master, this frame
    only) -> `submit_and_wait` -> re-read `account_info` until it shows the key.
    """
    if cfg.network != "testnet":
        raise SetupError("SetRegularKey by script is a testnet step; on mainnet set the "
                         "RegularKey from Xaman (spec §5 step 2)")
    data = read_wallet("testnet")
    address, regular = data["address"], data["regular_address"]
    if not regular or not data["regular_seed"]:
        raise SetupError("wallet.json has no RegularKey: run `fly setup keygen` first")
    if not address or not data["master_seed"]:
        raise SetupError("wallet.json has no funded master wallet: run `fly setup faucet` first")
    info = await ledger.account_info(address)
    if info is None:
        raise SetupError(f"{address} is not on the testnet ledger: run `fly setup faucet` first")
    if info["account_data"].get("RegularKey") == regular:
        return {"state": "already_set", "regular_key": regular}
    tx = await ledger.autofill(
        {"TransactionType": "SetRegularKey", "Account": address, "RegularKey": regular}
    )
    _check_regular_key_tx(tx, address, regular)
    await ledger.simulate(dict(tx))
    blob = _sign_with_master(tx, data.pop("master_seed"), address)
    del data
    result = await ledger.submit_and_wait(blob)
    digest = tx_hash(blob)
    reported = result.get("hash") if isinstance(result, dict) else None
    if isinstance(reported, str) and reported.upper() != digest:
        raise RuntimeError(f"ledger reported hash {reported} for a blob hashing to {digest}")
    for attempt in range(verify_attempts):
        if attempt:
            await sleep(2.0)
        info = await ledger.account_info(address)
        if info is not None and info["account_data"].get("RegularKey") == regular:
            log.info("RegularKey %s set on %s (%s)", regular, address, digest)
            return {"state": "set", "regular_key": regular, "hash": digest}
    raise SetupError(f"SetRegularKey {digest} validated but account_info does not show the key "
                     "yet; re-run `fly setup regular-key` (it is idempotent)")


# ---------------------------------------------------------------- the signer and the pins


def wallet_address(cfg: FlyConfig, env: Mapping[str, str] | None = None) -> str:
    """The fly's wallet: testnet reads `wallet.json`; mainnet reads FLY_WALLET (the contract
    has no field for it; the RegularKey seed alone cannot name its wallet)."""
    if cfg.network == "testnet":
        data = read_wallet("testnet")
        if not data["address"]:
            raise SetupError("wallet.json has no address: run `fly setup faucet` first")
        return data["address"]
    env = os.environ if env is None else env
    raw = (env.get("FLY_WALLET") or "").strip()
    if not raw or not addresscodec.is_valid_classic_address(raw):
        raise SetupError("FLY_WALLET must be the fly's mainnet wallet (a classic r-address), "
                         "the account whose RegularKey FLY_REGULAR_SEED is")
    return raw


def build_signer(cfg: FlyConfig, ledger: Any, *, wallet: str | None = None,
                 env: Mapping[str, str] | None = None) -> Signer:
    """The RegularKey signer (spec §4.3): testnet from `wallet.json`, mainnet from
    FLY_REGULAR_SEED plus `wallet` (or FLY_WALLET)."""
    if cfg.network == "testnet":
        data = read_wallet("testnet")
        if not data["regular_seed"]:
            raise SetupError("wallet.json has no RegularKey: run `fly setup keygen` first")
        if not data["address"]:
            raise SetupError("wallet.json has no address: run `fly setup faucet` first")
        return Signer(data["regular_seed"], data["address"], ledger, cfg)
    if not cfg.regular_seed:
        raise SetupError("FLY_REGULAR_SEED is not set: mainnet keeps the RegularKey seed in "
                         "~/lfg-fly/.env (spec §4.4)")
    return Signer(cfg.regular_seed, wallet or wallet_address(cfg, env=env), ledger, cfg)


def brix_pair(cfg: FlyConfig, env: Mapping[str, str] | None = None) -> tuple[str, str]:
    """(currency, issuer) of the BRIX line the TrustSet row pins (spec §4.3): FLY_BRIX_CURRENCY
    / FLY_BRIX_ISSUER, else the mainnet constant, else (testnet) LFG's signing account."""
    env = os.environ if env is None else env
    currency = (env.get("FLY_BRIX_CURRENCY") or "").strip() or BRIX_CURRENCY_HEX
    issuer = (env.get("FLY_BRIX_ISSUER") or "").strip()
    if not issuer:
        issuer = MAINNET_BRIX_ISSUER if cfg.network == "mainnet" else (cfg.signing_account or "")
    if not issuer or not addresscodec.is_valid_classic_address(issuer):
        raise SetupError("the BRIX issuer is unknown: set FLY_BRIX_ISSUER (on testnet it is "
                         "LFG's SEED address, also FLY_LFG_SIGNING_ACCOUNT)")
    return currency, issuer


def pick_hero(characters: list[dict]) -> str | None:
    """The loop's default hero (spec §4.1 step 5): the first mutable, non-blank male."""
    for c in characters:
        if c.get("body") == "male" and c.get("mutable") and not c.get("blank"):
            return c["nft_id"]
    return None


def _look(attributes: Any) -> list[str]:
    got = {}
    if isinstance(attributes, list):
        got = {a.get("trait_type"): a.get("value") for a in attributes if isinstance(a, dict)}
    return [str(got.get(slot) or "None") for slot in SLOTS]


# ---------------------------------------------------------------- the signed-in steps


@dataclass
class Setup:
    """One signed-in setup session: `trustline`, `closet`, `mint`, `harvest`, `status`.

    `signer_factory(cfg) -> Signer` builds the RegularKey signer; it is called again when a
    testnet run pins LFG's signing account from the first mint session (`cfg` is replaced).
    `sleep` and `poll_every` are injectable so tests run without waiting.
    """

    cfg: FlyConfig
    ledger: Any
    client: Any
    signer_factory: Callable[[FlyConfig], Any]
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    poll_every: float = 3.0
    payment_timeout: float = 420.0  # LFG detects a payment within 300 s (PAYMENT_TIMEOUT_SECONDS)
    harvest_timeout: float = 600.0
    brix_currency: str | None = None
    brix_issuer: str | None = None
    signer: Any = field(init=False)
    pinned_signing_account: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.signer = self.signer_factory(self.cfg)

    # -- plumbing ---------------------------------------------------------------------------

    def _delivery_owners(self) -> frozenset[str]:
        """Besides the NFTokenID's issuer, LFG's signing account may own a delivery offer
        (an authorized minter); it is pinned config, never something LFG told us today."""
        return frozenset({self.cfg.signing_account}) if self.cfg.signing_account else frozenset()

    async def _poll(self, getter: Callable[[str], Awaitable[dict]], sid: str,
                    done: Callable[[dict], bool], timeout: float, what: str) -> dict:
        deadline = time.monotonic() + timeout
        max_polls = int(timeout / max(self.poll_every, 0.25)) + 1
        polls = 0
        while True:
            status = await getter(sid)
            if done(status):
                return status
            polls += 1
            if polls >= max_polls or time.monotonic() >= deadline:
                raise SetupError(f"{what} {sid} is still {status.get('state')!r} after "
                                 f"{timeout:g}s")
            await self.sleep(self.poll_every)

    async def _reject(self, sid: str, why: str) -> None:
        log.warning("rejecting sign request %s: %s", sid, why)
        try:
            await self.client.sign_result(sid, rejected=True)
        except LfgError as e:
            log.warning("could not reject %s: %s", sid, e)

    async def _report_error(self, sid: str, exc: BaseException) -> None:
        try:
            await self.client.sign_result(sid, error=str(exc)[:200])
        except LfgError as e:
            log.warning("could not report %s: %s", sid, e)

    async def _sign_request(self, sid: str, purpose_for: Callable[[dict], Purpose]) -> str:
        """GET the sign request, sign and submit its txjson under `purpose_for(txjson)`, and
        POST the hash (docs/lfg-api.md §8). A policy refusal is told to LFG as `rejected`, a
        pre-flight or final ledger failure as `error`; an unknown outcome is told nothing."""
        req = await self.client.sign_request(sid)
        state = req.get("state")
        if state != "pending":
            raise SetupError(f"sign request {sid} is {state!r}, not pending")
        tx = req.get("txjson")
        if not isinstance(tx, dict):
            raise SetupError(f"sign request {sid} carries no txjson")
        purpose = purpose_for(tx)
        try:
            signed = await self.signer.sign_and_submit(tx, purpose)
        except PolicyError as e:
            await self._reject(sid, str(e))
            raise
        except (PreflightError, SimulateFailed) as e:
            await self._report_error(sid, e)
            raise
        except SubmitFailed as e:
            if (e.engine_result or "")[:3] in ("tem", "tef", "tec"):
                await self._report_error(sid, e)
            raise
        digest = signed["hash"]
        await self.client.sign_result(sid, tx_hash=digest)
        return digest

    # -- trustline (§4.3 TrustSet; docs/lfg-api.md §6) ---------------------------------------

    async def trustline(self) -> dict:
        """The BRIX line: `POST /api/brix/trustline`, sign the TrustSet under `trustset` with
        the pinned pair, report the hash, wait for `signed`."""
        r = await self.client.brix_trustline()
        if r.get("state") == "already_set":
            return {"state": "already_set"}
        sid = r.get("uuid") or LfgClient.sign_id(r["xumm_url"])
        currency, issuer = ((self.brix_currency or BRIX_CURRENCY_HEX, self.brix_issuer)
                            if self.brix_issuer else brix_pair(self.cfg))
        digest = await self._sign_request(sid, lambda tx: Purpose.trustset(currency, issuer))
        status = await self._poll(self.client.brix_trustline_status, sid,
                                  lambda s: s.get("state") in TRUSTLINE_FINAL,
                                  self.payment_timeout, "trustline")
        if status.get("state") != "signed":
            raise SetupError(f"trustline {sid} ended {status.get('state')!r} after {digest}")
        return {"state": "signed", "tx_hash": status.get("tx_hash") or digest}

    # -- closet (§4.3 NFTokenAcceptOffer; docs/lfg-api.md §5) --------------------------------

    async def closet(self, *, attempts: int = 12) -> dict:
        """`POST /api/closet` until `active`, signing the accept under `accept_offer` once. A
        503 `closet_mint_transient` is retried; a 502 is indeterminate and stops."""
        signed: set[str] = set()
        for attempt in range(attempts):
            try:
                r = await self.client.closet()
            except LfgError as e:
                if e.status == 503 and attempt + 1 < attempts:
                    log.warning("closet: %s; retrying in %gs", e, CLOSET_RETRY_SECONDS)
                    await self.sleep(CLOSET_RETRY_SECONDS)
                    continue
                if e.status == 502:
                    raise SetupError(f"POST /api/closet answered 502 ({e.code}): indeterminate; "
                                     "check /api/economy before retrying") from e
                raise
            status = r.get("status")
            if status == "active":
                return {"status": "active", "nft_id": r.get("nft_id"), "accepted": bool(signed)}
            if status != "pending_accept" or not r.get("accept"):
                raise SetupError(f"unexpected closet answer {r!r}")
            sid = LfgClient.sign_id(r["accept"])
            if sid in signed:  # accepted on-ledger; LFG has not seen ownership yet
                await self.sleep(self.poll_every)
                continue
            nft_id = r.get("nft_id")
            owners = self._delivery_owners()
            await self._sign_request(
                sid, lambda tx, n=nft_id, o=owners: Purpose.accept_offer(n, o))
            signed.add(sid)
        raise SetupError(f"the Closet did not become active after {attempts} attempts")

    # -- mint (§4.3 Payment (mint) + NFTokenAcceptOffer; docs/lfg-api.md §7) -----------------

    async def mint(self, count: int, *, bulk: bool = False, body: str | None = None) -> dict:
        """Mint donors through LFG until `count` of them (of `body`, when given) have arrived,
        or the spend cap stops it (spec §5.3: bodies come by live-share odds, so the loop
        keeps minting until the target or the cap). One session at a time, or bulk jobs of up
        to BULK_MINT_MAX. Returns what happened; a server refusal ends the loop, a policy or
        session failure raises.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        if self.cfg.signing_account is None and self.cfg.network != "testnet":
            raise SetupError("FLY_LFG_SIGNING_ACCOUNT is not set: LFG's mint destination must be "
                             "pinned on mainnet (spec §4.3)")
        minted: list[dict] = []
        obtained = 0
        stopped, reason = "count", None
        unit_price: Decimal | None = None
        while obtained < count:
            qty = min(count - obtained, BULK_MINT_MAX) if bulk else 1
            spend = self.signer.spend
            need = xrp_to_drops(unit_price * qty) if unit_price is not None else 1
            if spend.remaining_drops < need:
                stopped = "spend_cap"
                reason = (f"setup spend cap: {spend.spent_drops} drops spent of FLY_SETUP_MAX_XRP "
                          f"({spend.cap_drops} drops) on {self.cfg.network}; the next "
                          f"{qty} mint(s) need {need} drops")
                break
            try:
                batch = await (self._mint_bulk(qty) if bulk else self._mint_one())
            except SpendCapReached as e:
                stopped, reason = "spend_cap", str(e)
                break
            except LfgError as e:
                stopped, reason = "refused", f"LFG {e.status} {e.code or e.body}"
                break
            for m in batch:
                minted.append(m)
                if body is None or m.get("body") == body:
                    obtained += 1
                if m.get("unit_price") is not None:
                    unit_price = m["unit_price"]
        return {
            "requested": count, "body": body, "obtained": obtained, "stopped": stopped,
            "reason": reason, "minted": [{k: v for k, v in m.items() if k != "unit_price"}
                                         for m in minted],
            "spent_xrp": _xrp(self.signer.spend.spent_drops),
            "pinned_signing_account": self.pinned_signing_account,
        }

    def _mint_destination(self, tx: dict) -> str:
        """The pinned destination; on testnet only, an unpinned config learns it from the first
        payment's txjson and pins it for the rest of the run (a disclosed deviation)."""
        if self.cfg.signing_account:
            return self.cfg.signing_account
        if self.cfg.network != "testnet":
            raise SetupError("FLY_LFG_SIGNING_ACCOUNT is not set: mint refused (spec §4.3)")
        dest = tx.get("Destination")
        if not isinstance(dest, str) or not addresscodec.is_valid_classic_address(dest):
            raise SetupError(f"mint txjson has no valid Destination: {dest!r}")
        log.warning("testnet: pinning LFG's signing account to %s from the mint session; set "
                    "FLY_LFG_SIGNING_ACCOUNT to pin it yourself", dest)
        self.cfg = replace(self.cfg, signing_account=dest)
        self.signer = self.signer_factory(self.cfg)
        self.pinned_signing_account = dest
        return dest

    @staticmethod
    def _mint_purpose(total: Decimal, quantity: int, dest: str, pay_with: str) -> Purpose:
        """pay_amount x quantity (spec §4.3): the per-unit price when the total divides evenly,
        else the total once."""
        per = total / quantity
        if per * quantity == total:
            with contextlib.suppress(ValueError):
                xrp_to_drops(per)
                return Purpose.mint_payment(per, quantity, dest, pay_with=pay_with)
        return Purpose.mint_payment(total, 1, dest, pay_with=pay_with)

    async def _pay(self, session: dict, quantity: int, what: str) -> Decimal | None:
        """Sign the mint or bulk payment under `mint_payment`; returns the per-unit XRP price,
        or None when the session is sponsored (no payment link)."""
        link = session.get("payment_link")
        if not link:
            return None
        pay_sid = LfgClient.sign_id(link)
        pay_with = session.get("pay_with")
        if pay_with != "XRP":
            await self._reject(pay_sid, f"the fly pays XRP only, LFG quoted {pay_with}")
            raise SetupError(f"{what} {session.get('id')} wants {pay_with!r}: the fly pays XRP "
                             "only (spec §4.3); is an LFGO line funded on this wallet?")
        try:
            total = Decimal(str(session.get("pay_amount")))
            drops = xrp_to_drops(total)
        except (InvalidOperation, ValueError) as e:
            await self._reject(pay_sid, "unreadable price")
            raise SetupError(f"{what} price {session.get('pay_amount')!r} is not XRP: {e}") from e
        try:
            self.signer.spend.check(drops)
        except PolicyError as e:
            await self._reject(pay_sid, "over FLY_SETUP_MAX_XRP")
            raise SpendCapReached(str(e)) from e
        await self._sign_request(
            pay_sid,
            lambda tx: self._mint_purpose(total, quantity, self._mint_destination(tx), pay_with),
        )
        return total / quantity

    async def _mint_one(self) -> list[dict]:
        try:
            s = await self.client.mint()
        except LfgError as e:
            if e.status == 409 and "in progress" in str(e.code or ""):
                s = await self.client.mint_active()  # a crashed run left it; resume
                log.info("resuming mint session %s (%s)", s.get("id"), s.get("state"))
            else:
                raise
        sid = s["id"]
        unit_price = None
        if s.get("state") == "awaiting_payment":
            unit_price = await self._pay(s, 1, "mint")
        s = await self._poll(self.client.mint_status, sid,
                             lambda st: st.get("state") in MINT_PAID, self.payment_timeout,
                             "mint")
        if s["state"] in MINT_DEAD:
            raise SetupError(f"mint {sid} ended {s['state']!r}: "
                             f"{s.get('error') or s.get('reason')}")
        if s["state"] == "offer_ready" and not s.get("accept_signed"):
            link = s.get("accept_deeplink")
            if not link:
                raise SetupError(f"mint {sid} is offer_ready without an accept_deeplink")
            nft_id, owners = s["nft_id"], self._delivery_owners()
            await self._sign_request(LfgClient.sign_id(link),
                                     lambda tx: Purpose.accept_offer(nft_id, owners))
            s = await self.client.mint_status(sid)
        log.info("minted #%s %s (%s)", s.get("nft_number"), s.get("nft_id"), s.get("body_type"))
        return [{"session": sid, "nft_id": s.get("nft_id"), "number": s.get("nft_number"),
                 "body": s.get("body_type"), "state": s.get("state"), "unit_price": unit_price}]

    async def _already_owned(self, nft_id: str) -> bool:
        if self.ledger is None or not nft_id:
            return False
        want = nft_id.upper()
        entries = await self.ledger.account_nfts(self.signer.account)
        return any(str(e.get("NFTokenID", "")).upper() == want for e in entries)

    async def _mint_bulk(self, quantity: int) -> list[dict]:
        job = await self.client.bulk_mint(quantity)
        jid = job["id"]
        unit_price = None
        if job.get("state") == "awaiting_payment":
            unit_price = await self._pay(job, int(job.get("quantity") or quantity), "bulk mint")

        def settled(j: dict) -> bool:
            units = j.get("units") or []
            return j.get("state") in ("done", "failed") or bool(units) and all(
                u.get("state") in ("offered", "failed") for u in units
            )

        job = await self._poll(self.client.bulk_status, jid, settled, self.payment_timeout,
                               "bulk mint")
        out: list[dict] = []
        owners = self._delivery_owners()
        for unit in job.get("units") or []:
            if unit.get("state") != "offered" or unit.get("accepted"):
                if unit.get("state") == "failed":
                    log.warning("bulk %s unit %s failed: %s", jid, unit.get("index"),
                                unit.get("error"))
                continue
            nft_id = unit.get("nft_id")
            link = await self.client.bulk_unit_accept(jid, int(unit["index"]))
            try:
                await self._sign_request(LfgClient.sign_id(link["link"]),
                                         lambda tx, n=nft_id: Purpose.accept_offer(n, owners))
            except PolicyError:
                if not await self._already_owned(nft_id):
                    raise
                log.info("bulk %s unit %s already accepted", jid, unit.get("index"))
            out.append({"job": jid, "index": unit.get("index"), "nft_id": nft_id,
                        "number": unit.get("nft_number"), "body": unit.get("body_type"),
                        "state": "accepted", "unit_price": unit_price})
        return out

    # -- harvest (docs/lfg-api.md §4) --------------------------------------------------------

    async def harvest(self, nft_ids: list[str] | None = None, *, all_: bool = False,
                      hero: str | None = None) -> dict:
        """Harvest donors into the Closet: `--all` takes every mutable, non-blank character
        but the hero (given, else the loop's default pick); explicit ids refuse the hero. A
        legacy (burn + remint) harvest's accept is signed under `accept_offer`. Sessions are
        run one at a time (LFG allows one economy action at a time)."""
        if bool(nft_ids) == bool(all_):
            raise SetupError("harvest takes either --all or explicit nft ids")
        eco = await self.client.economy()
        characters = eco.get("characters") or []
        by_id = {c["nft_id"]: c for c in characters}
        token = (eco.get("closet") or {}).get("token") or {}
        if token.get("status") != "active":
            raise SetupError("the Closet is not active: run `fly setup closet` first")
        hero_id = hero or pick_hero(characters)
        if all_:
            targets = [c["nft_id"] for c in characters if c["nft_id"] != hero_id]
        else:
            targets = list(nft_ids or [])
        harvested: list[dict] = []
        skipped: list[dict] = []
        for nft_id in targets:
            c = by_id.get(nft_id)
            if nft_id == hero_id:
                raise SetupError(f"{nft_id} is the hero; it is never harvested")
            if c is None:
                skipped.append({"nft_id": nft_id, "why": "not owned"})
            elif c.get("blank"):
                skipped.append({"nft_id": nft_id, "why": "already blank"})
            elif all_ and not c.get("mutable"):
                skipped.append({"nft_id": nft_id, "why": "not mutable"})  # spec §5: donors are
            else:
                # An explicit id may be a legacy (burnable, not mutable) character: /api/economy
                # says only `mutable`, so LFG's can_harvest decides and the burn + remint path
                # runs (docs/lfg-api.md §4); a 400 is reported as skipped, not raised.
                try:
                    harvested.append(await self._harvest_one(nft_id))
                except LfgError as e:
                    if e.status != 400:
                        raise
                    skipped.append({"nft_id": nft_id, "why": f"LFG: {e.code or e.body}"})
        return {"hero": hero_id, "harvested": harvested, "skipped": skipped}

    async def _harvest_one(self, nft_id: str) -> dict:
        status = await self.client.harvest(nft_id)
        sid = status["id"]
        deadline = time.monotonic() + self.harvest_timeout
        max_polls = int(self.harvest_timeout / max(self.poll_every, 0.25)) + 1
        signed = False
        polls = 0
        while True:
            if status.get("accept") and not signed:
                new_id, owners = status.get("new_nft_id"), self._delivery_owners()
                if not new_id:
                    raise SetupError(f"harvest {sid} offers an accept without a new_nft_id")
                await self._sign_request(
                    LfgClient.sign_id(status["accept"]),
                    lambda tx, n=new_id, o=owners: Purpose.accept_offer(n, o))
                signed = True
            if status.get("state") in HARVEST_FINAL:
                break
            polls += 1
            if polls >= max_polls or time.monotonic() >= deadline:
                raise SetupError(f"harvest {sid} is still {status.get('state')!r} after "
                                 f"{self.harvest_timeout:g}s")
            await self.sleep(self.poll_every)
            status = await self.client.harvest_status(sid)
        log.info("harvest %s of %s: %s (%d assets)", sid, nft_id, status.get("state"),
                 len(status.get("moved_assets") or []))
        return {"nft_id": nft_id, "session": sid, "state": status.get("state"),
                "error": status.get("error"),
                "moved": len(status.get("moved_assets") or []),
                "new_nft_id": status.get("new_nft_id")}

    # -- status ------------------------------------------------------------------------------

    async def status(self) -> dict:
        """Wallet, balance and RegularKey from the ledger; characters, Closet and BRIX from
        LFG; the setup spend. Never a seed."""
        wallet = self.signer.account
        balance = regular = None
        if self.ledger is not None:
            info = await self.ledger.account_info(wallet)
            if info is not None:
                data = info.get("account_data") or {}
                balance = _xrp(data.get("Balance", 0))
                regular = data.get("RegularKey")
        out: dict[str, Any] = {
            "network": self.cfg.network, "api_base": self.cfg.api_base, "wallet": wallet,
            "balance_xrp": balance, "regular_key": regular,
            "key_address": self.signer.key_address,
            "regular_key_matches": regular == self.signer.key_address,
            "spend": {"spent_xrp": _xrp(self.signer.spend.spent_drops),
                      "cap_xrp": _xrp(self.signer.spend.cap_drops)},
        }
        try:
            eco = await self.client.economy()
        except LfgError as e:
            out["characters"], out["closet"] = [], {"error": str(e)}
        else:
            out["characters"] = [
                {"nft_id": c.get("nft_id"), "body": c.get("body"), "mutable": c.get("mutable"),
                 "blank": c.get("blank"), "look": _look(c.get("attributes"))}
                for c in eco.get("characters") or []
            ]
            closet = eco.get("closet") or {}
            token = closet.get("token") or {}
            out["closet"] = {"status": token.get("status"), "nft_id": token.get("nft_id"),
                             "units": sum(int(a.get("count") or 0)
                                          for a in closet.get("assets") or [])}
        try:
            out["brix"] = await self.client.brix()
        except LfgError as e:
            out["brix"] = {"error": str(e)}
        return out
