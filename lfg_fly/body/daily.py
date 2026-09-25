"""The fly's day outside the move loop: fly-claim and fly-retrain (spec §4 table).

- `claim` (04:10 UTC): `POST /api/brix/claim`, then poll `GET /api/brix/claim/{id}` until
  the payout is `confirmed` or `failed`. LFG's staging stack cannot pay BRIX today (spec
  §5.3: no `BRIX_DISTRIBUTOR_SEED`), so a `503 claims_disabled` is the operator's
  prerequisite, not the fly's failure: an outbox alert, a clear message, exit 0.
- `retrain` (04:30 UTC): the live supply from `GET /api/rarity/supply`, saved as a stamped
  snapshot named by the sha256 of its canonical JSON; a sample of looks from the catalog
  (`teacher.sample.base_look`, N = 2000) simulated under the live concentrations
  (`readout.concentrations`); a new rarity head (`readout.fit_rarity` on
  `FlyBrain.features(looks, conc)` against `readout.nft_rarity`) saved at
  `paths.rarity_head_path(network, snapshot_hash)` (spec §2 Readout, rarity head).

Both sign in fresh with the `agent` provider and the RegularKey proof, hold the token in
memory only, and log out in `finally` (spec §4.1 step 1, via `LfgClient.__aexit__`); both
run the §4.4 chain-identity check, once per process (`ensure_identity`). The session is
closed before any simulation starts, so the token lives only as long as the reads need.
Nothing here signs anything but the sign-in proof, and the seed is read (testnet
`wallet.json`, mainnet `FLY_REGULAR_SEED`) and never written or logged.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
from collections.abc import AsyncIterator, Mapping, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from xrpl.core import addresscodec

from lfg_fly import paths
from lfg_fly.body import outbox
from lfg_fly.body.chain import ChainIdentity, Ledger
from lfg_fly.body.client import LfgClient, LfgError
from lfg_fly.body.config import FlyConfig
from lfg_fly.body.records import Stamp, StampMismatch
from lfg_fly.body.signer import Signer
from lfg_fly.brain import readout as R
from lfg_fly.teacher.sample import base_look

log = logging.getLogger(__name__)

Look = tuple[str, ...]

DEFAULT_N_LOOKS = 2000
DEFAULT_LAM = 1.0
HELDOUT_FRAC = 0.2  # of the sampled looks, for the reported R²; the saved head uses them all
LATEST_HEAD = "latest-rarity-head.json"  # snapshots_dir/<this>: the pointer the loop reads
CLAIM_TERMINAL = frozenset({"confirmed", "failed"})
CLAIM_EXIT_OK = frozenset({"confirmed", "nothing", "disabled"})
WALLET_ACCOUNT_KEYS = ("account", "address", "wallet", "classic_address", "master_address")
WALLET_SEED_KEYS = ("regular_seed", "regular_key_seed")
_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DailyError(RuntimeError):
    """Base of what this module raises on its own account."""


class CredentialsError(DailyError):
    """The fly's wallet or RegularKey seed cannot be found or is malformed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ .env and credentials


def load_env_file(path: str | Path, env: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Load `KEY=VALUE` lines of a dotenv file into `env` (default `os.environ`), setting only
    keys the environment does not already have; returns the keys it set.

    `~/lfg-fly/.env` holds `FLY_REGULAR_SEED` (spec §4.4) and the FLY_* configuration; the
    contract has the CLI load it, never `load_config`. Blank lines, `#` comments, a leading
    `export `, single or double quotes and a trailing ` # comment` are understood; anything
    else is skipped. Values are never logged.
    """
    if env is None:
        env = os.environ
    path = Path(path)
    if not path.is_file():
        return {}
    added: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _ENV_KEY.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if key not in env:
            env[key] = value
            added[key] = value
    return added


@dataclass(frozen=True)
class Credentials:
    """The fly's wallet and the RegularKey seed that signs for it. The seed is excluded from
    `repr`/`str` (spec §4.4: never logged)."""

    account: str
    seed: str = field(repr=False)


def _wallet_file(path: Path, network: str) -> Credentials:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        # the file may hold seeds: name the failure, never its contents
        raise CredentialsError(f"{path} is not readable JSON ({type(e).__name__})") from None
    if not isinstance(doc, dict):
        raise CredentialsError(f"{path} is not a JSON object")
    if doc.get("network") not in (None, network):
        raise CredentialsError(f"{path} is stamped {doc.get('network')!r}, the fly runs {network}")
    account = next((doc[k] for k in WALLET_ACCOUNT_KEYS if isinstance(doc.get(k), str)), None)
    seed = next((doc[k] for k in WALLET_SEED_KEYS if isinstance(doc.get(k), str)), None)
    if seed is None:
        for nested in ("regular_key", "regular"):
            inner = doc.get(nested)
            if isinstance(inner, dict) and isinstance(inner.get("seed"), str):
                seed = inner["seed"]
                break
    if not seed:
        raise CredentialsError(f"{path} has no RegularKey seed ({'/'.join(WALLET_SEED_KEYS)}): "
                               "run `fly setup keygen`")
    if not account:
        raise CredentialsError(f"{path} names no wallet ({'/'.join(WALLET_ACCOUNT_KEYS)})")
    with contextlib.suppress(OSError):
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            log.warning("%s is mode %o; it holds a seed and should be 600", path, mode)
    return Credentials(account=account, seed=seed)


def load_credentials(cfg: FlyConfig, env: Mapping[str, str] | None = None) -> Credentials:
    """Where the fly's key lives (contract §config: testnet `wallet.json`, mainnet
    `FLY_REGULAR_SEED` from `~/lfg-fly/.env`). The wallet address comes from the file on
    testnet and from `FLY_WALLET` on mainnet (the config has no field for it)."""
    if env is None:
        env = os.environ
    if cfg.network == "mainnet":
        seed = cfg.regular_seed
        if not seed:
            raise CredentialsError("FLY_REGULAR_SEED is not set: the mainnet RegularKey seed "
                                   "lives in ~/lfg-fly/.env (chmod 600), spec §4.4")
        account = (env.get("FLY_WALLET") or "").strip()
        if not account:
            raise CredentialsError("FLY_WALLET is not set: the fly's mainnet wallet address "
                                   "(the account whose RegularKey FLY_REGULAR_SEED is)")
    else:
        path = paths.wallet_path(cfg.network)
        if not path.is_file():
            raise CredentialsError(f"no {path} (wallet.json): run `fly setup keygen`, "
                                   "`fly setup faucet` and `fly setup regular-key` first")
        creds = _wallet_file(path, cfg.network)
        account, seed = creds.account, creds.seed
    if not addresscodec.is_valid_classic_address(account):
        raise CredentialsError(f"the fly's wallet {account!r} is not a classic r-address")
    return Credentials(account=account, seed=seed)


def make_signer(cfg: FlyConfig, ledger: Ledger, creds: Credentials | None = None) -> Signer:
    """The RegularKey signer for the fly's wallet (spec §4.3); refuses a master seed."""
    creds = creds if creds is not None else load_credentials(cfg)
    return Signer(creds.seed, creds.account, ledger, cfg)


# ------------------------------------------------------------------ §4.4 identity, once


_IDENTITIES: dict[tuple, ChainIdentity] = {}


async def ensure_identity(ledger: Ledger, cfg: FlyConfig) -> ChainIdentity:
    """Spec §4.4: the chain-identity check runs once per process start. The result is kept
    per (network, endpoints, pinned hash), so a second session in the same process reuses
    it; a mismatch raises `ChainIdentityError` before anything talks to LFG."""
    key = (cfg.network, tuple(ledger.urls), (cfg.expected_ledger_hash or "").upper())
    identity = _IDENTITIES.get(key)
    if identity is None:
        identity = await ledger.identity_check(cfg.expected_ledger_hash)
        _IDENTITIES[key] = identity
    return identity


def reset_identity_cache() -> None:
    """Forget the process's identity checks (tests; a re-pinned testnet hash)."""
    _IDENTITIES.clear()


@asynccontextmanager
async def session(cfg: FlyConfig, *, ledger: Ledger | None = None,
                  signer: Signer | None = None) -> AsyncIterator[tuple[Ledger, LfgClient]]:
    """One fresh agent session (spec §4.1 step 1): identity check, sign in with the
    RegularKey proof, yield `(ledger, client)`, log out in `finally` whatever happened.

    A ledger or signer passed in is used as is (tests, the loop); otherwise the ledger is
    built from `cfg.rpc_urls` and closed on exit, and the signer from `load_credentials`.
    """
    own_ledger = ledger is None
    if ledger is None:
        ledger = Ledger(cfg.rpc_urls, cfg.network)
    client: LfgClient | None = None
    try:
        await ensure_identity(ledger, cfg)
        if signer is None:
            signer = make_signer(cfg, ledger)
        async with LfgClient(cfg.api_base) as client:
            await client.sign_in(signer, ledger)
            log.info("signed in as %s (agent) at %s", client.wallet, cfg.api_base)
            yield ledger, client
    finally:
        if client is not None and client.logout_error is not None:
            log.warning("logout at %s failed: %s (the token is dropped regardless)",
                        cfg.api_base, client.logout_error)
        if own_ledger:
            await ledger.close()


def _alert(cfg: FlyConfig, job: str, code: str, message: str, **extra: Any) -> Path:
    """An operator alert in the outbox (spec §4.2 style). Never carries a token or a seed:
    the payload is the job, the code, the message and LFG's own answer."""
    payload = {"job": job, "code": code, "message": message, "when": _now(), **extra}
    path = outbox.write(cfg.network, "alert", payload)
    log.warning("%s: %s (alert %s)", job, message, path)
    return path


# ------------------------------------------------------------------ fly-claim


@dataclass
class ClaimResult:
    """What `claim` did. `outcome` is one of confirmed | nothing | disabled | failed |
    unknown | refused; only the first three exit 0."""

    outcome: str
    message: str
    claim: dict | None = None
    alert: Path | None = None

    @property
    def exit_code(self) -> int:
        return 0 if self.outcome in CLAIM_EXIT_OK else 1


def _claim_line(body: dict) -> str:
    cid, amount, tx = body.get("claim_id"), body.get("amount"), body.get("tx_hash")
    return f"claim {cid}: {body.get('state')}, {amount} BRIX" + (f", tx {tx}" if tx else "")


async def claim(cfg: FlyConfig, *, ledger: Ledger | None = None, signer: Signer | None = None,
                timeout: float = 600.0, every: float = 3.0) -> ClaimResult:
    """fly-claim (spec §4 table): claim the BRIX drip, then poll its status to the end.

    - `503 claims_disabled` (spec §5.3, staging today): outbox alert, exit 0.
    - `400 nothing_to_claim`: nothing to do, no alert.
    - `409 claim_in_flight`: poll the open claim `/api/brix` reports instead.
    - any other refusal (`trustline_required`, `claim_unavailable`, `claim_unconfirmed`,
      ...): outbox alert, exit 1.
    - `failed`, or not settled within `timeout` (UNKNOWN): outbox alert, exit 1.
    Non-LFG errors (the wire) propagate; the logout still runs.
    """
    job = "fly-claim"
    async with session(cfg, ledger=ledger, signer=signer) as (_ledger, client):
        try:
            body = await client.brix_claim()
        except LfgError as e:
            code = e.code or f"http_{e.status}"
            if code == "claims_disabled":
                msg = ("BRIX claims are disabled on this LFG stack (503 claims_disabled). "
                       "Staging cannot pay BRIX today: it has no BRIX_DISTRIBUTOR_SEED and no "
                       "funded testnet distributor (spec §5.3 prerequisites). Nothing for the "
                       "fly to do until the operator completes them.")
                path = _alert(cfg, job, code, msg, status=e.status, lfg=_safe(e.body))
                return ClaimResult("disabled", msg, alert=path)
            if code == "nothing_to_claim":
                return ClaimResult("nothing", "nothing to claim today: claimable BRIX is 0")
            if code == "claim_in_flight":
                brix = await client.brix()
                open_claim = brix.get("open_claim") or {}
                if not open_claim.get("claim_id"):
                    msg = "LFG reports a claim in flight but /api/brix names no open claim"
                    path = _alert(cfg, job, code, msg, status=e.status, lfg=_safe(brix))
                    return ClaimResult("refused", msg, alert=path)
                body = {"claim_id": open_claim["claim_id"], "state": open_claim.get("state"),
                        "amount": brix.get("claimable"), "tx_hash": open_claim.get("tx_hash")}
                log.info("a claim is already in flight; polling %s", body["claim_id"])
            else:
                msg = f"claim refused by LFG: {e.status} {code}"
                path = _alert(cfg, job, code, msg, status=e.status, lfg=_safe(e.body))
                return ClaimResult("refused", msg, alert=path)
        cid = str(body.get("claim_id"))
        if body.get("state") not in CLAIM_TERMINAL:
            log.info("%s; polling every %.1fs for up to %.0fs", _claim_line(body), every, timeout)
            try:
                body = await client.wait(client.brix_claim_status, cid, set(CLAIM_TERMINAL),
                                         timeout, every)
            except TimeoutError:
                with contextlib.suppress(LfgError):
                    body = await client.brix_claim_status(cid)
                msg = (f"claim {cid} not settled after {timeout:.0f}s (last state "
                       f"{body.get('state')!r}); outcome UNKNOWN, check /api/brix")
                path = _alert(cfg, job, "claim_unknown", msg, claim=_safe(body))
                return ClaimResult("unknown", msg, claim=body, alert=path)
        if body.get("state") == "confirmed":
            msg = f"claimed {body.get('amount')} BRIX: {_claim_line(body)}"
            log.info(msg)
            return ClaimResult("confirmed", msg, claim=body)
        msg = f"claim failed: {_claim_line(body)}"
        path = _alert(cfg, job, "claim_failed", msg, claim=_safe(body))
        return ClaimResult("failed", msg, claim=body, alert=path)


def _safe(obj: Any) -> Any:
    """LFG's answer as JSON-serialisable data for an alert payload."""
    try:
        json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)
    return obj


# ------------------------------------------------------------------ fly-retrain


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, UTF-8 as is: the same payload always hashes the same."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def snapshot_hash(supply: dict) -> str:
    """sha256 of the canonical JSON of `GET /api/rarity/supply`; names the snapshot and the
    rarity head fitted from it (spec §2: 'identified by its snapshot hash')."""
    return hashlib.sha256(canonical_json(supply).encode("utf-8")).hexdigest()


def _hash_component(h: str) -> str:
    if not isinstance(h, str) or not _SHA256.fullmatch(h):
        raise ValueError(f"not a sha256 hex digest: {h!r}")
    return h


def snapshot_path(network: str, h: str) -> Path:
    """`FLY_DATA_DIR/<network>/snapshots/<sha256>.json`."""
    return paths.snapshots_dir(network) / f"{_hash_component(h)}.json"


def head_files(network: str, h: str) -> tuple[Path, Path]:
    """The rarity head's `.npz` and `.json` beside each other at
    `paths.rarity_head_path(network, h)` (a suffix-less base path)."""
    base = paths.rarity_head_path(network, _hash_component(h))
    return base.with_name(base.name + ".npz"), base.with_name(base.name + ".json")


def _meta_path(network: str, h: str) -> Path:
    """The head's stamp and fit stats: `rarity-meta-<hash>.json`, deliberately outside the
    `rarity-head-*.json` pattern the loop globs for the newest head."""
    return paths.snapshots_dir(network) / f"rarity-meta-{_hash_component(h)}.json"


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Serialize first (an unserialisable object leaves nothing behind), then tmp + replace."""
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise


def _check_stamp(found: Any, expect: Stamp, what: Path | str) -> None:
    """Spec §1: every snapshot and rarity-head artifact is stamped {network, lfg_api_base,
    wallet}, and loaders refuse a mismatch."""
    if not isinstance(found, dict):
        raise StampMismatch(f"{what}: no stamp")
    for f in fields(Stamp):
        if found.get(f.name) != getattr(expect, f.name):
            raise StampMismatch(f"{what}: stamp {f.name}={found.get(f.name)!r}, "
                                f"this fly is {getattr(expect, f.name)!r}")


def write_snapshot(network: str, supply: dict, stamp: Stamp) -> tuple[Path, str]:
    """Save the supply snapshot, stamped, under its hash; returns (path, hash)."""
    h = snapshot_hash(supply)
    path = snapshot_path(network, h)
    _atomic_write_json(path, {"stamp": asdict(stamp), "snapshot_hash": h, "saved_at": _now(),
                              "supply": supply})
    return path, h


def read_snapshot(path: str | Path, expect: Stamp) -> dict:
    """The supply held by a snapshot file; refuses a foreign stamp (StampMismatch) and a
    payload that does not hash to its name (ValueError)."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a snapshot")
    _check_stamp(doc.get("stamp"), expect, path)
    supply = doc.get("supply")
    if not isinstance(supply, dict):
        raise ValueError(f"{path}: no supply")
    h = snapshot_hash(supply)
    if h != doc.get("snapshot_hash") or h != path.stem:
        raise ValueError(f"{path}: the supply hashes to {h[:16]}, not the recorded name")
    return supply


def latest_head(network: str, expect: Stamp) -> dict | None:
    """The pointer `retrain` leaves for the loop: {snapshot_hash, snapshot, head, stamp,
    stats, ...}; None when no retrain has run; StampMismatch on a foreign stamp."""
    path = paths.snapshots_dir(network) / LATEST_HEAD
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a pointer")
    _check_stamp(doc.get("stamp"), expect, path)
    return doc


def sample_looks(catalog, n: int, seed: int) -> list[Look]:
    """`n` distinct looks from `teacher.sample.base_look` (the Phase 0 mix: 40% realistic,
    30% uniform, 30% hybrid), reproducible from `seed`. A catalog too small for `n`
    distinct looks yields what it has."""
    rng = np.random.default_rng(seed)
    seen: dict[Look, None] = {}
    attempts = 0
    while len(seen) < n and attempts < 20 * max(n, 1):
        seen.setdefault(tuple(base_look(catalog, rng)), None)
        attempts += 1
    if len(seen) < n:
        log.warning("only %d distinct looks after %d draws (asked for %d)", len(seen), attempts, n)
    return list(seen)


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    ss_tot = float(((target - target.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_res = float(((np.asarray(pred, dtype=np.float64) - target) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _predict(head: R.RarityHead, X: np.ndarray) -> np.ndarray:
    """The head's standardized score back in `nft_rarity` units."""
    return head.score(X) * head.target_sd + head.target_mu


def fit_head(brain, looks: list[Look], supply: dict, h: str,
             lam: float = DEFAULT_LAM) -> tuple[R.RarityHead, dict]:
    """The live-concentration re-simulation and the ridge (spec §2 Readout, rarity head).

    `brain.features(looks, conc=concentrations(looks, supply, brain.catalog))` gives the
    features; the target is `nft_rarity(look, supply)` per look. The last `HELDOUT_FRAC`
    of the (randomly drawn) looks report a held-out R² from a fit on the rest; the saved
    head is fitted on every look.
    """
    if not looks:
        raise DailyError("no looks to fit the rarity head on")
    conc = R.concentrations(looks, supply, brain.catalog)
    X, sim_stats = brain.features(looks, conc=conc)
    X = np.asarray(X, dtype=np.float64)
    target = np.array([R.nft_rarity(lk, supply) for lk in looks], dtype=np.float64)
    n = len(looks)
    n_held = int(round(HELDOUT_FRAC * n)) if n >= 10 else 0
    heldout_r2 = float("nan")
    if n_held:
        split = n - n_held
        trial = R.fit_rarity(X[:split], target[:split], h, lam)
        heldout_r2 = _r2(_predict(trial, X[split:]), target[split:])
    head = R.fit_rarity(X, target, h, lam)
    stats = {
        "n_looks": n, "n_heldout": n_held, "lam": float(lam),
        "train_r2": _r2(_predict(head, X), target), "heldout_r2": heldout_r2,
        "target_mu": head.target_mu, "target_sd": head.target_sd,
        "target_min": float(target.min()), "target_max": float(target.max()),
    }
    for k in ("neurons_fired", "ms", "active_frac", "readout_active_frac", "max_step_frac"):
        if k in sim_stats:
            stats[k] = sim_stats[k]
    return head, stats


@dataclass
class RetrainResult:
    """What `retrain` produced (or found already there, `skipped`)."""

    snapshot_hash: str
    snapshot_path: Path
    head_path: Path  # the .npz; its .json and rarity-meta-<hash>.json sit beside it
    n_looks: int
    stats: dict
    skipped: bool = False
    looks: list = field(default_factory=list)


def _seed_from_hash(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h[:16]), "little")


async def retrain(cfg: FlyConfig, brain, *, n: int = DEFAULT_N_LOOKS, lam: float = DEFAULT_LAM,
                  seed: int | None = None, force: bool = False, ledger: Ledger | None = None,
                  signer: Signer | None = None) -> RetrainResult:
    """fly-retrain (spec §4 table): supply snapshot → live-concentration re-simulation →
    new rarity head, named by the snapshot hash and stamped {network, lfg_api_base, wallet}.

    `brain` is a `FlyBrain` (`.catalog`, `.features(looks, conc)`); loading it is the
    caller's job, before the session opens. The session is closed once the supply is read.
    A supply stamped with another network is refused before anything is written. The
    sample's seed defaults to the snapshot hash, so a retrain is reproducible; a head that
    already exists for this snapshot is kept unless `force`.
    """
    async with session(cfg, ledger=ledger, signer=signer) as (_ledger, client):
        supply = await client.rarity_supply()
        stamp = Stamp(network=cfg.network, lfg_api_base=cfg.api_base, wallet=str(client.wallet))
    # the token is gone; everything below is local
    if not isinstance(supply, dict) or not isinstance(supply.get("counts"), dict):
        raise DailyError(f"/api/rarity/supply answered no supply: {_safe(supply)!r}"[:300])
    net = supply.get("network")
    if net is not None and net != cfg.network:
        raise StampMismatch(f"/api/rarity/supply at {cfg.api_base} is stamped {net!r}, "
                            f"this fly runs {cfg.network}")
    h = snapshot_hash(supply)
    snap_path, _ = write_snapshot(cfg.network, supply, stamp)
    npz, js = head_files(cfg.network, h)
    if seed is None:
        seed = _seed_from_hash(h)
    if npz.exists() and js.exists() and not force:
        head = R.load_head(npz)
        if isinstance(head, R.RarityHead) and head.snapshot_hash == h:
            meta: dict = {}
            with contextlib.suppress(OSError, ValueError):
                meta = json.loads(_meta_path(cfg.network, h).read_text(encoding="utf-8"))
            log.info("rarity head for snapshot %s already exists at %s; kept", h[:16], npz)
            return RetrainResult(h, snap_path, npz, int(meta.get("n_looks", 0)),
                                 dict(meta.get("stats") or {}), skipped=True)
    looks = sample_looks(brain.catalog, n, seed)
    log.info("snapshot %s (%s live, as_of %s): simulating %d looks under live concentrations",
             h[:16], supply.get("n_live"), supply.get("as_of"), len(looks))
    head, stats = fit_head(brain, looks, supply, h, lam)
    R.save_head(npz, head)
    meta = {"stamp": asdict(stamp), "snapshot_hash": h, "snapshot": str(snap_path),
            "head": str(npz), "version": cfg.version, "seed": int(seed), "n_looks": len(looks),
            "lam": float(lam), "stats": stats, "created_at": _now()}
    _atomic_write_json(_meta_path(cfg.network, h), meta)
    _atomic_write_json(paths.snapshots_dir(cfg.network) / LATEST_HEAD, meta)
    log.info("rarity head %s: train R² %.3f, held-out R² %.3f over %d looks -> %s", h[:16],
             stats["train_r2"], stats["heldout_r2"], len(looks), npz)
    return RetrainResult(h, snap_path, npz, len(looks), stats, looks=looks)
