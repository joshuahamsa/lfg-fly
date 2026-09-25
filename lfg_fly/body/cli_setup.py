"""`fly setup <step>`: the one-off steps before the daily loop (spec §5.3, §5.4, §4.3).

`register(sub)` adds the command; `lfg_fly.cli` wires it. Every step loads `~/lfg-fly/.env`
into the environment (the contract: the CLI does, `load_config` never reads a file), resolves
the config, and runs one coroutine of `lfg_fly.body.setup`. The steps that talk to LFG run
inside `_run_session`: chain identity first, then a fresh agent sign-in, the step, and a
logout whatever happens. Nothing here prints a seed, except `keygen` on mainnet, which shows
the new RegularKey's seed once so the operator can put it in `~/lfg-fly/.env` (spec §4.4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

from lfg_fly.body import setup as SU
from lfg_fly.body.config import FlyConfig, load_config

DOTENV = Path.home() / "lfg-fly" / ".env"
SECRET_KEYS = ("seed", "secret", "token")


def load_env_file(path: str | Path, env: MutableMapping[str, str] | None = None) -> list[str]:
    """`KEY=VALUE` lines of `path` into `env` (default `os.environ`), never overriding a key
    the environment already has; returns the keys it set, in file order. Missing file: []."""
    from lfg_fly.body.daily import load_env_file as _load

    return list(_load(path, env))


def _redact(obj: Any) -> Any:
    """Strip any key that could carry a secret before printing a step's result."""
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()
                if not any(s in str(k).lower() for s in SECRET_KEYS)}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _print(result: Any) -> None:
    print(json.dumps(_redact(result), indent=1, sort_keys=True, default=str))


async def _run_ledger(cfg: FlyConfig, fn: Callable[[Any], Awaitable[Any]]) -> Any:
    """A step that needs the ledger but no LFG session (faucet, regular-key)."""
    from lfg_fly.body.chain import Ledger

    async with Ledger(list(cfg.rpc_urls), cfg.network) as ledger:
        await ledger.identity_check(cfg.expected_ledger_hash)
        return await fn(ledger)


async def _run_session(cfg: FlyConfig, step: Callable[[SU.Setup], Awaitable[Any]]) -> Any:
    """Chain identity (spec §4.4) → fresh agent sign-in (§4.1 step 1) → `step(setup)` →
    logout in `finally` (the client's `__aexit__`), token in memory only."""
    from lfg_fly.body.chain import Ledger
    from lfg_fly.body.client import LfgClient

    async with Ledger(list(cfg.rpc_urls), cfg.network) as ledger:
        await ledger.identity_check(cfg.expected_ledger_hash)
        async with LfgClient(cfg.api_base) as client:
            setup = SU.Setup(cfg, ledger, client, lambda c: SU.build_signer(c, ledger))
            await client.sign_in(setup.signer, ledger)
            return await step(setup)


def _config() -> FlyConfig:
    load_env_file(DOTENV)
    return load_config()


def _guarded(fn: Callable[[argparse.Namespace, FlyConfig], int],
             ) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        from lfg_fly.body.chain import ChainError
        from lfg_fly.body.client import LfgError
        from lfg_fly.body.signer import PolicyError

        try:
            return fn(args, _config())
        except SU.InsideRepoError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        except SU.SetupError as e:
            print(f"setup failed: {e}", file=sys.stderr)
            return 1
        except LfgError as e:
            hint = ""
            if e.code == "agent_disabled":
                hint = (" (LFG's AGENT_SIGNIN_ENABLED is off on this stack: the companion "
                        "spec's rollout step 3)")
            print(f"setup failed: LFG answered {e.status} {e.code or e.body}{hint}",
                  file=sys.stderr)
            return 1
        except (ChainError, PolicyError) as e:
            print(f"setup failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
    return run


@_guarded
def cmd_keygen(args: argparse.Namespace, cfg: FlyConfig) -> int:
    out = SU.keygen(cfg, force=args.force)
    if cfg.network == "mainnet":
        print(f"RegularKey address: {out['regular_address']}")
        print("Set it on the fly's wallet from Xaman (spec §5.4 step 2), and put this line in "
              "~/lfg-fly/.env (chmod 600); it is shown once and written nowhere:")
        print(f"FLY_REGULAR_SEED={out['regular_seed']}")
        return 0
    print(f"RegularKey address: {out['regular_address']}")
    print(f"written to {out['path']} (the seed stays there; next: `fly setup faucet`, then "
          "`fly setup regular-key`)")
    return 0


@_guarded
def cmd_faucet(args: argparse.Namespace, cfg: FlyConfig) -> int:
    if cfg.network != "testnet":
        raise SU.SetupError("the faucet is a testnet step (spec §5.3)")
    out = asyncio.run(_run_ledger(cfg, lambda ledger: SU.faucet(cfg, ledger=ledger)))
    _print(out)
    return 0


@_guarded
def cmd_regular_key(args: argparse.Namespace, cfg: FlyConfig) -> int:
    out = asyncio.run(_run_ledger(cfg, lambda ledger: SU.set_regular_key(cfg, ledger)))
    _print(out)
    return 0


@_guarded
def cmd_trustline(args: argparse.Namespace, cfg: FlyConfig) -> int:
    _print(asyncio.run(_run_session(cfg, lambda s: s.trustline())))
    return 0


@_guarded
def cmd_closet(args: argparse.Namespace, cfg: FlyConfig) -> int:
    _print(asyncio.run(_run_session(cfg, lambda s: s.closet())))
    return 0


@_guarded
def cmd_mint(args: argparse.Namespace, cfg: FlyConfig) -> int:
    _print(asyncio.run(_run_session(
        cfg, lambda s: s.mint(args.count, bulk=args.bulk, body=args.body))))
    return 0


@_guarded
def cmd_harvest(args: argparse.Namespace, cfg: FlyConfig) -> int:
    _print(asyncio.run(_run_session(
        cfg, lambda s: s.harvest(args.nft_ids or None, all_=args.all))))
    return 0


@_guarded
def cmd_accept(args: argparse.Namespace, cfg: FlyConfig) -> int:
    """An operator's donors, accepted on-ledger (spec §5 step 3(b)); no LFG session. Exit 1
    when any requested NFT was not accepted, so a skipped one is never silent."""
    if not cfg.donor_sources:
        raise SU.SetupError("FLY_DONOR_SOURCES is empty: `fly setup accept` takes offers only "
                            "from the wallets it lists (spec §4.3, §5 step 3(b))")
    out = asyncio.run(_run_ledger(
        cfg, lambda ledger: SU.accept(cfg, ledger, SU.build_signer(cfg, ledger), args.nft_ids)))
    _print(out)
    return 0 if not out["skipped"] else 1


@_guarded
def cmd_status(args: argparse.Namespace, cfg: FlyConfig) -> int:
    _print(asyncio.run(_run_session(cfg, lambda s: s.status())))
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "keygen": cmd_keygen,
    "faucet": cmd_faucet,
    "regular-key": cmd_regular_key,
    "trustline": cmd_trustline,
    "closet": cmd_closet,
    "mint": cmd_mint,
    "accept": cmd_accept,
    "harvest": cmd_harvest,
    "status": cmd_status,
}


def _dispatch(args: argparse.Namespace) -> int:
    return COMMANDS[args.setup_command](args)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("setup", help="one-off setup steps (spec §5.3, §5.4): keygen, faucet, "
                                     "regular-key, trustline, closet, mint, accept, harvest, "
                                     "status")
    steps = p.add_subparsers(dest="setup_command", required=True)
    s = steps.add_parser("keygen", help="a new RegularKey (testnet: into wallet.json; "
                                        "mainnet: shown once for Xaman and .env)")
    s.add_argument("--force", action="store_true", help="replace an existing testnet key")
    steps.add_parser("faucet", help="testnet: create or top up the wallet from the faucet")
    steps.add_parser("regular-key", help="testnet: SetRegularKey, signed once by the master")
    steps.add_parser("trustline", help="set the BRIX trust line (client-signed)")
    steps.add_parser("closet", help="create and accept the Closet")
    s = steps.add_parser("mint", help="mint donors through LFG's mint flow (spend-capped)")
    s.add_argument("--count", type=int, required=True)
    s.add_argument("--bulk", action="store_true", help="one bulk-mint job instead of singles")
    s.add_argument("--body", default=None, help="keep minting until this body arrives")
    s = steps.add_parser("accept", help="accept an operator's zero-price sell offers on-ledger "
                                        "(owner in FLY_DONOR_SOURCES; spec §5 step 3(b))")
    s.add_argument("nft_ids", nargs="+", help="the NFTokenIDs the operator offered")
    s = steps.add_parser("harvest", help="harvest donors into the Closet")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="every mutable non-hero character")
    g.add_argument("nft_ids", nargs="*", default=[], help="specific characters")
    steps.add_parser("status", help="wallet, key, characters, Closet, BRIX, spend")
    p.set_defaults(func=_dispatch)
