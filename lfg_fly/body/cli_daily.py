"""`fly claim` and `fly retrain`: the pm2 jobs of spec §4's table.

`register(sub)` adds both subcommands; the integrator wires it into `lfg_fly.cli`. Each
command loads `~/lfg-fly/.env` into the environment (the contract: the CLI does, never
`load_config`), resolves the config, and runs the coroutine in `lfg_fly.body.daily`.
Heavy imports stay inside the commands so `fly --help` stays light.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
from pathlib import Path

DOTENV = Path.home() / "lfg-fly" / ".env"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _prepare():
    """Logging for pm2's log files, the .env file, the config. Returns (daily, cfg)."""
    from lfg_fly.body import daily as D
    from lfg_fly.body.config import load_config

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    D.load_env_file(DOTENV)
    return D, load_config()


def _cmd_claim(args: argparse.Namespace) -> int:
    """fly-claim (spec §4): claim the BRIX drip and poll its status; exit 0 on confirmed,
    nothing to claim, or claims disabled on this stack (spec §5.3, alert written); exit 2
    when the kill switch is off (`FLY_ENABLED` != 1, spec §4.4)."""
    D, cfg = _prepare()
    try:
        result = asyncio.run(D.claim(cfg, timeout=args.timeout, every=args.every))
    except D.DailyError as e:
        print(f"claim refused: {e}")
        return 2
    print(result.message)
    if result.alert is not None:
        print(f"alert: {result.alert}")
    return result.exit_code


def _cmd_retrain(args: argparse.Namespace) -> int:
    """fly-retrain (spec §4): supply snapshot -> live-concentration re-simulation -> a new
    rarity head named by the snapshot hash. The kill switch is checked before the brain
    loads (exit 2); the brain loads before the session opens."""
    D, cfg = _prepare()
    try:
        D.require_enabled(cfg, "fly-retrain")
    except D.DailyError as e:
        print(f"retrain refused: {e}")
        return 2
    from lfg_fly import env, paths
    from lfg_fly.brain.checkpoint import FlyBrain

    env.configure_libraries()
    brain = FlyBrain.load(paths.checkpoint_dir(cfg.version), args.device,
                          paths.catalog_dir(cfg.network))
    try:
        result = asyncio.run(D.retrain(cfg, brain, n=args.n, lam=args.lam, seed=args.seed,
                                       force=args.force))
    except D.DailyError as e:
        print(f"retrain refused: {e}")
        return 2
    if result.skipped:
        print(f"snapshot {result.snapshot_hash[:16]} unchanged; rarity head kept at "
              f"{result.head_path}")
        return 0
    held = result.stats.get("heldout_r2", float("nan"))
    held_s = f"{held:.3f}" if not math.isnan(held) else "n/a"
    print(f"snapshot {result.snapshot_hash[:16]} -> {result.snapshot_path}")
    print(f"rarity head over {result.n_looks} looks: train R² {result.stats['train_r2']:.3f}, "
          f"held-out R² {held_s}, {result.stats.get('ms', 0.0):.0f} ms of simulation -> "
          f"{result.head_path}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Add `claim` and `retrain` to the `fly` parser (contract: `register(sub)`)."""
    from lfg_fly.body.daily import DEFAULT_LAM, DEFAULT_N_LOOKS

    p = sub.add_parser("claim", help="claim the BRIX drip and poll the payout (fly-claim, "
                                     "04:10 UTC)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="seconds to wait for a submitted claim to settle (default 600)")
    p.add_argument("--every", type=float, default=3.0, help="poll interval in seconds")
    p.set_defaults(func=_cmd_claim)

    p = sub.add_parser("retrain", help="supply snapshot -> live-concentration re-simulation "
                                       "-> new rarity head (fly-retrain, 04:30 UTC)")
    p.add_argument("--n", type=int, default=DEFAULT_N_LOOKS,
                   help=f"looks to sample from the catalog (default {DEFAULT_N_LOOKS})")
    p.add_argument("--device", default="cuda")
    p.add_argument("--lam", type=float, default=DEFAULT_LAM, help="ridge strength")
    p.add_argument("--seed", type=int, default=None,
                   help="sample seed (default: derived from the snapshot hash)")
    p.add_argument("--force", action="store_true",
                   help="refit even if a head for this snapshot already exists")
    p.set_defaults(func=_cmd_retrain)
