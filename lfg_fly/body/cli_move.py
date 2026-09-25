"""`fly move [--dry-run] [--date YYYY-MM-DD] [--hero NFT_ID] [--device cuda|cpu]` (spec §4.1).

`register(sub)` adds the command; the integrator wires it into `lfg_fly/cli.py`. The
command loads `~/lfg-fly/.env` into the environment when present (the contract: the CLI
loads it, `load_config` never reads a file), resolves the config, loads the checkpoint's
brain and runs `loop.move`. A refusal or a stop (`LoopRefused`, `LoopStopped`) prints its
reason and exits 2; the record is summarised on success. Nothing here prints a token or a
seed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import date
from pathlib import Path

from lfg_fly.body import loop
from lfg_fly.body.config import load_config

DOTENV = Path.home() / "lfg-fly" / ".env"


def load_dotenv(path: Path = DOTENV) -> int:
    """Put `KEY=value` lines of `path` into the environment without overriding what is
    already set; returns how many were added. Missing file: 0."""
    path = Path(path)
    if not path.is_file():
        return 0
    added = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            added += 1
    return added


def _load_brain(cfg, device: str):
    from lfg_fly import env, paths
    from lfg_fly.brain.checkpoint import FlyBrain

    env.configure_libraries()
    return FlyBrain.load(paths.checkpoint_dir(cfg.version), device, paths.catalog_dir(cfg.network))


def _summary(record) -> str:
    lines = [
        f"{record.date} {record.state}  hero {record.hero}  network {record.stamp.network}",
        f"  before: {' | '.join(record.before)}",
        f"  after:  {' | '.join(record.after)}",
    ]
    if record.changes:
        lines.append("  changes: " + ", ".join(f"{c['slot']} -> {c['value']}"
                                               for c in record.changes))
    else:
        lines.append("  changes: none (stay)")
    stats = record.neuron_stats or {}
    lines.append(f"  considered {record.considered} of {record.total}; "
                 f"{stats.get('neurons_fired', 0)} neurons fired in {stats.get('ms', 0):.0f} ms")
    if record.equip_id:
        lines.append(f"  equip {record.equip_id}: {record.resolution or record.state}"
                     + (f" ({record.error})" if record.error else ""))
    if record.post:
        lines.append(f"  post: {record.post.get('channel')} {record.post.get('path', '')}".rstrip())
    return "\n".join(lines)


def _cmd_move(args: argparse.Namespace) -> int:
    load_dotenv(DOTENV)
    cfg = load_config()
    today = date.fromisoformat(args.date) if args.date else None
    brain = _load_brain(cfg, args.device)
    from lfg_fly.body.chain import ChainError
    from lfg_fly.body.client import LfgError

    try:
        record = asyncio.run(loop.move(cfg, brain, dry_run=args.dry_run, today=today,
                                       hero=args.hero))
    except (loop.LoopRefused, loop.LoopStopped) as e:
        print(f"move refused: {e}")
        return 2
    except LfgError as e:
        hint = (" (LFG's AGENT_SIGNIN_ENABLED is off on this stack)"
                if e.code == "agent_disabled" else "")
        print(f"move failed: LFG answered {e.status} {e.code or e.body}{hint}")
        return 2
    except ChainError as e:
        print(f"move failed: {type(e).__name__}: {e}")
        return 2
    print(_summary(record))
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("move", help="the fly's daily move (spec §4.1): sign in, reconcile, "
                                    "decide, record, equip, post")
    p.add_argument("--dry-run", action="store_true",
                   help="decide and write the record, but never equip (needs no FLY_ENABLED)")
    p.add_argument("--date", default=None, help="the UTC day to move for (default: today)")
    p.add_argument("--hero", default=None, help="the hero's nft_id (default: FLY_HERO, "
                                                "else the last record's, else the first "
                                                "mutable male)")
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_move)
