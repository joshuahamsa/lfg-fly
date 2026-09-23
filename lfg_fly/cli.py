"""`fly` command line. Subcommands are added by later tasks."""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_fetch(args: argparse.Namespace) -> int:
    from lfg_fly import paths
    from lfg_fly.connectome.fetch import fetch

    manifest = fetch(paths.raw_dir(), local_from=args.local_from)
    for key, row in manifest.items():
        print(f"{key}: {row['bytes']:,} bytes sha256 {row['sha256'][:16]}…")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fly", description="The fly: an LFG-dressing connectome")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("fetch", help="download (or hardlink) the MaleCNS feathers")
    p.add_argument("--from", dest="local_from", type=Path, default=None)
    p.set_defaults(func=_cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 0
    return int(args.func(args) or 0)
