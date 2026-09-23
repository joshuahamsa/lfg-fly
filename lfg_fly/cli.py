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


def _cmd_build(args: argparse.Namespace) -> int:
    from lfg_fly import env, paths
    from lfg_fly.connectome import build as B
    from lfg_fly.connectome.fetch import FILES

    env.configure_libraries()
    raw = paths.raw_dir()
    ann = B.load_annotations(raw / FILES["annotations"])
    nt = B.load_nt(raw / FILES["neurotransmitters"])
    edges = B.load_edges(raw / FILES["weights"], args.min_syn)
    g = B.build_graph(ann, nt, edges, min_syn=args.min_syn, device=args.device)
    out = paths.graph_dir() / f"graph-syn{args.min_syn}.npz"
    B.save_graph(g, out)
    print(f"{g.n:,} neurons, {len(g.col):,} edges (>= {g.min_syn}), NT missing {g.nt_missing}, "
          f"hash {g.graph_hash()[:16]} -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fly", description="The fly: an LFG-dressing connectome")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("fetch", help="download (or hardlink) the MaleCNS feathers")
    p.add_argument("--from", dest="local_from", type=Path, default=None)
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("build", help="build the thresholded integer-CSR graph")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 0
    return int(args.func(args) or 0)
