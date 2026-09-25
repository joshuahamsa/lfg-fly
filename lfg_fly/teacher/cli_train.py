"""`fly train`: the taste head on the critic's round, its controls, the checkpoint and
REPORT.md (spec §3.3, §3.4). `register(sub)` is wired by the integrator in `cli.py`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _load_context(device: str, min_syn: int):
    """The Phase 0 pipeline's context (graph, populations, retina, mainnet catalog, layer bank
    and pinned z-order), with no probe set: the labels are the critic's."""
    from lfg_fly import paths
    from lfg_fly.brain.senses import build_retina
    from lfg_fly.connectome import build as B
    from lfg_fly.connectome.columns import load_columns
    from lfg_fly.connectome.neurons import populations
    from lfg_fly.teacher.catalog import Catalog
    from lfg_fly.teacher.grid import Context
    from lfg_fly.teacher.render import LayerBank, load_zorder

    gdir = paths.graph_dir()
    g = B.load_graph(gdir / f"graph-syn{min_syn}.npz")
    pops = populations(g)
    retina = build_retina(load_columns(gdir / f"columns-syn{min_syn}.npz"), g)
    cache = paths.catalog_dir("mainnet")
    cat = Catalog.from_json((cache / "catalog-male.json").read_text(encoding="utf-8"))
    bank = LayerBank(cat, cache, size=64, device=device)
    return Context(graph=g, pops=pops, retina=retina, catalog=cat, bank=bank,
                   zorder=load_zorder(cache), device=device, probe_set=None, seed=0)


def _control_line(name: str, row: dict, paired: dict) -> str:
    if "skipped" in row:
        return f"  {name:14s} not run: {row['skipped']}"
    d = paired.get(name)
    diff = "" if d is None else f"  fly - model {d['diff']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})"
    return f"  {name:14s} {row['heldout']:.3f} ({row['ci_lo']:.3f}-{row['ci_hi']:.3f}){diff}"


def _cmd_train(args: argparse.Namespace, *, root: Path | None = None,
               load_context=_load_context) -> int:
    from lfg_fly import env, paths
    from lfg_fly.teacher import train as TR

    env.configure_libraries()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    root = Path(root) if root is not None else paths.repo_root()
    p = TR.round_paths(root, args.round, args.version)
    if not p["labels"].exists():
        print(f"no labels for round {args.round}: {p['labels']} does not exist "
              "(run `fly critic assemble` first)")
        return 2
    setting = TR.phase0_setting(p["phase0_verdict"])
    print(f"round {args.round} -> {args.version}: brain {setting.key()}")
    ctx = load_context(args.device, args.min_syn)
    results = TR.train_round(ctx, setting, round_name=args.round, version=args.version,
                             root=root, controls=not args.no_controls)
    lab, fly, g = results["labels"], results["fly"], results["gate"]
    print(f"labels: {lab['n_pairs']} pairs kept ({lab['n_dropped']} flipped dropped), "
          f"{lab['n_train']} train / {lab['n_test']} test over {lab['n_test_families']} "
          "held-out families")
    print(f"fly: held-out {fly['heldout']:.3f} ({fly['ci_lo']:.3f}-{fly['ci_hi']:.3f}), "
          f"lambda {fly['lam']}, {fly['seconds']} s")
    for name in TR.CONTROL_NAMES:
        if name in results["controls"]:
            print(_control_line(name, results["controls"][name], results["paired"]))
    print(f"Gate B: {'PASS' if g['pass'] else 'FAIL'} (CI lower bound {g['ci_lo']:.3f}, "
          f"needs > {g['ci_lo_min']:.2f})")
    print(f"wrote {p['checkpoint']}, {p['out_dir'] / TR.RESULTS} and {p['report']}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="train the taste head on the critic's round, score the "
                                     "controls, write checkpoints/<version> and REPORT.md "
                                     "(spec §3.3, §3.4)")
    p.add_argument("--round", default="r1", help="the critic's round: data/critic/<round>.jsonl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--version", default="fly-v1", help="the checkpoint version to write")
    p.add_argument("--no-controls", action="store_true",
                   help="skip the spec §3.4 controls (the report says so on every row)")
    p.set_defaults(func=_cmd_train)
