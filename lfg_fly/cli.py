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


def _cmd_columns(args: argparse.Namespace) -> int:
    import numpy as np

    from lfg_fly import env, paths
    from lfg_fly.connectome import build as B
    from lfg_fly.connectome import columns as C
    from lfg_fly.connectome.fetch import FILES
    from lfg_fly.connectome.neurons import populations

    env.configure_libraries()
    g = B.load_graph(paths.graph_dir() / f"graph-syn{args.min_syn}.npz")
    pops = populations(g)
    ann = B.load_annotations(paths.raw_dir() / FILES["annotations"])
    edges = C.load_pr_edges(paths.raw_dir() / FILES["weights"], g.body_id[pops.photoreceptor])
    cols = C.assign_columns(g, pops, ann, edges)
    C.save_columns(cols, paths.graph_dir() / f"columns-syn{args.min_syn}.npz")
    print(f"assigned {cols.total - cols.dropped}/{cols.total} photoreceptors; "
          f"median top-column share {float(np.median(cols.share)):.2f}")
    return 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    from lfg_fly import paths
    from lfg_fly.teacher.catalog import build_catalog
    from lfg_fly.teacher.render import load_zorder

    cache = paths.network_dir("mainnet") / "catalog"
    cat = build_catalog(args.api.rstrip("/"), cache, body=args.body)
    (cache / f"catalog-{args.body}.json").write_text(cat.to_json())
    load_zorder(cache)
    print({slot: len(v) for slot, v in cat.values.items()})
    return 0


def _load_context(args: argparse.Namespace):
    from lfg_fly import paths
    from lfg_fly.brain.senses import build_retina
    from lfg_fly.connectome import build as B
    from lfg_fly.connectome.columns import load_columns
    from lfg_fly.connectome.neurons import populations
    from lfg_fly.teacher.catalog import Catalog
    from lfg_fly.teacher.grid import CONFIRM_SEED_OFFSET, Context
    from lfg_fly.teacher.render import LayerBank, load_zorder
    from lfg_fly.teacher.sample import make_probe_set

    g = B.load_graph(paths.graph_dir() / f"graph-syn{args.min_syn}.npz")
    pops = populations(g)
    retina = build_retina(load_columns(paths.graph_dir() / f"columns-syn{args.min_syn}.npz"), g)
    cache = paths.network_dir("mainnet") / "catalog"
    cat = Catalog.from_json((cache / "catalog-male.json").read_text())
    bank = LayerBank(cat, cache, size=64, device=args.device)
    seed = 0
    # the confirmation set: a fresh instance of the task (pairs, families and, since
    # make_probe_set seeds the planted taste from seed + 1, the taste itself). It is
    # not in the context fingerprint, so adding it left every Stage A/B record current.
    return Context(graph=g, pops=pops, retina=retina, catalog=cat, bank=bank,
                   zorder=load_zorder(cache), device=args.device,
                   probe_set=make_probe_set(cat, n_pairs=args.pairs, seed=seed), seed=seed,
                   confirm_set=make_probe_set(cat, n_pairs=args.pairs,
                                              seed=seed + CONFIRM_SEED_OFFSET))


def _confirmation_line(c: dict, best: dict) -> str:
    return (f"confirmation held-out {c['confirmation']['heldout']:.3f} "
            f"(selection {best['probe']['heldout']:.3f}, optimistic); on the confirmation set: "
            f"rewired {c['rewired']['heldout']:.3f}, "
            f"sign-shuffled {c['sign_shuffled']['heldout']:.3f}, "
            f"one-hot {c['one_hot_confirmation']['heldout']:.3f}, "
            f"Bayes {c['bayes_confirmation']:.3f}")


def _cmd_grid(args: argparse.Namespace) -> int:
    import json
    import logging

    from lfg_fly import env, paths
    from lfg_fly.teacher import grid as G
    from lfg_fly.teacher import probe as P
    from lfg_fly.teacher.sample import bayes_ceiling

    env.configure_libraries()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ctx = _load_context(args)
    context = G.context_fingerprint(ctx)
    print(f"context {context}")
    out = paths.repo_root() / "data" / "probe"
    # records from another context (graph, --min-syn, --pairs, seed, catalog) are never reused
    if args.stage in ("a", "all"):
        a = G.stage_a(ctx, G.default_grid(), out / "stage_a.jsonl")
        print(f"stage A: {sum(r['passed'] for r in a)}/{len(a)} passed")
    else:
        a = G.current_records(out / "stage_a.jsonl", context)
    if args.stage in ("b", "all"):
        b = G.stage_b(ctx, a, out / "stage_b.jsonl", cap=args.cap)
        print(f"stage B: probed {len(b)}")
    else:
        b = G.current_records(out / "stage_b.jsonl", context)
    c_path = out / "stage_c.json"
    c = json.loads(c_path.read_text()) if c_path.exists() else {}
    best = G.best_eligible(b)
    # Stage C confirms the best on the independent confirmation set. A stage_c.json for
    # another setting, context or confirmation seed (or one from before the confirmation
    # set, with no confirm_seed) is never reused.
    if best is not None and not G.stage_c_matches(c, best, ctx.seed):
        if args.stage in ("c", "all"):
            c = G.stage_c(ctx, best, c_path)
            print(f"stage C: {_confirmation_line(c, best)}")
        else:
            print("stage C: stage_c.json is not the current best setting's (or context's, or "
                  "confirmation set's); the verdict is PENDING until `--stage c` confirms it")
    elif best is not None and args.stage in ("c", "all"):
        print("stage C: stage_c.json already confirms the best setting in this context; reused: "
              + _confirmation_line(c, best))
    ps = ctx.probe_set  # selection-set controls: the verdict's only while Stage C is missing
    one_hot = P.evaluate(P.one_hot(ps.looks, ctx.catalog), ps, device=args.device).as_dict()
    v = G.verdict(a, b, c, one_hot, bayes_ceiling(ps.p[ps.test]), seed=ctx.seed)
    v["context"] = context
    v["graph_hash"] = ctx.graph.graph_hash()
    v["catalog_values"] = {slot: len(vals) for slot, vals in ctx.catalog.values.items()}
    out.mkdir(parents=True, exist_ok=True)  # `--stage b|c` on a fresh checkout wrote nothing yet
    (out / "verdict.json").write_text(json.dumps(v, indent=1, sort_keys=True) + "\n")
    print(G.verdict_line(v, args.cap))
    return 0


def _cmd_interact(args: argparse.Namespace) -> int:
    import json
    import logging

    from lfg_fly import env, paths
    from lfg_fly.teacher import grid as G
    from lfg_fly.teacher import interact as I
    from lfg_fly.teacher import interact_report as IR

    env.configure_libraries()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ctx = _load_context(args)
    context, phase0 = I.interact_fingerprint(ctx), G.context_fingerprint(ctx)
    print(f"context {context} (Phase 0 context {phase0})")
    p0 = paths.repo_root() / "data" / "probe"
    out = paths.repo_root() / "data" / "probe-interact"
    a = G.current_records(p0 / "stage_a.jsonl", phase0)
    if not a:
        print(f"no Phase 0 Stage A records in context {phase0}: run `fly grid` first")
        return 2
    sets = I.taste_sets(ctx)
    cal_path = out / "calibration.json"
    if args.stage in ("calib", "all"):
        cal = I.calibrate(ctx, sets, cal_path)
        print("\n".join(IR.calibration_lines(cal)))
    else:
        cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    if args.stage in ("b", "all"):
        why = I.calibration_blocks(cal, context)
        if why:
            print(f"stage B refused: {why}")
            return 2
        rows = I.stage_b(ctx, sets, a, G.current_records(p0 / "stage_b.jsonl", phase0),
                         out / "stage_b.jsonl", limit=args.limit)
        print(f"stage B: {len(rows)} probed; "
              f"{I.unprobed(a, rows)} Stage A passer(s) not probed yet")
    else:
        rows = G.current_records(out / "stage_b.jsonl", context)
    if args.stage in ("c", "all"):
        left = I.unprobed(a, rows)
        if left:
            print(f"stage C refused: {left} Stage A passer(s) have no Stage B row; "
                  "run `fly interact --stage b` without --limit first")
            return 2
        c = I.stage_c(ctx, sets, rows, out / "stage_c.json")
        print("\n".join(IR.reading_lines(c)))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from lfg_fly import paths
    from lfg_fly.teacher import interact_report, report

    data, docs = paths.repo_root() / "data", paths.repo_root() / "docs"
    report.write_report(data / "probe", docs / "PHASE0.md")
    if (data / "probe-interact" / "stage_c.json").exists():
        interact_report.write_report(data / "probe-interact", docs / "PHASE0B.md")
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

    p = sub.add_parser("columns", help="assign photoreceptor columns from synaptic partners")
    p.add_argument("--min-syn", type=int, default=3)
    p.set_defaults(func=_cmd_columns)

    p = sub.add_parser("catalog", help="read the hero's wardrobe catalog from LFG's public API")
    p.add_argument("--api", default="http://localhost:8176")
    p.add_argument("--body", default="male")
    p.set_defaults(func=_cmd_catalog)

    p = sub.add_parser("grid",
                       help="run the Phase 0 probe grid (Stages A, B, C) and write a verdict")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--cap", type=int, default=None,
                   help="probe at most N Stage A passers in Stage B (default: every passer, as "
                        "spec §3.0 requires; a cap is disclosed in verdict.json and on the "
                        "VERDICT line, and it makes a FAIL inconclusive)")
    p.add_argument("--stage", choices=["a", "b", "c", "all"], default="all")
    p.set_defaults(func=_cmd_grid)

    p = sub.add_parser("interact",
                       help="run Phase 0b, the interaction probe (spec §3.0b), on Phase 0's grid")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-syn", type=int, default=3)
    p.add_argument("--pairs", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None,
                   help="probe only the first N Stage A passers in Stage B (best Phase 0 score "
                        "first): the reproduction pre-flight. Stage C refuses a partial Stage B")
    p.add_argument("--stage", choices=["calib", "b", "c", "all"], default="all")
    p.set_defaults(func=_cmd_interact)

    p = sub.add_parser("report", help="write docs/PHASE0.md (and PHASE0B.md) from data/")
    p.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 0
    return int(args.func(args) or 0)
