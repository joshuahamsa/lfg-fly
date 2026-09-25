"""`python -m lfg_fly`: the `fly` command line (spec §1: pm2 runs `<venv python> -m lfg_fly`
under `nice -n 10 ionice -c3`). `lfg_fly/__init__.py` has already applied the thread caps."""

from lfg_fly.cli import main

raise SystemExit(main())
