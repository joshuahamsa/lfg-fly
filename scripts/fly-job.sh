#!/usr/bin/env bash
# pm2 entry for the fly's jobs (spec §1 CPU discipline): nice + ionice around the venv python.
# Usage: scripts/fly-job.sh <fly subcommand> [args...]   (FLY_PYTHON overrides the interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${FLY_PYTHON:-$PWD/.venv/bin/python}"
exec /usr/bin/nice -n 10 ionice -c3 "$PY" -m lfg_fly "$@"
