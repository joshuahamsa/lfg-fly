// pm2 entries for the fly's day (spec §4 table), nice + ionice wrapped (spec §1 CPU
// discipline). pm2 has no nice option, so each job's `script` is /usr/bin/nice and the venv
// python is an argument; `interpreter: "none"` runs the script directly.
//
//   fly-claim    04:10 UTC  claim the BRIX drip, poll its status
//   fly-retrain  04:30 UTC  supply snapshot -> live-concentration re-simulation -> rarity head
//   fly-move     15:00 UTC  the move loop
//
// Times are UTC (this box is Etc/UTC; pm2 schedules crons in the daemon's zone), clear of
// LFG's own cron slots (00:10-03:40 UTC). Each job runs once and exits: `autorestart: false`
// keeps pm2 from re-running it, `autostart: false` keeps `pm2 start ecosystem.config.js`
// from firing a run outside its slot (a stray fly-move is a real move), and `cron_restart`
// starts it on schedule (pm2 registers the cron before it honours autostart, so a stopped
// job still fires).
//
//   pm2 start ecosystem.config.js && pm2 save
//   pm2 start fly-claim          # run one job by hand, off schedule
//   pm2 logs fly-move --lines 200
//
// Configuration comes from ~/lfg-fly/.env, which the CLI loads (FLY_NETWORK, FLY_API_BASE,
// FLY_ENABLED, ...; the RegularKey seed on mainnet). The python is `-m lfg_fly`, this
// checkout's package via lfg_fly/__main__.py (lfg_fly/cli.py has no __main__ guard).
// FLY_PYTHON overrides the interpreter (default: this checkout's .venv).

const path = require("path");

const root = __dirname;
const python = process.env.FLY_PYTHON || path.join(root, ".venv", "bin", "python");

function job(name, cron, command) {
  return {
    name,
    cwd: root,
    script: "/usr/bin/nice",
    args: ["-n", "10", "ionice", "-c3", python, "-m", "lfg_fly", ...command],
    interpreter: "none",
    cron_restart: cron,
    autorestart: false,
    autostart: false,
    instances: 1,
    exec_mode: "fork",
    time: true,
    merge_logs: true,
    kill_timeout: 60000,
    env: { PYTHONUNBUFFERED: "1", TZ: "UTC" },
  };
}

module.exports = {
  apps: [
    job("fly-claim", "10 4 * * *", ["claim"]),
    job("fly-retrain", "30 4 * * *", ["retrain"]),
    job("fly-move", "0 15 * * *", ["move"]),
  ],
};
