#!/usr/bin/env bash
# =============================================================================
# run_intraday_monitor.sh -- launches intraday_monitor.py: builds today's
# pre-market near-trigger watchlist once, then polls minute_updater's live
# option quotes every 5 minutes through both trading sessions and pushes via
# PushPlus the moment a signal crosses its trigger level. Self-exiting after
# the 15:00 close, so a single daily cron trigger a couple minutes after
# run_minute_updater.sh's is enough. Depends on minute_updater.py already
# running and writing today's Data/options/minute/*.csv files.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
VENV_PY="$HERE/../../.venv/bin/python"
PYTHON="${PYTHON:-$([ -x "$VENV_PY" ] && echo "$VENV_PY" || echo python3)}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
LOG="$HERE/intraday_monitor_sh.log"
echo "----------------------------------------" >> "$LOG"
echo "Run started: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
"$PYTHON" intraday_monitor.py >> "$LOG" 2>&1
echo "Run finished: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
