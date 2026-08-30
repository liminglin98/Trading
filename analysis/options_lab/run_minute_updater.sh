#!/usr/bin/env bash
# =============================================================================
# run_minute_updater.sh -- launches minute_updater.py's live polling loop
# (SSE ETF options + CFFEX index options). The script itself waits for
# market open, polls roughly once a minute through both trading sessions,
# and exits after the 15:00 close -- so this only needs a single daily cron
# trigger shortly before 9:30 China time (see the crontab block this script
# ships with), not a repeating trigger.
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
VENV_PY="$HERE/../../.venv/bin/python"
PYTHON="${PYTHON:-$([ -x "$VENV_PY" ] && echo "$VENV_PY" || echo python3)}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
LOG="$HERE/minute_updater_sh.log"
echo "----------------------------------------" >> "$LOG"
echo "Run started: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
"$PYTHON" minute_updater.py >> "$LOG" 2>&1
echo "Run finished: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
