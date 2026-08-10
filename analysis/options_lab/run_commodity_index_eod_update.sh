#!/usr/bin/env bash
# =============================================================================
# run_commodity_index_eod_update.sh -- daily EOD job for the two live sources
# not covered by run_dashboard_update.sh / run_snapshot.sh (SSE/SZSE ETF
# options): commodity_updater.py (SHFE/DCE/CZCE/GFEX commodity options ->
# Data/options/option_daily_{CODE}.csv) and cffex_updater.py (CFFEX index
# options -> Data/买断日更/consolidated/{underlying}.parquet).
#
# Scheduled later than run_dashboard_update.sh's 16:10/17:10/18:10 China-time
# windows: commodity exchange settlement data (SHFE/CZCE/GFEX) typically
# publishes a bit after the SSE/SZSE feeds. Both updaters are safe to rerun
# (they replace same-day rows rather than duplicate), so extra runs are
# no-ops once the day's data is on file.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

VENV_PY="$HERE/../../.venv/bin/python"
PYTHON="${PYTHON:-$([ -x "$VENV_PY" ] && echo "$VENV_PY" || echo python3)}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

LOG="$HERE/commodity_index_eod_update.log"
echo "----------------------------------------" >> "$LOG"
echo "Run started: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
"$PYTHON" commodity_updater.py >> "$LOG" 2>&1
"$PYTHON" cffex_updater.py >> "$LOG" 2>&1
echo "Run finished: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
