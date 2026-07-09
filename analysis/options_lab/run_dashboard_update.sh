#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
VENV_PY="$HERE/../../.venv/bin/python"
PYTHON="${PYTHON:-$([ -x "$VENV_PY" ] && echo "$VENV_PY" || echo python3)}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
LOG="$HERE/dashboard_update.log"
echo "----------------------------------------" >> "$LOG"
echo "Run started: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
"$PYTHON" dashboard_data.py >> "$LOG" 2>&1
echo "Run finished: $(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$LOG"
