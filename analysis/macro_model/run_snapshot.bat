@echo off
REM ============================================================================
REM Daily EOD job: snapshot positions + strategies, rebuild the website snapshot,
REM then publish to GitHub so Streamlit Cloud auto-redeploys with fresh data.
REM Point Windows Task Scheduler at this ~30 min after the 15:00 close.
REM cd to this script's own folder so imports + CSVs resolve correctly.
REM ============================================================================
cd /d "%~dp0"

REM Prefer the Windows "py" launcher; fall back to python on PATH.
set "PY=python"
where py >nul 2>nul && set "PY=py -3"

echo. >> snapshot_log.txt
echo ==================== %date% %time% ==================== >> snapshot_log.txt

REM 1. EOD snapshot: strategy tracker + all open saved positions
%PY% pipeline.py --snapshot >> snapshot_log.txt 2>&1

REM 2. Rebuild the website snapshot (heavy: model fit + SSE IV chain)
%PY% precompute.py >> snapshot_log.txt 2>&1

REM 3. Publish -> Streamlit Cloud redeploys on push.
REM    (git add -A only stages the app files; the whitelist .gitignore keeps the
REM    rest of the Trading drive out. A no-op commit just exits quietly.)
git add -A >> snapshot_log.txt 2>&1
git commit -m "Daily snapshot %date%" >> snapshot_log.txt 2>&1
git push >> snapshot_log.txt 2>&1

echo done %date% %time% >> snapshot_log.txt
