@echo off
REM ============================================================================
REM run_daily_push.bat
REM Daily WeChat push job for options_lab: builds the alerts + overnight
REM overseas market digest and sends it via PushPlus (see daily_push.py).
REM Intended to run once each weekday morning via Windows Task Scheduler,
REM before the 9:30 market open (~8:30am China time).
REM
REM Requires: `python` on PATH with this repo's dependencies installed
REM (requests, akshare, pandas). If you launch the app from a venv/conda,
REM replace `python` below with the full interpreter path, e.g.
REM "C:\Users\earth\miniconda3\python.exe" (same note as run_snapshot.bat).
REM ============================================================================

setlocal
cd /d "D:\My Drive\Trading\analysis\options_lab"

echo ---------------------------------------- >> daily_push_bat.log
echo Run started: %DATE% %TIME% >> daily_push_bat.log

python daily_push.py >> daily_push_bat.log 2>&1

echo Run finished: %DATE% %TIME% >> daily_push_bat.log
endlocal
