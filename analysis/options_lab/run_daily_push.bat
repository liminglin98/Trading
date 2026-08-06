@echo off
chcp 65001 >nul
REM ============================================================================
REM run_daily_push.bat
REM Daily WeChat push job for options_lab: builds the alerts + overnight
REM overseas market digest and sends it via PushPlus (see daily_push.py).
REM Intended to run once each weekday morning via Windows Task Scheduler,
REM before the 9:30 market open (~8:30am China time).
REM
REM Requires this repo's dependencies (requests, akshare, pandas), which live
REM in the .venv-win virtualenv at the repo root (see setup_options_lab.bat).
REM ============================================================================

setlocal
set "ROOT=C:\Users\流浪猫\Nextcloud2\My Drive\My Drive\Trading\"
set "PYEXE=%ROOT%.venv-win\Scripts\python.exe"
cd /d "%ROOT%analysis\options_lab"

echo ---------------------------------------- >> daily_push_bat.log
echo Run started: %DATE% %TIME% >> daily_push_bat.log

"%PYEXE%" daily_push.py >> daily_push_bat.log 2>&1

echo Run finished: %DATE% %TIME% >> daily_push_bat.log
endlocal
