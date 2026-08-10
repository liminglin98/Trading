@echo off
chcp 65001 >nul
REM ============================================================================
REM run_minute_updater.bat
REM Launches minute_updater.py's live polling loop (SSE ETF options + CFFEX
REM index options). The script itself waits for market open, polls roughly
REM once a minute through both trading sessions, and exits after the 15:00
REM close -- so this only needs a single daily Task Scheduler trigger shortly
REM before 9:30 (see install_minute_task.bat), not a repeating trigger.
REM
REM Requires this repo's dependencies (requests, akshare, pandas, scipy),
REM which live in the .venv-win virtualenv at the repo root (see
REM setup_options_lab.bat).
REM ============================================================================

setlocal
set "ROOT=%~dp0..\..\"
set "PYEXE=%ROOT%.venv-win\Scripts\python.exe"
set "LAB=%ROOT%analysis\options_lab"
set "LOG=%LAB%\minute_updater.log"

if not exist "%PYEXE%" (
  echo No Windows environment found at %PYEXE%
  echo Run setup_options_lab.bat first.
  if "%1"=="" pause
  exit /b 1
)

cd /d "%LAB%" || exit /b 1

echo ---------------------------------------- >> "%LOG%"
echo Run started: %DATE% %TIME% >> "%LOG%"

"%PYEXE%" minute_updater.py >> "%LOG%" 2>&1

echo Run finished: %DATE% %TIME% >> "%LOG%"
endlocal
