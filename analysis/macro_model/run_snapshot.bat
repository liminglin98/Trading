@echo off
REM EOD snapshot of all open saved positions -> position_tracking.csv
REM cd to this script's own folder so imports + CSVs resolve correctly.
cd /d "%~dp0"

REM Prefer the Windows "py" launcher; fall back to python on PATH.
where py >nul 2>nul && (
    py -3 pipeline.py --snapshot >> snapshot_log.txt 2>&1
) || (
    python pipeline.py --snapshot >> snapshot_log.txt 2>&1
)
