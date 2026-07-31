@echo off
setlocal EnableExtensions
REM MorphAgent UI installer for Windows (double-click this file).
REM IMPORTANT: every exit path ends at :HOLD with "pause" so the window never flash-closes.
REM Full log is kept under MorphAgent_UI\logs\.

set "EXITCODE=0"
set "HANDOFF="

cd /d "%~dp0\.."
if errorlevel 1 (
  echo [ERROR] Could not cd to MorphAgent_UI folder from "%~dp0"
  set "EXITCODE=1"
  goto :HOLD
)
set "HANDOFF=%CD%"
title MorphAgent UI Setup

REM Prefer UTF-8 console; ignore failure on old systems.
chcp 65001 >nul 2>&1

echo ============================================================
echo  MorphAgent UI setup (Windows)
echo  Working directory: %CD%
echo  Scripts live under: MorphAgent_UI\scripts\
echo  Will create:
echo    - morphagent          (Qt UI + agent)
echo    - morphagent_sandbox  (frozen extract() science stack)
echo    - morphagent_allen    (optional Allen segmentation)
echo  Logs will be saved under: MorphAgent_UI\logs\
echo ============================================================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
  echo [ERROR] PowerShell was not found on PATH.
  echo Install Windows PowerShell or PowerShell 7, then try again.
  set "EXITCODE=1"
  goto :HOLD
)

if not exist "%~dp0setup_windows.ps1" (
  echo [ERROR] Missing script: "%~dp0setup_windows.ps1"
  set "EXITCODE=1"
  goto :HOLD
)

REM .bat owns the final pause (see :HOLD). ps1 skips its own pause.
set "MORPHAGENT_NO_PAUSE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CONDA_REPORT_ERRORS=false"
REM Skip conda-anaconda-tos interactive prompt (EOFError in double-click setup).
set "CONDA_NO_PLUGINS=true"
REM CONDA_NO_PLUGINS disables libmamba; use classic solver for this process only.
set "CONDA_SOLVER=classic"

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

goto :HOLD

REM ---- single exit gate: ALWAYS pause, success or failure ----
:HOLD
echo.
echo ============================================================
echo  Setup log / status (kept on disk even after this window closes):
if defined HANDOFF if exist "%HANDOFF%\logs\setup_last_status.txt" (
  echo  --- logs\setup_last_status.txt ---
  type "%HANDOFF%\logs\setup_last_status.txt"
  echo  ----------------------------------
) else if exist "%CD%\logs\setup_last_status.txt" (
  echo  --- logs\setup_last_status.txt ---
  type "%CD%\logs\setup_last_status.txt"
  echo  ----------------------------------
) else (
  echo  (status file not found yet: MorphAgent_UI\logs\setup_last_status.txt)
  echo  If setup crashed early, scroll up for the error text in this window.
)
echo ============================================================
echo.

if not "%EXITCODE%"=="0" (
  echo  Setup FAILED ^(exit %EXITCODE%^).
  echo  Common causes:
  echo    1^) Network blocked while downloading Miniconda / conda packages
  echo    2^) Antivirus blocked the silent Miniconda installer
  echo    3^) Old scripts - git pull latest and retry
  echo    4^) Stale CONDA_ROOT/CONDA_PREFIX pointing at a missing drive
  echo        ^(clear those env vars, then retry^)
  echo  Note: if conda is missing, setup_windows.ps1 auto-installs Miniconda
  echo        into %%USERPROFILE%%\miniconda3 ^(needs network once^).
  echo  Manual Miniconda: https://docs.conda.io/en/latest/miniconda.html
  echo  Then double-click this file again.
  echo  Do NOT use Explorer "Run with PowerShell" on the .ps1 - use this .bat.
) else (
  echo  Setup finished. Envs: morphagent + morphagent_sandbox
  echo  ^(plus morphagent_allen if Allen install succeeded^).
  echo  Next: double-click scripts\start_ui_windows.bat
)

echo.
echo ============================================================
echo  THIS WINDOW STAYS OPEN so you can read the result above.
echo  Press any key to close...
echo ============================================================
pause >nul
endlocal & exit /b %EXITCODE%
