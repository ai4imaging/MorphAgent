@echo off
setlocal EnableExtensions
REM MorphAgent UI installer for Windows (double-click this file).
REM Uses ExecutionPolicy Bypass so Explorer "Run with PowerShell" is not needed.
REM Sets UTF-8 (chcp 65001) so GBK consoles do not mojibake, and so pip gets UTF-8
REM (PowerShell 5.1 Set-Content otherwise writes UTF-16LE).
REM Creates: morphagent (UI) + morphagent_sandbox (feature code) + optional morphagent_allen.
REM Always pauses at the end. Full log is kept under MorphAgent_UI\logs\.

cd /d "%~dp0\.."
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
  echo.
  pause
  exit /b 1
)

REM Let the .bat own the final pause so one Enter is enough; ps1 still writes logs.
set "MORPHAGENT_NO_PAUSE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CONDA_REPORT_ERRORS=false"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Setup log / status (kept on disk even after this window closes):
if exist "%CD%\logs\setup_last_status.txt" (
  echo  --- logs\setup_last_status.txt ---
  type "%CD%\logs\setup_last_status.txt"
  echo  ----------------------------------
) else (
  echo  (status file not found yet: logs\setup_last_status.txt)
)
echo ============================================================
echo.

if not "%EXITCODE%"=="0" (
  echo  Setup FAILED ^(exit %EXITCODE%^).
  echo  Common causes:
  echo    1^) Miniconda / Miniforge / Anaconda is not installed
  echo    2^) Network blocked while downloading packages
  echo    3^) conda not on PATH - this script auto-finds common installs;
  echo       if it still fails, install Miniconda and reopen this .bat
  echo    4^) Old scripts used conda.bat with specs containing "^<" which
  echo       cmd treats as file redirection ("file not found"). Pull latest.
  echo  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
  echo  Or Miniforge:      https://github.com/conda-forge/miniforge
  echo  Then double-click this file again.
  echo  Do NOT use Explorer "Run with PowerShell" on the .ps1 - use this .bat.
  echo.
  echo  Press any key to close this window...
  pause >nul
  exit /b %EXITCODE%
)

echo  Setup finished. Envs: morphagent + morphagent_sandbox
echo  ^(plus morphagent_allen if Allen install succeeded^).
echo  Next: double-click scripts\start_ui_windows.bat
echo.
echo  Press any key to close this window...
pause >nul
exit /b 0
