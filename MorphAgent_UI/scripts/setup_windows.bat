@echo off
setlocal EnableExtensions
REM MorphAgent UI installer for Windows (double-click this file).
REM Uses ExecutionPolicy Bypass so Explorer "Run with PowerShell" is not needed.
REM Also sets UTF-8 for pip (PowerShell 5.1 otherwise writes UTF-16LE temp files).

cd /d "%~dp0\.."
title MorphAgent UI Setup

echo ============================================================
echo  MorphAgent UI setup (Windows)
echo  Working directory: %CD%
echo  Scripts live under: MorphAgent_UI\scripts\
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

set "MORPHAGENT_NO_PAUSE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
  echo ============================================================
  echo  Setup FAILED ^(exit %EXITCODE%^).
  echo  Common causes:
  echo    1^) Miniconda / Miniforge / Anaconda is not installed
  echo    2^) Network blocked while downloading packages
  echo    3^) conda not on PATH — this script auto-finds common installs;
  echo       if it still fails, install Miniconda and reopen this .bat
  echo  Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
  echo  Or Miniforge:      https://github.com/conda-forge/miniforge
  echo  Then double-click this file again.
  echo  Do NOT use Explorer "Run with PowerShell" on the .ps1 — use this .bat.
  echo ============================================================
  echo.
  pause
  exit /b %EXITCODE%
)

echo ============================================================
echo  Setup finished. You can close this window, then double-click
echo  scripts\start_ui_windows.bat to launch MorphAgent.
echo ============================================================
echo.
pause
exit /b 0
