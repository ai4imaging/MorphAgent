@echo off
setlocal EnableExtensions
REM MorphAgent UI Lite installer for Windows (double-click this file).

set "EXITCODE=0"
set "HANDOFF="

cd /d "%~dp0\.."
if errorlevel 1 (
  echo [ERROR] Could not cd to MorphAgent_UI_Lite folder from "%~dp0"
  set "EXITCODE=1"
  goto :HOLD
)
set "HANDOFF=%CD%"
title MorphAgent UI Lite Setup

chcp 65001 >nul 2>&1

echo ============================================================
echo  MorphAgent UI Lite setup (Windows)
echo  Scope: Tau demo + Code/VLM
echo  Knowledge: precomputed txt (skip PDF/PubMed/Allen)
echo  Working directory: %CD%
echo  Creates single conda env: morphagent_lite
echo  Live PDF/literature/Allen: use MorphAgent_UI\
echo ============================================================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
  echo [ERROR] PowerShell was not found on PATH.
  set "EXITCODE=1"
  goto :HOLD
)

if not exist "%~dp0setup_windows.ps1" (
  echo [ERROR] Missing script: "%~dp0setup_windows.ps1"
  set "EXITCODE=1"
  goto :HOLD
)

set "MORPHAGENT_NO_PAUSE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CONDA_REPORT_ERRORS=false"
REM Do not force CONDA_NO_PLUGINS / CONDA_SOLVER=classic here.
REM setup_windows.ps1 accepts ToS and only falls back to classic for a tiny python+pip create.

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

goto :HOLD

:HOLD
echo.
echo ============================================================
if defined HANDOFF if exist "%HANDOFF%\logs\setup_last_status.txt" (
  echo  --- logs\setup_last_status.txt ---
  type "%HANDOFF%\logs\setup_last_status.txt"
  echo.
)
if "%EXITCODE%"=="0" (
  echo  Setup finished OK. Next: double-click start_ui_windows.bat
) else (
  echo  Setup FAILED ^(exit %EXITCODE%^). See logs\ above if present.
)
echo ============================================================
echo.
pause
endlocal & exit /b %EXITCODE%
