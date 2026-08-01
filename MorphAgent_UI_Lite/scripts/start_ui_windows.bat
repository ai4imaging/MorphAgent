@echo off
setlocal EnableExtensions
REM MorphAgent UI Lite launcher for Windows (double-click this file).

cd /d "%~dp0\.."
title MorphAgent UI Lite

chcp 65001 >nul 2>&1

echo ============================================================
echo  Starting MorphAgent UI Lite
echo  Working directory: %CD%
echo  Env: morphagent_lite
echo ============================================================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
  echo [ERROR] PowerShell was not found on PATH.
  pause
  exit /b 1
)

set "MORPHAGENT_NO_PAUSE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CONDA_NO_PLUGINS=true"
set "CONDA_SOLVER=classic"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ui_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
  echo Launch FAILED ^(exit %EXITCODE%^). Run setup_windows.bat first.
  pause
)
endlocal & exit /b %EXITCODE%
