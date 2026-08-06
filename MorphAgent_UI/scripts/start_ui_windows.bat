@echo off
setlocal EnableExtensions
REM MorphAgent UI launcher for Windows (double-click this file).
REM Uses ExecutionPolicy Bypass; auto-finds conda when PATH is empty.
REM UI runs in morphagent; feature extract() uses morphagent_sandbox (set by setup / UI).

cd /d "%~dp0\.."
title MorphAgent UI

REM Prefer UTF-8 console; ignore failure on old systems.
chcp 65001 >nul 2>&1

echo ============================================================
echo  Starting MorphAgent UI
echo  Working directory: %CD%
echo  UI env: morphagent
echo  Code sandbox ^(extract^): morphagent_sandbox
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
set "CONDA_NO_PLUGINS=true"
set "CONDA_SOLVER=classic"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ui_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" (
  echo ============================================================
  echo  Launch FAILED ^(exit %EXITCODE%^).
  echo  If you have not installed yet, double-click setup_windows.bat first.
  echo  Common causes:
  echo    1^) setup_windows.bat was never run
  echo    2^) conda env "morphagent" is missing / broken
  echo    3^) conda env "morphagent_sandbox" is missing - re-run setup_windows.bat
  echo    4^) Qt / display driver issue - re-run setup_windows.bat
  echo ============================================================
  echo.
  pause
  exit /b %EXITCODE%
)

exit /b 0
