@echo off
setlocal EnableExtensions
REM MorphAgent web UI launcher for Windows.

cd /d "%~dp0\.."

where powershell >nul 2>&1
if errorlevel 1 (
  echo [ERROR] PowerShell was not found on PATH.
  pause
  exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "CONDA_REPORT_ERRORS=false"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web_ui_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
  echo.
  echo Launch FAILED ^(exit %EXITCODE%^). Run setup_windows.bat first.
  pause
)
endlocal & exit /b %EXITCODE%
