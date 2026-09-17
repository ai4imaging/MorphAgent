@echo off
setlocal
call "%~dp0start_ui_windows.bat" %*
set "DESKTOP_EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %DESKTOP_EXIT_CODE%
