@echo off
setlocal
if not defined MORPHAGENT_ENV_NAME set MORPHAGENT_ENV_NAME=morphagent_lite
conda run --no-capture-output -n "%MORPHAGENT_ENV_NAME%" python "%~dp0..\launch_web_ui.py" %*
endlocal
