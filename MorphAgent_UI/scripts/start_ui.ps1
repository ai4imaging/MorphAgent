param(
    [string]$EnvName = "morphagent"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HandoffRoot = Split-Path -Parent $ScriptDir
$Launcher = Join-Path $HandoffRoot "MorphAgent\launch_ui.py"

conda run --no-capture-output -n $EnvName python $Launcher

