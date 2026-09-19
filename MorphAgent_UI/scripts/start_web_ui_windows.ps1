# MorphAgent UI Lite Windows web launcher.
param()

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$EnvName = if ($env:MORPHAGENT_ENV_NAME) { $env:MORPHAGENT_ENV_NAME } else { "morphagent_lite" }
$Launch = Join-Path $RepoRoot "launch_web_ui.py"

. (Join-Path $ScriptDir "conda_windows.ps1")
Ensure-CondaUtf8Env

function Find-CondaExe {
    $root = Find-CondaRoot
    if (-not $root) {
        Write-CondaNotFoundHelp
        return $false
    }
    return [bool](Use-CondaRoot $root)
}

try {
    if (-not (Find-CondaExe)) {
        throw "conda not found. Run setup_windows.bat first."
    }
    if (-not (Test-PathSafe $Launch)) {
        throw "Missing $Launch"
    }
    Write-Host "[OK] Launching web UI via env $EnvName"
    & $script:CondaExe run --no-capture-output -n $EnvName python $Launch @args
    exit $LASTEXITCODE
} catch {
    Write-Host "[ERROR] $_"
    if ($env:MORPHAGENT_NO_PAUSE -ne "1") { pause }
    exit 1
}
