# MorphAgent UI launcher (Windows). Prefer double-clicking start_ui_windows.bat
# (ExecutionPolicy Bypass + conda auto-discovery). Advanced:
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_ui_windows.ps1
param(
    [string]$EnvName = "morphagent"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HandoffRoot = Split-Path -Parent $ScriptDir
$Launcher = Join-Path $HandoffRoot "MorphAgent\launch_ui.py"

function Wait-IfInteractive {
    param([int]$ExitCode = 0)
    if ($env:MORPHAGENT_NO_PAUSE -eq "1") { return }
    try {
        $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction SilentlyContinue).ParentProcessId
        $parentName = if ($parent) {
            (Get-CimInstance Win32_Process -Filter "ProcessId=$parent" -ErrorAction SilentlyContinue).Name
        } else { "" }
    } catch {
        $parentName = ""
    }
    if (($parentName -match '(?i)explorer\.exe') -or ($ExitCode -ne 0)) {
        Write-Host ""
        Write-Host "Press Enter to close this window..."
        try { [void][Console]::ReadLine() } catch { Start-Sleep -Seconds 12 }
    }
}

function Initialize-Conda {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        Write-Host "[OK] conda is on PATH: $((Get-Command conda).Source)"
        return
    }

    $candidates = @(
        $env:CONDA_ROOT,
        $env:CONDA_PREFIX,
        (Join-Path $env:USERPROFILE "miniconda3"),
        (Join-Path $env:USERPROFILE "Miniconda3"),
        (Join-Path $env:USERPROFILE "anaconda3"),
        (Join-Path $env:USERPROFILE "Anaconda3"),
        (Join-Path $env:USERPROFILE "miniforge3"),
        (Join-Path $env:USERPROFILE "Miniforge3"),
        (Join-Path $env:USERPROFILE "mambaforge"),
        (Join-Path $env:USERPROFILE "Mambaforge"),
        (Join-Path $env:LOCALAPPDATA "miniconda3"),
        (Join-Path $env:LOCALAPPDATA "Miniconda3"),
        (Join-Path $env:LOCALAPPDATA "anaconda3"),
        (Join-Path $env:LOCALAPPDATA "Miniforge3"),
        (Join-Path $env:ProgramData "miniconda3"),
        (Join-Path $env:ProgramData "anaconda3"),
        (Join-Path $env:ProgramData "Miniforge3"),
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\tools\miniconda3"
    ) | Where-Object { $_ -and $_.Trim() -ne "" } | Select-Object -Unique

    foreach ($root in $candidates) {
        $condaExe = Join-Path $root "Scripts\conda.exe"
        $condaBat = Join-Path $root "condabin\conda.bat"
        if (-not (Test-Path $condaExe) -and -not (Test-Path $condaBat)) {
            # When CONDA_PREFIX points at an env, try its parent install root.
            $parentRoot = Split-Path (Split-Path $root -Parent) -Parent
            if ($parentRoot) {
                $condaExe = Join-Path $parentRoot "Scripts\conda.exe"
                $condaBat = Join-Path $parentRoot "condabin\conda.bat"
                $root = $parentRoot
            }
        }
        if (-not (Test-Path $condaExe) -and -not (Test-Path $condaBat)) { continue }

        $prepend = @(
            (Join-Path $root "condabin"),
            (Join-Path $root "Scripts"),
            (Join-Path $root "Library\bin"),
            $root
        ) -join ";"
        $env:Path = "$prepend;$env:Path"
        if (Get-Command conda -ErrorAction SilentlyContinue) {
            Write-Host "[OK] Found conda under $root and added it to PATH for this session."
            return
        }
    }

    throw @"
conda was not found.

Please install Miniconda/Miniforge, then double-click scripts\setup_windows.bat first.
未找到 conda。请先安装 Miniconda/Miniforge，再双击 setup_windows.bat 完成安装。
"@
}

$scriptExit = 0
try {
    Set-Location $HandoffRoot
    if (-not (Test-Path $Launcher)) {
        throw "Missing launcher: $Launcher"
    }
    Initialize-Conda
    & conda run --no-capture-output -n $EnvName python $Launcher
    if ($LASTEXITCODE -ne 0) {
        throw "MorphAgent UI exited with code $LASTEXITCODE"
    }
} catch {
    $scriptExit = 1
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Wait-IfInteractive -ExitCode $scriptExit
}

exit $scriptExit
