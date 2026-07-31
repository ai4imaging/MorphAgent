# MorphAgent UI launcher (Windows). Prefer double-clicking start_ui_windows.bat
# (ExecutionPolicy Bypass + conda auto-discovery). Advanced:
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_ui_windows.ps1
#
# IMPORTANT: this file is ASCII-only (no non-ASCII in comments or strings) so
# Windows PowerShell 5.1 / GBK consoles never mis-decode the script.
param(
    [string]$EnvName = "morphagent"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HandoffRoot = Split-Path -Parent $ScriptDir
$Launcher = Join-Path $HandoffRoot "MorphAgent\launch_ui.py"
$script:CondaExe = $null

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

function Resolve-CondaExeFromRoot([string]$Root) {
    if (-not $Root) { return $null }
    foreach ($c in @(
        (Join-Path $Root "Scripts\conda.exe"),
        (Join-Path $Root "condabin\conda.exe"),
        (Join-Path $Root "Library\bin\conda.exe")
    )) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Initialize-Conda {
    # Prefer conda.exe (not conda.bat) so args with < > = are not eaten by cmd.
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) {
        $src = $cmd.Source
        if ($src -like '*.exe') {
            $script:CondaExe = $src
            Write-Host "[OK] conda.exe: $($script:CondaExe)"
            return
        }
        $dir = Split-Path -Parent $src
        $root = Split-Path -Parent $dir
        $script:CondaExe = Resolve-CondaExeFromRoot $root
        if (-not $script:CondaExe) {
            $script:CondaExe = Resolve-CondaExeFromRoot (Split-Path -Parent $root)
        }
        if ($script:CondaExe) {
            Write-Host "[OK] conda.exe: $($script:CondaExe)"
            return
        }
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
        "C:\tools\miniconda3",
        "D:\Anaconda3",
        "D:\Miniconda3"
    ) | Where-Object { $_ -and $_.Trim() -ne "" } | Select-Object -Unique

    foreach ($root in $candidates) {
        $condaExe = Resolve-CondaExeFromRoot $root
        $condaBat = Join-Path $root "condabin\conda.bat"
        if (-not $condaExe -and -not (Test-Path $condaBat)) {
            $parentRoot = Split-Path (Split-Path $root -Parent) -Parent
            if ($parentRoot) {
                $condaExe = Resolve-CondaExeFromRoot $parentRoot
                $condaBat = Join-Path $parentRoot "condabin\conda.bat"
                $root = $parentRoot
            }
        }
        if (-not $condaExe -and -not (Test-Path $condaBat)) { continue }

        $prepend = @(
            (Join-Path $root "condabin"),
            (Join-Path $root "Scripts"),
            (Join-Path $root "Library\bin"),
            $root
        ) -join ";"
        $env:Path = "$prepend;$env:Path"

        if (-not $condaExe) { $condaExe = Resolve-CondaExeFromRoot $root }
        if ($condaExe) {
            $script:CondaExe = $condaExe
            Write-Host "[OK] Found conda.exe under $root"
            Write-Host "     $($script:CondaExe)"
            return
        }
    }

    throw @"
conda was not found.

Please install Miniconda/Miniforge/Anaconda, then double-click scripts\setup_windows.bat first.
Do not use Explorer 'Run with PowerShell' on this .ps1 - use start_ui_windows.bat.
"@
}

$scriptExit = 0
try {
    try { cmd /c "chcp 65001 >nul" | Out-Null } catch {}
    if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = "1" }
    if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }

    Set-Location $HandoffRoot
    if (-not (Test-Path $Launcher)) {
        throw "Missing launcher: $Launcher"
    }
    Initialize-Conda
    & $script:CondaExe run --no-capture-output -n $EnvName python $Launcher
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
