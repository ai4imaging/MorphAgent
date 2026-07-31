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

function Use-CondaRoot([string]$Root) {
    $condaExe = Resolve-CondaExeFromRoot $Root
    if (-not $condaExe) { return $false }
    $prepend = @(
        (Join-Path $Root "condabin"),
        (Join-Path $Root "Scripts"),
        (Join-Path $Root "Library\bin"),
        $Root
    ) -join ";"
    $env:Path = "$prepend;$env:Path"
    $env:CONDA_ROOT = $Root
    $script:CondaExe = $condaExe
    Write-Host "[OK] conda.exe: $($script:CondaExe)"
    return $true
}

function Install-MinicondaSilent {
    if ($env:MORPHAGENT_SKIP_MINICONDA_INSTALL -eq "1") {
        throw "conda was not found, and MORPHAGENT_SKIP_MINICONDA_INSTALL=1 disables auto-install."
    }

    $prefix = Join-Path $env:USERPROFILE "miniconda3"
    if (Use-CondaRoot $prefix) { return }

    Write-Host "[INFO] conda not found. Installing Miniconda silently (one-time)..."
    Write-Host "       prefix: $prefix"
    Write-Host "       Prefer running scripts\setup_windows.bat first for a full install."

    $isArm = ($env:PROCESSOR_ARCHITECTURE -match 'ARM64') -or ("$env:PROCESSOR_IDENTIFIER" -match 'ARM')
    $urls = New-Object System.Collections.Generic.List[string]
    if ($isArm) {
        [void]$urls.Add("https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-arm64.exe")
    }
    [void]$urls.Add("https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe")

    $installer = Join-Path $env:TEMP "MorphAgent-Miniconda3-latest-Windows.exe"
    $downloaded = $false
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch {}

    foreach ($url in $urls) {
        try {
            Write-Host "  downloading: $url"
            if (Test-Path -LiteralPath $installer) {
                Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
            }
            Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
            if ((Test-Path -LiteralPath $installer) -and ((Get-Item -LiteralPath $installer).Length -gt 1MB)) {
                $downloaded = $true
                break
            }
        } catch {
            Write-Host "  [WARN] download failed: $($_.Exception.Message)"
        }
    }
    if (-not $downloaded) {
        throw "Failed to download Miniconda. Run scripts\setup_windows.bat after installing Miniconda manually."
    }

    Write-Host "  running silent installer..."
    $argList = "/S /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /D=$prefix"
    $proc = Start-Process -FilePath $installer -ArgumentList $argList -Wait -PassThru
    if (-not (Use-CondaRoot $prefix)) {
        $code = if ($null -ne $proc) { [int]$proc.ExitCode } else { -1 }
        throw "Miniconda install finished but conda.exe missing under $prefix (exit $code). Run setup_windows.bat after a manual Miniconda install."
    }
    Write-Host "[OK] Miniconda installed at $prefix"
    try { Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue } catch {}
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
        if (Use-CondaRoot $root) {
            Write-Host "[OK] Found conda under $root"
            return
        }
    }

    Install-MinicondaSilent
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
