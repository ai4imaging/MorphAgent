# MorphAgent UI Lite Windows setup — single env morphagent_lite.
# ASCII-only. Dot-sources conda_windows.ps1 for conda discovery.
#
# Avoids classic+conda-forge mega-solves that crash old conda.exe (0xc0000005).
# Flow: accept ToS -> conda create python+pip only -> pip install requirements.
param(
    [string]$EnvName = $(if ($env:MORPHAGENT_ENV_NAME) { $env:MORPHAGENT_ENV_NAME } else { "morphagent_lite" }),
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$StatusFile = Join-Path $LogDir "setup_last_status.txt"
$ReqFile = Join-Path $RepoRoot "dependencies\requirements-lite.txt"
$EnvFile = Join-Path $RepoRoot ".env"
$EnvExample = Join-Path $RepoRoot ".env.example"

function Write-Status([string]$Text) {
    Set-Content -LiteralPath $StatusFile -Value $Text -Encoding UTF8
}

. (Join-Path $ScriptDir "conda_windows.ps1")
Ensure-CondaUtf8Env

function Find-CondaExe {
    $root = Find-CondaRootOnDisk
    if (-not $root) { return $false }
    return [bool](Use-CondaRoot $root)
}

try {
    if (-not (Find-CondaExe)) {
        throw "conda.exe not found. Install Miniconda/Anaconda, then re-run setup_windows.bat."
    }
    if (-not (Test-PathSafe $ReqFile)) {
        throw "Missing requirements: $ReqFile"
    }

    Write-Host "[OK] Using conda: $($script:CondaExe)"
    Accept-AnacondaTosBestEffort
    Write-Host "[OK] Desktop: conda only creates python+pip; science/Qt packages install via pip"
    Write-Host "[OK] CONDA_NO_PLUGINS=$($env:CONDA_NO_PLUGINS) CONDA_SOLVER=$($env:CONDA_SOLVER)"
    Write-Host "[..] New Qt6/WebEngine workspace + lightweight Code/VLM analysis"
    Write-Host "[..] No automatic PubMed retrieval or Allen installation"
    
    $envList = & $script:CondaExe env list 2>$null | Out-String
    $exists = $envList -match "(?m)^\s*$([regex]::Escape($EnvName))\s"
    if ($exists -and ($Recreate -or $env:MORPHAGENT_RECREATE_ENVS -eq "1")) {
        Write-Host "[..] Removing $EnvName"
        & $script:CondaExe env remove -n $EnvName -y
        if ($LASTEXITCODE -ne 0) { throw "conda env remove failed ($LASTEXITCODE)" }
        $exists = $false
    }
    if (-not $exists) {
        New-MorphAgentLiteEnv -EnvName $EnvName
    } else {
        Write-Host "[OK] Env $EnvName already exists"
    }

    # Clear classic fallback before pip/run so we do not keep a broken solver mode around.
    Clear-CondaClassicFallback
    Ensure-CondaUtf8Env

    Write-Host "[..] Upgrading pip"
    & $script:CondaExe run --no-capture-output -n $EnvName python -m pip install -U pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed ($LASTEXITCODE)" }

    Write-Host "[..] Installing science dependencies, PySide6/WebEngine, and legacy PyQt5 via pip"
    & $script:CondaExe run --no-capture-output -n $EnvName python -m pip install -r $ReqFile --retries 5
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Full pip install failed; retrying PyQt5 via pip only, then requirements again"
        & $script:CondaExe run --no-capture-output -n $EnvName python -m pip install "PyQt5>=5.15,<5.16" qtpy --retries 5
        & $script:CondaExe run --no-capture-output -n $EnvName python -m pip install -r $ReqFile --retries 5
        if ($LASTEXITCODE -ne 0) {
            throw "pip install requirements failed ($LASTEXITCODE). Check network/SSL; do not conda-install the science stack on Lite."
        }
    }

    Write-Host "[..] pip install -e MorphAgent"
    $MorphRoot = $RepoRoot
    & $script:CondaExe run --no-capture-output -n $EnvName python -m pip install -e $MorphRoot
    if ($LASTEXITCODE -ne 0) { throw "pip install -e MorphAgent failed ($LASTEXITCODE)" }

    if (-not (Test-PathSafe $EnvFile) -and (Test-PathSafe $EnvExample)) {
        Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
    }
    if (Test-PathSafe $EnvFile) {
        $lines = Get-Content -LiteralPath $EnvFile -Encoding UTF8
        $wanted = @{
            "CONDA_ENV" = $EnvName
            "SEGMENTATION_BACKEND" = "none"
        }
        $seen = @{}
        $out = New-Object System.Collections.Generic.List[string]
        foreach ($line in $lines) {
            if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith("#") -or ($line -notmatch "=")) {
                if ($line -like "SEGMENTATION_CONDA_ENV=*") { continue }
                [void]$out.Add($line)
                continue
            }
            $key = ($line -split "=", 2)[0].Trim()
            if ($wanted.ContainsKey($key)) {
                [void]$out.Add("$key=$($wanted[$key])")
                $seen[$key] = $true
            } elseif ($key -eq "SEGMENTATION_CONDA_ENV") {
                continue
            } else {
                [void]$out.Add($line)
            }
        }
        foreach ($key in $wanted.Keys) {
            if (-not $seen.ContainsKey($key)) {
                [void]$out.Add("$key=$($wanted[$key])")
            }
        }
        Set-Content -LiteralPath $EnvFile -Value $out -Encoding UTF8
        Write-Host "[OK] Updated $EnvFile"
    }

    Write-Host "[..] verify_install.py"
    $Verify = Join-Path $ScriptDir "verify_install.py"
    & $script:CondaExe run --no-capture-output -n $EnvName python $Verify
    if ($LASTEXITCODE -ne 0) { throw "verify_install.py failed ($LASTEXITCODE)" }

    Write-Status "OK env=$EnvName"
    Write-Host "[OK] Desktop setup complete in $EnvName. Next: start_ui_windows.bat"
    Write-Host "[..] Opens the new desktop workspace; enter your API connection in Settings."
    exit 0
} catch {
    $msg = "$_"
    Write-Host "[ERROR] $msg"
    Write-Status "FAIL $msg"
    if ($env:MORPHAGENT_NO_PAUSE -ne "1") { pause }
    exit 1
}
