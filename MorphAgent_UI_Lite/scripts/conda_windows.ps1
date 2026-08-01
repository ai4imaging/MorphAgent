# Shared Windows conda helpers for setup_windows.ps1 / start_ui_windows.ps1.
# ASCII-only (Windows PowerShell 5.1 / GBK consoles).
# Dot-source from the caller so $script:CondaExe stays in the caller's scope.
#
# Lite policy: never force classic+conda-forge mega-solves (that crashes old conda.exe).
# Prefer: accept Anaconda ToS -> keep plugins/libmamba -> conda create python+pip only -> pip rest.

function Ensure-CondaUtf8Env {
    $env:CONDA_REPORT_ERRORS = "false"
    if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = "1" }
    if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }
}

function Enable-CondaClassicFallback {
    # Last resort for ToS-interactive hangs on old Miniconda.
    # Safe for Lite only because we install python+pip (tiny solve), never the science stack via conda.
    $env:CONDA_NO_PLUGINS = "true"
    $env:CONDA_SOLVER = "classic"
    Write-Host "[WARN] Using classic solver fallback (python+pip create only; science stack stays on pip)"
}

function Clear-CondaClassicFallback {
    Remove-Item Env:\CONDA_NO_PLUGINS -ErrorAction SilentlyContinue
    Remove-Item Env:\CONDA_SOLVER -ErrorAction SilentlyContinue
}

function Accept-AnacondaTosBestEffort {
    # Persist ToS acceptance so plugins/libmamba can stay enabled (avoids classic mega-solves).
    if (-not $script:CondaExe) { return }
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $hadPlugins = $env:CONDA_NO_PLUGINS
        $hadSolver = $env:CONDA_SOLVER
        Clear-CondaClassicFallback
        foreach ($ch in @(
            "https://repo.anaconda.com/pkgs/main",
            "https://repo.anaconda.com/pkgs/r",
            "https://repo.anaconda.com/pkgs/msys2"
        )) {
            try {
                & $script:CondaExe tos accept --override-channels --channel $ch 2>$null | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    & $script:CondaExe tos accept -c $ch 2>$null | Out-Null
                }
            } catch {}
        }
        if ($null -ne $hadPlugins -and "$hadPlugins" -ne "") { $env:CONDA_NO_PLUGINS = $hadPlugins }
        if ($null -ne $hadSolver -and "$hadSolver" -ne "") { $env:CONDA_SOLVER = $hadSolver }
        Write-Host "[OK] Best-effort Anaconda ToS accept (plugins may stay enabled)"
    } catch {
        # ignore missing tos subcommand / plugin
    } finally {
        $ErrorActionPreference = $prevEap
        Ensure-CondaUtf8Env
    }
}

# Backward-compatible alias used by older callers.
function Ensure-CondaNonInteractiveEnv {
    Ensure-CondaUtf8Env
}

function Test-PathSafe {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    try {
        return [bool](Test-Path -LiteralPath $Path -ErrorAction SilentlyContinue)
    } catch {
        return $false
    }
}

function Test-DriveReady {
    param([string]$Path)
    # True when path is usable: existing drive root, or UNC / relative.
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    try {
        if ($Path.StartsWith("\\")) { return $true }
        $root = [System.IO.Path]::GetPathRoot($Path)
        if ([string]::IsNullOrWhiteSpace($root)) { return $true }
        return (Test-PathSafe $root)
    } catch {
        return $false
    }
}

function Get-ReadyDriveLetters {
    # Prefer live FileSystem drives; also probe C-Z so missing D/E/F never throws.
    $found = New-Object System.Collections.Generic.List[string]
    try {
        Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue | ForEach-Object {
            $name = "$($_.Name)".ToUpperInvariant()
            if ($name.Length -eq 1 -and $name -match '^[A-Z]$') {
                if (-not $found.Contains($name)) { [void]$found.Add($name) }
            }
        }
    } catch {}
    foreach ($code in 65..90) {  # A..Z
        $letter = [string][char]$code
        if ($found.Contains($letter)) { continue }
        if (Test-PathSafe "${letter}:\") {
            [void]$found.Add($letter)
        }
    }
    return @($found)
}

function Resolve-CondaExeFromRoot {
    param([string]$Root)
    if (-not $Root) { return $null }
    if (-not (Test-DriveReady $Root)) { return $null }
    foreach ($c in @(
        (Join-Path $Root "Scripts\conda.exe"),
        (Join-Path $Root "condabin\conda.exe"),
        (Join-Path $Root "Library\bin\conda.exe")
    )) {
        if (Test-PathSafe $c) { return $c }
    }
    return $null
}

function Use-CondaRoot {
    param([string]$Root)
    Ensure-CondaUtf8Env
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

function Get-CondaSearchRoots {
    # General Windows layout: profile/LocalAppData/ProgramData, then every
    # ready drive letter (C, D, E, F, ...) with common install folder names.
    $dirNames = @(
        "miniconda3", "Miniconda3",
        "anaconda3", "Anaconda3",
        "miniforge3", "Miniforge3",
        "mambaforge", "Mambaforge"
    )
    $list = New-Object System.Collections.Generic.List[string]

    foreach ($p in @($env:CONDA_ROOT, $env:CONDA_PREFIX)) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        if (Test-DriveReady $p) {
            [void]$list.Add($p.Trim())
        } else {
            Write-Host "[WARN] ignoring CONDA_* on missing/unreachable drive: $p"
        }
    }

    foreach ($base in @($env:USERPROFILE, $env:LOCALAPPDATA, $env:ProgramData)) {
        if ([string]::IsNullOrWhiteSpace($base)) { continue }
        if (-not (Test-DriveReady $base)) { continue }
        foreach ($name in $dirNames) {
            [void]$list.Add((Join-Path $base $name))
        }
    }

    foreach ($letter in (Get-ReadyDriveLetters)) {
        $driveRoot = "${letter}:\"
        foreach ($name in $dirNames) {
            [void]$list.Add((Join-Path $driveRoot $name))
        }
        [void]$list.Add((Join-Path $driveRoot "ProgramData\miniconda3"))
        [void]$list.Add((Join-Path $driveRoot "ProgramData\anaconda3"))
        [void]$list.Add((Join-Path $driveRoot "ProgramData\Miniforge3"))
        [void]$list.Add((Join-Path $driveRoot "tools\miniconda3"))
        [void]$list.Add((Join-Path $driveRoot "tools\anaconda3"))
    }

    $out = New-Object System.Collections.Generic.List[string]
    $seen = @{}
    foreach ($p in $list) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        $key = $p.Trim().ToLowerInvariant()
        if ($seen.ContainsKey($key)) { continue }
        if (-not (Test-DriveReady $p)) { continue }
        $seen[$key] = $true
        [void]$out.Add($p.Trim())
    }
    return @($out)
}

function Find-CondaRootOnDisk {
    # Walk search roots; return first root that has conda.exe or conda.bat.
    foreach ($root in (Get-CondaSearchRoots)) {
        $condaExe = Resolve-CondaExeFromRoot $root
        $condaBat = Join-Path $root "condabin\conda.bat"
        if ($condaExe -or (Test-PathSafe $condaBat)) {
            return $root
        }
        # Sometimes CONDA_PREFIX points at an env; try grandparent (install root).
        try {
            $parentRoot = Split-Path (Split-Path $root -Parent) -Parent
        } catch {
            $parentRoot = $null
        }
        if ($parentRoot -and (Test-DriveReady $parentRoot)) {
            $condaExe = Resolve-CondaExeFromRoot $parentRoot
            $condaBat = Join-Path $parentRoot "condabin\conda.bat"
            if ($condaExe -or (Test-PathSafe $condaBat)) {
                return $parentRoot
            }
        }
    }
    return $null
}

function New-MorphAgentLiteEnv {
    # Create morphagent_lite with python+pip only (defaults channels; no conda-forge mega-solve).
    param(
        [Parameter(Mandatory = $true)][string]$EnvName
    )
    if (-not $script:CondaExe) { throw "conda.exe not set" }

    Write-Host "[..] Creating $EnvName (python=3.10 + pip only; science stack via pip)"
    Clear-CondaClassicFallback
    Ensure-CondaUtf8Env

    & $script:CondaExe create -n $EnvName python=3.10 pip -y `
        -c defaults --override-channels
    if ($LASTEXITCODE -eq 0) { return }

    Write-Host "[WARN] conda create failed (exit $LASTEXITCODE); retry with classic fallback (still python+pip only)"
    Enable-CondaClassicFallback
    & $script:CondaExe create -n $EnvName python=3.10 pip -y `
        -c defaults --override-channels
    if ($LASTEXITCODE -ne 0) {
        throw "conda create failed ($LASTEXITCODE). Upgrade Miniconda (>=23.9 recommended) or check network/SSL, then re-run."
    }
}
