# Shared Windows conda helpers for setup_windows.ps1 / start_ui_windows.ps1.
# ASCII-only (Windows PowerShell 5.1 / GBK consoles).
# Dot-source from the caller so $script:CondaExe stays in the caller's scope.
#
# Lite policy: never force classic+conda-forge mega-solves (that crashes old conda.exe).
# Prefer: accept Anaconda ToS -> keep plugins/libmamba -> conda create python+pip only -> pip rest.
#
# Find-CondaRoot tries, in order, until one hits:
#   1. CONDA_ROOT / MORPHAGENT_CONDA_ROOT override, then CONDA_EXE, then CONDA_PREFIX
#   2. conda.exe / conda.bat on PATH
#   3. the registry: installer records, uninstall entries, and the stored PATH
#      (Explorer keeps a stale environment block until logoff, so a fresh install
#       is often only visible there)
#   4. common install folder names on every ready drive
#   5. any *conda* / *forge* / *mamba* folder in the usual parent directories,
#      which is what finds non-standard names such as D:\Anaconda

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

function Resolve-CondaLauncherFromRoot {
    # conda.exe when present, else the condabin\conda.bat shim: some installs
    # (and most conda-forge based ones) only ship the shim on PATH.
    param([string]$Root)
    $exe = Resolve-CondaExeFromRoot $Root
    if ($exe) { return $exe }
    if (-not $Root) { return $null }
    if (-not (Test-DriveReady $Root)) { return $null }
    $bat = Join-Path $Root "condabin\conda.bat"
    if (Test-PathSafe $bat) { return $bat }
    return $null
}

function Test-CondaRoot {
    param([string]$Root)
    return [bool](Resolve-CondaLauncherFromRoot $Root)
}

function Resolve-CondaRootFromLocation {
    # Walk up from anything conda-ish (an exe, Scripts\, condabin\, Library\bin\,
    # or an env under envs\) to the install root that owns it.
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $null }
    $current = $Path.Trim().Trim('"')
    if ($current -match '(?i)\.(exe|bat|cmd)$') {
        try { $current = Split-Path -LiteralPath $current -Parent } catch { return $null }
    }
    for ($depth = 0; $depth -lt 4; $depth++) {
        if ([string]::IsNullOrWhiteSpace($current)) { break }
        if (Test-CondaRoot $current) { return $current }
        $parent = $null
        try { $parent = Split-Path -LiteralPath $current -Parent } catch { $parent = $null }
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $current) { break }
        $current = $parent
    }
    return $null
}

function Use-CondaRoot {
    param([string]$Root)
    Ensure-CondaUtf8Env
    $launcher = Resolve-CondaLauncherFromRoot $Root
    if (-not $launcher) { return $false }
    $prepend = @(
        (Join-Path $Root "condabin"),
        (Join-Path $Root "Scripts"),
        (Join-Path $Root "Library\bin"),
        $Root
    ) -join ";"
    $env:Path = "$prepend;$env:Path"
    $env:CONDA_ROOT = $Root
    $script:CondaExe = $launcher
    Write-Host "[OK] conda: $($script:CondaExe)"
    return $true
}

function Find-CondaRootFromEnvironment {
    foreach ($override in @($env:MORPHAGENT_CONDA_ROOT, $env:CONDA_ROOT)) {
        if ([string]::IsNullOrWhiteSpace($override)) { continue }
        $root = Resolve-CondaRootFromLocation $override
        if ($root) { return $root }
        Write-Host "[WARN] CONDA_ROOT is set but holds no conda install: $override"
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE)) {
        $root = Resolve-CondaRootFromLocation $env:CONDA_EXE
        if ($root) { return $root }
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        # An active env lives at <root>\envs\<name>, so the walk needs both levels.
        $root = Resolve-CondaRootFromLocation $env:CONDA_PREFIX
        if ($root) { return $root }
    }
    return $null
}

function Find-CondaRootOnPath {
    foreach ($name in @("conda.exe", "conda.bat")) {
        $commands = @()
        try {
            $commands = @(Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue)
        } catch {}
        foreach ($command in $commands) {
            $root = Resolve-CondaRootFromLocation $command.Source
            if ($root) { return $root }
        }
    }
    return $null
}

function Get-CondaRegistryRoots {
    # Installer records plus the stored PATH. Reading the stored PATH matters
    # because a double-clicked .bat inherits Explorer's environment block, which
    # is not refreshed until logoff after installing conda.
    $roots = New-Object System.Collections.Generic.List[string]

    foreach ($key in @(
        "HKCU:\SOFTWARE\Python\ContinuumAnalytics",
        "HKLM:\SOFTWARE\Python\ContinuumAnalytics",
        "HKLM:\SOFTWARE\WOW6432Node\Python\ContinuumAnalytics"
    )) {
        try {
            if (-not (Test-Path -LiteralPath $key)) { continue }
            foreach ($child in (Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue)) {
                try {
                    $install = Get-Item -LiteralPath (Join-Path $child.PSPath "InstallPath") -ErrorAction SilentlyContinue
                    if (-not $install) { continue }
                    $value = "$($install.GetValue(''))"
                    if (-not [string]::IsNullOrWhiteSpace($value)) { [void]$roots.Add($value) }
                } catch {}
            }
        } catch {}
    }

    foreach ($key in @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    )) {
        try {
            if (-not (Test-Path -LiteralPath $key)) { continue }
            foreach ($entry in (Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue)) {
                try {
                    $props = Get-ItemProperty -LiteralPath $entry.PSPath -ErrorAction SilentlyContinue
                    if (-not $props) { continue }
                    if ("$($props.DisplayName)" -notmatch '(?i)(conda|forge|mamba)') { continue }
                    $location = "$($props.InstallLocation)"
                    if (-not [string]::IsNullOrWhiteSpace($location)) { [void]$roots.Add($location) }
                } catch {}
            }
        } catch {}
    }

    foreach ($key in @("HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment", "HKCU:\Environment")) {
        try {
            $props = Get-ItemProperty -LiteralPath $key -Name "Path" -ErrorAction SilentlyContinue
            if (-not $props) { continue }
            foreach ($entry in ("$($props.Path)" -split ";")) {
                if ([string]::IsNullOrWhiteSpace($entry)) { continue }
                if ($entry -notmatch '(?i)(conda|forge|mamba)') { continue }
                [void]$roots.Add([System.Environment]::ExpandEnvironmentVariables($entry.Trim()))
            }
        } catch {}
    }

    return @($roots)
}

function Get-CondaWildcardRoots {
    # Non-standard install folder names (D:\Anaconda, C:\tools\conda, ...) are
    # only reachable by looking at what is actually there.
    $containers = New-Object System.Collections.Generic.List[string]
    foreach ($base in @($env:USERPROFILE, $env:LOCALAPPDATA, $env:ProgramData, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ([string]::IsNullOrWhiteSpace($base)) { continue }
        [void]$containers.Add($base)
        [void]$containers.Add((Join-Path $base "Programs"))
    }
    foreach ($letter in (Get-ReadyDriveLetters)) {
        $driveRoot = "${letter}:\"
        [void]$containers.Add($driveRoot)
        foreach ($name in @("tools", "ProgramData", "Program Files", "opt", "apps")) {
            [void]$containers.Add((Join-Path $driveRoot $name))
        }
    }

    $found = New-Object System.Collections.Generic.List[string]
    foreach ($container in $containers) {
        if (-not (Test-DriveReady $container)) { continue }
        if (-not (Test-PathSafe $container)) { continue }
        try {
            foreach ($dir in (Get-ChildItem -LiteralPath $container -Directory -Force -ErrorAction SilentlyContinue)) {
                if ($dir.Name -match '(?i)(conda|forge|mamba)') { [void]$found.Add($dir.FullName) }
            }
        } catch {}
    }
    return @($found)
}

function Get-CondaSearchRoots {
    # General Windows layout: profile/LocalAppData/ProgramData/Program Files, then
    # every ready drive letter (C, D, E, F, ...) with common install folder names,
    # then whatever *conda* folders actually exist in those places.
    $dirNames = @(
        "miniconda3", "miniconda",
        "anaconda3", "anaconda",
        "miniforge3", "miniforge",
        "mambaforge", "micromamba",
        "conda"
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

    foreach ($base in @($env:USERPROFILE, $env:LOCALAPPDATA, $env:ProgramData, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ([string]::IsNullOrWhiteSpace($base)) { continue }
        if (-not (Test-DriveReady $base)) { continue }
        foreach ($name in $dirNames) {
            [void]$list.Add((Join-Path $base $name))
            [void]$list.Add((Join-Path $base (Join-Path "Programs" $name)))
        }
    }

    foreach ($letter in (Get-ReadyDriveLetters)) {
        $driveRoot = "${letter}:\"
        foreach ($name in $dirNames) {
            [void]$list.Add((Join-Path $driveRoot $name))
            [void]$list.Add((Join-Path $driveRoot (Join-Path "ProgramData" $name)))
            [void]$list.Add((Join-Path $driveRoot (Join-Path "tools" $name)))
            [void]$list.Add((Join-Path $driveRoot (Join-Path "Program Files" $name)))
        }
    }

    foreach ($p in (Get-CondaWildcardRoots)) { [void]$list.Add($p) }

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

function Find-CondaRoot {
    $root = Find-CondaRootFromEnvironment
    if ($root) { return $root }

    $root = Find-CondaRootOnPath
    if ($root) { return $root }

    foreach ($candidate in (Get-CondaRegistryRoots)) {
        $resolved = Resolve-CondaRootFromLocation $candidate
        if ($resolved) { return $resolved }
    }

    foreach ($candidate in (Get-CondaSearchRoots)) {
        if (Test-CondaRoot $candidate) { return $candidate }
        # CONDA_PREFIX can point at an env; the install root is two levels up.
        try {
            $parentRoot = Split-Path (Split-Path $candidate -Parent) -Parent
        } catch {
            $parentRoot = $null
        }
        if ($parentRoot -and (Test-CondaRoot $parentRoot)) { return $parentRoot }
    }

    return $null
}

function Find-CondaRootOnDisk {
    # Kept for callers written before PATH/registry discovery existed.
    return Find-CondaRoot
}

function Write-CondaNotFoundHelp {
    Write-Host "[ERROR] No conda installation found."
    Write-Host "        Checked CONDA_ROOT / CONDA_EXE / CONDA_PREFIX, conda on PATH, the"
    Write-Host "        registry, and the usual install folders on every drive."
    Write-Host "        Install Miniconda, or point setup at the install you already have:"
    Write-Host '          set "CONDA_ROOT=D:\Anaconda"   (use your own folder)'
    Write-Host "        then re-run this script from that same window."
    Write-Host "        'where conda' in an Anaconda Prompt prints the folder to use."
}

function New-MorphAgentLiteEnv {
    # Create morphagent_lite with python+pip only (defaults channels; no conda-forge mega-solve).
    param(
        [Parameter(Mandatory = $true)][string]$EnvName
    )
    if (-not $script:CondaExe) { throw "conda launcher not set" }

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
