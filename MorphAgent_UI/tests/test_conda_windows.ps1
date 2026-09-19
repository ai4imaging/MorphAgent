# Discovery tests for scripts/conda_windows.ps1, runnable on any OS.
#
# Windows path handling is simulated: Join-Path / Split-Path / Test-PathSafe and
# the environment probes are shadowed with backslash semantics and an in-memory
# file system, so the real Find-CondaRoot chain runs unchanged.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $ScriptDir "../scripts/conda_windows.ps1")

$script:Failures = 0
$script:Checks = 0

function Check {
    param([string]$Name, $Expected, $Actual)
    $script:Checks++
    $same = ("$Expected" -eq "$Actual")
    if ($same) {
        Write-Host "  ok   $Name"
    } else {
        $script:Failures++
        Write-Host "  FAIL $Name"
        Write-Host "       expected: '$Expected'"
        Write-Host "       actual:   '$Actual'"
    }
}

# --- simulated Windows file system -----------------------------------------

$script:FakePaths = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
$script:FakeCommands = @{}
$script:FakeRegistryRoots = @()
$script:FakeWildcardRoots = @()

function Add-FakePath {
    param([string]$Path)
    [void]$script:FakePaths.Add($Path.TrimEnd('\'))
}

function Add-FakeCondaInstall {
    # Mirrors a real layout: <root>\Scripts\conda.exe plus the condabin shim.
    param([string]$Root, [switch]$ShimOnly)
    Add-FakePath $Root
    Add-FakePath "$Root\condabin"
    Add-FakePath "$Root\condabin\conda.bat"
    if (-not $ShimOnly) {
        Add-FakePath "$Root\Scripts"
        Add-FakePath "$Root\Scripts\conda.exe"
    }
}

function Reset-Fake {
    $script:FakePaths.Clear()
    $script:FakeCommands = @{}
    $script:FakeRegistryRoots = @()
    $script:FakeWildcardRoots = @()
    $script:CondaExe = $null
    foreach ($name in @("MORPHAGENT_CONDA_ROOT", "CONDA_ROOT", "CONDA_EXE", "CONDA_PREFIX")) {
        Remove-Item "Env:\$name" -ErrorAction SilentlyContinue
    }
    $env:USERPROFILE = "C:\Users\me"
    Add-FakePath "C:\"
    Add-FakePath "D:\"
}

# --- shadows: Windows semantics on a POSIX host -----------------------------

function Join-Path {
    param([Parameter(Position = 0)][string]$Path, [Parameter(Position = 1)][string]$ChildPath)
    if ([string]::IsNullOrEmpty($Path)) { return $ChildPath }
    return ($Path.TrimEnd('\') + '\' + $ChildPath.TrimStart('\'))
}

function Split-Path {
    param(
        [Parameter(Position = 0)][string]$Path,
        [string]$LiteralPath,
        [switch]$Parent,
        [switch]$Leaf
    )
    $target = if ($LiteralPath) { $LiteralPath } else { $Path }
    if ([string]::IsNullOrWhiteSpace($target)) { return "" }
    $trimmed = $target.TrimEnd('\')
    $index = $trimmed.LastIndexOf('\')
    if ($Leaf) {
        if ($index -lt 0) { return $trimmed }
        return $trimmed.Substring($index + 1)
    }
    if ($index -lt 0) { return "" }
    # Parent of D:\Anaconda is the drive root, not "D:".
    if ($index -eq 2 -and $trimmed[1] -eq ':') { return $trimmed.Substring(0, 3) }
    return $trimmed.Substring(0, $index)
}

function Test-PathSafe {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    return $script:FakePaths.Contains($Path.TrimEnd('\'))
}

function Test-DriveReady {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    if ($Path.Length -ge 2 -and $Path[1] -eq ':') {
        return (Test-PathSafe $Path.Substring(0, 3))
    }
    return $true
}

function Get-ReadyDriveLetters {
    return @("C", "D")
}

function Get-Command {
    # ErrorAction is left to the common parameter; redeclaring it is an error.
    param(
        [Parameter(Position = 0)][string]$Name,
        $CommandType,
        [switch]$All
    )
    if ($script:FakeCommands.ContainsKey($Name)) {
        return @([pscustomobject]@{ Source = $script:FakeCommands[$Name] })
    }
    return @()
}

function Get-CondaRegistryRoots { return @($script:FakeRegistryRoots) }
function Get-CondaWildcardRoots { return @($script:FakeWildcardRoots) }

# --- cases ------------------------------------------------------------------

Write-Host "conda_windows.ps1 discovery"

# The reported failure: conda installed as D:\Anaconda rather than D:\Anaconda3.
Reset-Fake
Add-FakeCondaInstall "D:\Anaconda"
$script:FakeCommands["conda.exe"] = "D:\Anaconda\Scripts\conda.exe"
Check "custom install folder on PATH" "D:\Anaconda" (Find-CondaRoot)

# Same install, PATH not inherited: the folder-name scan has to cover it too,
# which is why the name list carries the suffix-less spellings.
Reset-Fake
Add-FakeCondaInstall "D:\Anaconda"
Check "custom install folder without PATH" "D:\Anaconda" (Find-CondaRoot)

# A folder name that matches nothing at all is reachable only through PATH.
Reset-Fake
Add-FakeCondaInstall "D:\Tools\MyPythonStack"
Add-FakePath "D:\Tools"
Check "unnameable install is not guessable" "" ((Get-CondaSearchRoots | Where-Object { Test-CondaRoot $_ }) -join ",")
$script:FakeCommands["conda.exe"] = "D:\Tools\MyPythonStack\Scripts\conda.exe"
Check "unnameable install via PATH" "D:\Tools\MyPythonStack" (Find-CondaRoot)

# conda-forge style installs often expose only the condabin shim.
Reset-Fake
Add-FakeCondaInstall "D:\Miniforge" -ShimOnly
$script:FakeCommands["conda.bat"] = "D:\Miniforge\condabin\conda.bat"
Check "shim-only install is found" "D:\Miniforge" (Find-CondaRoot)
Check "shim is used as the launcher" "D:\Miniforge\condabin\conda.bat" (Resolve-CondaLauncherFromRoot "D:\Miniforge")

# A conda-initialised shell exports CONDA_EXE even when PATH is unhelpful.
Reset-Fake
Add-FakeCondaInstall "D:\Anaconda"
$env:CONDA_EXE = "D:\Anaconda\Scripts\conda.exe"
Check "CONDA_EXE" "D:\Anaconda" (Find-CondaRoot)

# CONDA_PREFIX points at the active env, two levels below the install root.
Reset-Fake
Add-FakeCondaInstall "D:\Anaconda"
Add-FakePath "D:\Anaconda\envs\morphagent_lite"
$env:CONDA_PREFIX = "D:\Anaconda\envs\morphagent_lite"
Check "CONDA_PREFIX walks up to the root" "D:\Anaconda" (Find-CondaRoot)

# Documented escape hatch, and it must win over anything else on the machine.
Reset-Fake
Add-FakeCondaInstall "C:\Users\me\miniconda3"
Add-FakeCondaInstall "E:\Custom Tools\Anaconda"
Add-FakePath "E:\"
$script:FakeCommands["conda.exe"] = "C:\Users\me\miniconda3\Scripts\conda.exe"
$env:CONDA_ROOT = "E:\Custom Tools\Anaconda"
Check "CONDA_ROOT override wins" "E:\Custom Tools\Anaconda" (Find-CondaRoot)

# Explorer keeps a stale environment block, so a fresh install is visible only
# in the registry until the user logs out.
Reset-Fake
Add-FakeCondaInstall "D:\Anaconda"
$script:FakeRegistryRoots = @("D:\Anaconda\Scripts")
Check "registry PATH entry, nothing on PATH" "D:\Anaconda" (Find-CondaRoot)

# Last resort: a *conda* folder that is neither on PATH nor in the registry.
Reset-Fake
Add-FakeCondaInstall "D:\Anaconda"
$script:FakeWildcardRoots = @("D:\Anaconda")
Check "wildcard folder scan" "D:\Anaconda" (Find-CondaRoot)

# The layouts that already worked must keep working.
Reset-Fake
Add-FakeCondaInstall "C:\Users\me\miniconda3"
Check "standard profile install" "C:\Users\me\miniconda3" (Find-CondaRoot)
Check "backward-compatible alias" "C:\Users\me\miniconda3" (Find-CondaRootOnDisk)

Reset-Fake
Add-FakeCondaInstall "C:\ProgramData\Anaconda3"
$env:ProgramData = "C:\ProgramData"
Add-FakePath "C:\ProgramData"
Check "ProgramData install" "C:\ProgramData\Anaconda3" (Find-CondaRoot)

# A machine with no conda must report nothing rather than a half-usable root.
Reset-Fake
Check "no conda anywhere" "" (Find-CondaRoot)
Reset-Fake
Add-FakePath "D:\Anaconda"
Check "empty folder is not a conda root" "False" (Test-CondaRoot "D:\Anaconda")

# CONDA_ROOT pointing somewhere wrong must fall through, not abort discovery.
Reset-Fake
Add-FakeCondaInstall "C:\Users\me\miniconda3"
$env:CONDA_ROOT = "D:\gone"
Check "bad CONDA_ROOT falls through" "C:\Users\me\miniconda3" (Find-CondaRoot)

# Use-CondaRoot must publish both the launcher and the root for conda run.
Reset-Fake
Add-FakeCondaInstall "D:\Anaconda"
Check "Use-CondaRoot succeeds" "True" (Use-CondaRoot "D:\Anaconda")
Check "launcher is recorded" "D:\Anaconda\Scripts\conda.exe" $script:CondaExe
Check "CONDA_ROOT is exported" "D:\Anaconda" $env:CONDA_ROOT
Reset-Fake
Check "Use-CondaRoot rejects a non-root" "False" (Use-CondaRoot "D:\nothing")

Write-Host ""
if ($script:Failures -gt 0) {
    Write-Host "$script:Failures of $script:Checks checks failed"
    exit 1
}
Write-Host "all $script:Checks checks passed"
exit 0
