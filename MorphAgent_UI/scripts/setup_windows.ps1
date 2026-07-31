# MorphAgent UI setup (Windows only - macOS/Linux use setup.sh).
# Creates:
#   - morphagent          = Qt UI + agent
#   - morphagent_sandbox  = frozen scientific stack for extract() code
#   - optional morphagent_allen
#
# Qt (Windows): conda pyqt=5 only. Pip PyQt5 from requirements-demo-ui.txt is
# filtered out on purpose - Windows is not meant to follow the Unix pip-Qt path.
#
# IMPORTANT (Windows): never invoke conda.bat with specs containing "<" or ">".
# cmd.exe treats those as redirection ("The system cannot find the file specified"). Always use conda.exe.
# Also: old conda (4.14) rejects `conda run python -c` when the -c string has newlines.
# Use a temp .py file or PowerShell filtering instead.
#
# Simple path for non-dev machines: double-click scripts\setup_windows.bat
# (bypasses ExecutionPolicy, auto-finds conda, or silently installs Miniconda
# into %USERPROFILE%\miniconda3 when none is present). Advanced:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
# Opt out of auto Miniconda: set MORPHAGENT_SKIP_MINICONDA_INSTALL=1
param(
    [string]$EnvName = "morphagent",
    [string]$SandboxEnvName = "morphagent_sandbox",
    [string]$AllenEnvName = "morphagent_allen",
    [switch]$SkipAllen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HandoffRoot = Split-Path -Parent $ScriptDir
$Repository = Join-Path $HandoffRoot "MorphAgent"
$Requirements = Join-Path $HandoffRoot "dependencies\requirements-demo-ui.txt"
$SandboxRequirements = Join-Path $HandoffRoot "dependencies\requirements-sandbox.txt"
$Verifier = Join-Path $HandoffRoot "scripts\verify_install.py"
$AllenYml = Join-Path $Repository "envs\environment_allen.yml"
if (-not (Test-Path $AllenYml)) {
    $AllenYml = Join-Path $HandoffRoot "dependencies\environment-allen-optional.yml"
}
$AllenReq = Join-Path $Repository "envs\requirements-allen.txt"
$AllenPkg = Join-Path $Repository "segmentation_allen"
$AllenCheck = Join-Path $AllenPkg "check_installation.py"

# Script-scoped path to conda.exe (NOT conda.bat).
$script:CondaExe = $null
$script:SetupLogDir = Join-Path $HandoffRoot "logs"
$script:SetupLogPath = $null
$script:SetupStatusPath = Join-Path $script:SetupLogDir "setup_last_status.txt"

function Initialize-Utf8Console {
    # Reduce GBK/CP936 mojibake on Windows consoles (script body stays ASCII-only).
    try { cmd /c "chcp 65001 >nul" | Out-Null } catch {}
    try {
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [Console]::InputEncoding = $utf8
        [Console]::OutputEncoding = $utf8
        $global:OutputEncoding = $utf8
    } catch {}
    if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = "1" }
    if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }
    # Old conda (e.g. 4.14) otherwise prompts y/N after unexpected errors and hangs .bat runs.
    $env:CONDA_REPORT_ERRORS = "false"
    # Miniconda 24+ ships conda-anaconda-tos, which prompts interactively and
    # EOF-crashes non-interactive .bat / piped setup (repo.anaconda.com/pkgs/main).
    # Official workaround: CONDA_NO_PLUGINS=true (setup uses conda-forge anyway).
    $env:CONDA_NO_PLUGINS = "true"
}

function Accept-AnacondaTosBestEffort {
    # Optional: persist ToS acceptance for future interactive conda use.
    # Setup itself relies on CONDA_NO_PLUGINS=true; failures here are ignored.
    if (-not $script:CondaExe) { return }
    $prev = $env:CONDA_NO_PLUGINS
    Remove-Item Env:\CONDA_NO_PLUGINS -ErrorAction SilentlyContinue
    $channels = @(
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
        "https://repo.anaconda.com/pkgs/msys2"
    )
    foreach ($ch in $channels) {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $script:CondaExe tos accept --override-channels --channel $ch 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                & $script:CondaExe tos accept -c $ch 2>$null | Out-Null
            }
        } catch {
            # ignore missing tos subcommand / plugin
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }
    if ($null -ne $prev -and "$prev" -ne "") {
        $env:CONDA_NO_PLUGINS = $prev
    } else {
        $env:CONDA_NO_PLUGINS = "true"
    }
}

function Write-Utf8NoBomFile([string]$Path, [string]$Content) {
    # PowerShell 5.1 Set-Content defaults to UTF-16LE - never use it for .env / req files.
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Content, $utf8)
}

function Start-SetupTranscript {
    try {
        if (-not (Test-Path $script:SetupLogDir)) {
            New-Item -ItemType Directory -Path $script:SetupLogDir -Force | Out-Null
        }
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $script:SetupLogPath = Join-Path $script:SetupLogDir ("setup_windows_" + $stamp + ".log")
        Start-Transcript -Path $script:SetupLogPath -Force | Out-Null
        Write-Host "[OK] Setup log (kept after window closes): $($script:SetupLogPath)"
    } catch {
        Write-Warning "Could not start setup transcript: $($_.Exception.Message)"
        $script:SetupLogPath = $null
    }
}

function Stop-SetupTranscriptSafe {
    try { Stop-Transcript | Out-Null } catch {}
}

function Write-SetupStatusFile([int]$ExitCode, [string[]]$Lines) {
    try {
        if (-not (Test-Path $script:SetupLogDir)) {
            New-Item -ItemType Directory -Path $script:SetupLogDir -Force | Out-Null
        }
        $header = @(
            "MorphAgent UI Windows setup status"
            "time=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
            "exit_code=$ExitCode"
            "handoff=$HandoffRoot"
            "log=$($script:SetupLogPath)"
            "----"
        )
        Write-Utf8NoBomFile -Path $script:SetupStatusPath -Content (($header + $Lines) -join "`r`n")
        Write-Host "[OK] Status saved: $($script:SetupStatusPath)"
    } catch {
        Write-Warning "Could not write status file: $($_.Exception.Message)"
    }
}

function Resolve-EnvPythonExe([string]$Name) {
    # Best-effort locate python.exe for a named conda env (Windows layout).
    $candidates = @()
    if ($script:CondaExe) {
        $root = Split-Path (Split-Path $script:CondaExe -Parent) -Parent
        $candidates += (Join-Path $root "envs\$Name\python.exe")
    }
    foreach ($base in @(
        $env:CONDA_ROOT,
        $env:CONDA_BASE,
        (Join-Path $env:USERPROFILE "Miniconda3"),
        (Join-Path $env:USERPROFILE "miniconda3"),
        (Join-Path $env:USERPROFILE "Anaconda3"),
        (Join-Path $env:LOCALAPPDATA "Miniconda3")
    )) {
        if ($base) { $candidates += (Join-Path $base "envs\$Name\python.exe") }
    }
    foreach ($p in ($candidates | Select-Object -Unique)) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Write-InstallChecklist {
    param([int]$ExitCode)
    $lines = New-Object System.Collections.Generic.List[string]
    $uiPy = Resolve-EnvPythonExe $EnvName
    $sbPy = Resolve-EnvPythonExe $SandboxEnvName
    $uiOk = [bool]$uiPy
    $sbOk = [bool]$sbPy
    Write-Host ""
    Write-Host "============================================================"
    if ($ExitCode -eq 0 -and $uiOk -and $sbOk) {
        Write-Host " SETUP RESULT: PASS"
    } elseif ($ExitCode -eq 0) {
        Write-Host " SETUP RESULT: PARTIAL (script exit 0 but env python missing)"
        $ExitCode = 1
    } else {
        Write-Host " SETUP RESULT: FAIL (exit $ExitCode)"
    }
    Write-Host "============================================================"
    $uiLine = if ($uiOk) { "[OK] UI env $EnvName -> $uiPy" } else { "[MISSING] UI env $EnvName python.exe" }
    $sbLine = if ($sbOk) { "[OK] sandbox env $SandboxEnvName -> $sbPy" } else { "[MISSING] sandbox env $SandboxEnvName python.exe" }
    Write-Host $uiLine
    Write-Host $sbLine
    [void]$lines.Add($uiLine)
    [void]$lines.Add($sbLine)
    if ($script:SetupLogPath) {
        Write-Host "Full log: $($script:SetupLogPath)"
        [void]$lines.Add("log=$($script:SetupLogPath)")
    }
    Write-Host "============================================================"
    Write-SetupStatusFile -ExitCode $ExitCode -Lines $lines.ToArray()
    return $ExitCode
}

function Wait-Always {
    param([int]$ExitCode = 0)
    # Always keep the window open so the user can read PASS/FAIL + log path.
    # CI can set MORPHAGENT_NO_PAUSE=1 to skip.
    if ($env:MORPHAGENT_NO_PAUSE -eq "1") { return }
    Write-Host ""
    Write-Host "Press Enter to close this window (log is already saved)..."
    try { [void][Console]::ReadLine() } catch { Start-Sleep -Seconds 60 }
}

function Invoke-CondaPythonScript {
    # Old conda (4.x) rejects `conda run python -c` when the -c string contains newlines.
    # Always write a temp .py and run that file instead.
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$ScriptText
    )
    $py = Join-Path $env:TEMP ("morphagent-conda-run-" + [guid]::NewGuid().ToString() + ".py")
    try {
        Write-Utf8NoBomFile -Path $py -Content $ScriptText
        & $script:CondaExe run --no-capture-output -n $Name python $py
        if ($LASTEXITCODE -ne 0) {
            throw "conda run python script failed in env '$Name' (exit $LASTEXITCODE)"
        }
    } finally {
        Remove-Item $py -ErrorAction SilentlyContinue
    }
}

function Write-AsciiPipRequirements {
    # Write ASCII-only package lines as raw UTF-8/ASCII bytes (no BOM).
    # Never use Set-Content / Out-File (PowerShell 5.1 defaults to UTF-16LE and
    # breaks `pip install -r` on Windows with UnicodeDecodeError).
    param(
        [Parameter(Mandatory = $true)][string]$Src,
        [Parameter(Mandatory = $true)][string]$Dst,
        [switch]$SkipPyQt5
    )
    if (-not (Test-Path -LiteralPath $Src)) {
        throw "Missing requirements file: $Src"
    }
    $raw = [System.IO.File]::ReadAllBytes($Src)
    if ($raw.Length -ge 2 -and $raw[0] -eq 0xFF -and $raw[1] -eq 0xFE) {
        $text = [System.Text.Encoding]::Unicode.GetString($raw, 2, $raw.Length - 2)
    } elseif ($raw.Length -ge 2 -and $raw[0] -eq 0xFE -and $raw[1] -eq 0xFF) {
        $text = [System.Text.Encoding]::BigEndianUnicode.GetString($raw, 2, $raw.Length - 2)
    } elseif ($raw.Length -ge 3 -and $raw[0] -eq 0xEF -and $raw[1] -eq 0xBB -and $raw[2] -eq 0xBF) {
        $text = [System.Text.Encoding]::UTF8.GetString($raw, 3, $raw.Length - 3)
    } else {
        try {
            $enc = New-Object System.Text.UTF8Encoding $false, $true
            $text = $enc.GetString($raw)
        } catch {
            $text = [System.Text.Encoding]::Unicode.GetString($raw)
        }
    }

    $kept = New-Object System.Collections.Generic.List[string]
    foreach ($ln in ($text -split "`r?`n")) {
        $t = $ln.Trim()
        if ([string]::IsNullOrEmpty($t)) { continue }
        if ($t.StartsWith("#")) { continue }  # drop comments entirely
        if ($SkipPyQt5 -and ($t -match '^(?i)PyQt5([=<>!\.].*)?(\s+#.*)?$')) { continue }
        if ($t -notmatch '^[\x20-\x7E]+$') {
            throw "Non-ASCII requirement line in ${Src}: $t"
        }
        [void]$kept.Add($t)
    }
    if ($kept.Count -lt 1) {
        throw "No pip requirements left after filtering from $Src"
    }

    $payload = (($kept -join "`n") + "`n")
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($payload)
    [System.IO.File]::WriteAllBytes($Dst, $bytes)

    $verify = [System.IO.File]::ReadAllBytes($Dst)
    if ($verify.Length -ge 2 -and $verify[1] -eq 0x00) {
        throw "Temp requirements looks like UTF-16LE (null bytes); refusing to call pip: $Dst"
    }
    $roundtrip = [System.Text.Encoding]::UTF8.GetString($verify)
    if ($roundtrip -ne $payload) {
        throw "Temp requirements UTF-8 round-trip mismatch: $Dst"
    }
    Write-Host "[OK] wrote ASCII UTF-8 requirements ($($kept.Count) lines, $($verify.Length) bytes) -> $Dst"
}

function Write-FilteredUiRequirements([string]$Src, [string]$Dst) {
    Write-AsciiPipRequirements -Src $Src -Dst $Dst -SkipPyQt5
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

function Get-CondaSearchRoots {
    return @(
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
}

function Install-MinicondaSilent {
    # One-time silent Miniconda when the machine has no conda at all.
    # Set MORPHAGENT_SKIP_MINICONDA_INSTALL=1 to disable auto-install.
    if ($env:MORPHAGENT_SKIP_MINICONDA_INSTALL -eq "1") {
        throw "conda was not found, and MORPHAGENT_SKIP_MINICONDA_INSTALL=1 disables auto-install."
    }

    $prefix = Join-Path $env:USERPROFILE "miniconda3"
    if (Use-CondaRoot $prefix) { return }

    Write-Host "[INFO] conda not found. Installing Miniconda silently (one-time)..."
    Write-Host "       prefix: $prefix"
    Write-Host "       This downloads ~100MB and may take several minutes."

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
        throw @"
Failed to download Miniconda installer (network blocked?).
Install manually, then re-run setup_windows.bat:
  https://docs.conda.io/en/latest/miniconda.html
"@
    }

    Write-Host "  running silent installer..."
    # NSIS: /D=path must be last and must not be quoted.
    $argList = "/S /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /D=$prefix"
    $proc = Start-Process -FilePath $installer -ArgumentList $argList -Wait -PassThru
    $code = 0
    if ($null -ne $proc) { $code = [int]$proc.ExitCode }

    if (-not (Use-CondaRoot $prefix)) {
        throw @"
Miniconda silent install finished but conda.exe was not found under:
  $prefix
Installer exit code: $code
Install manually from https://docs.conda.io/en/latest/miniconda.html
then double-click scripts\setup_windows.bat again.
"@
    }

    Write-Host "[OK] Miniconda installed at $prefix"
    try {
        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    } catch {}
}

function Initialize-Conda {
    # Prefer conda.exe everywhere. conda.bat + specs with "<" break under cmd redirection.
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) {
        $src = $cmd.Source
        if ($src -like '*.exe') {
            $script:CondaExe = $src
        } else {
            $dir = Split-Path -Parent $src
            $root = Split-Path -Parent $dir
            $script:CondaExe = Resolve-CondaExeFromRoot $root
            if (-not $script:CondaExe) {
                $script:CondaExe = Resolve-CondaExeFromRoot (Split-Path -Parent $root)
            }
        }
        if ($script:CondaExe) {
            Write-Host "[OK] conda.exe: $($script:CondaExe)"
            return
        }
        Write-Host "[WARN] conda is on PATH as $src but conda.exe was not found nearby."
        throw "Could not resolve conda.exe next to $src. Reinstall Miniconda or add Scripts\conda.exe to PATH."
    }

    foreach ($root in (Get-CondaSearchRoots)) {
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

    # No existing install: download + silent Miniconda into %USERPROFILE%\miniconda3
    Install-MinicondaSilent
}

function Test-CondaEnv([string]$Name) {
    try {
        $jsonText = & $script:CondaExe env list --json 2>$null | Out-String
        $listed = $jsonText | ConvertFrom-Json
        if ($null -ne $listed.envs) {
            return [bool]($listed.envs | Where-Object { (Split-Path $_ -Leaf) -eq $Name })
        }
    } catch {
        # fall through
    }
    $lines = & $script:CondaExe env list 2>$null
    foreach ($line in $lines) {
        if (-not $line -or $line.StartsWith("#")) { continue }
        $token = ($line -split "\s+")[0]
        if ($token -eq $Name) { return $true }
    }
    return $false
}

function Remove-CondaEnvCompat([string]$Name) {
    # Older conda-env rejects -y (unrecognized arguments: -y). Try common forms.
    $attempts = @(
        { & $script:CondaExe env remove -y -n $Name },
        { & $script:CondaExe env remove --yes -n $Name },
        { "y`ny`ny" | & $script:CondaExe env remove -n $Name },
        { & $script:CondaExe remove -y -n $Name --all }
    )
    foreach ($attempt in $attempts) {
        Write-Host "  try: conda env remove variant for $Name"
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $attempt | Out-Null
            if (($LASTEXITCODE -eq 0) -and -not (Test-CondaEnv $Name)) { return }
        } catch {
            # try next
        } finally {
            $ErrorActionPreference = $prevEap
            $global:LASTEXITCODE = 0
        }
    }
    throw "Failed to remove conda env '$Name' with all remove variants"
}

function New-CondaEnvFromYamlCompat([string]$Name, [string]$Yml) {
    $env:CONDA_NO_PLUGINS = "true"
    # Older conda (e.g. 23.1): `conda env create -y` -> unrecognized arguments: -y
    $attempts = @(
        @{ Label = "env create -y -n -f"; Script = { & $script:CondaExe env create -y -n $Name -f $Yml } },
        @{ Label = "env create --yes -n -f"; Script = { & $script:CondaExe env create --yes -n $Name -f $Yml } },
        @{ Label = "env create -n -f (no yes)"; Script = { & $script:CondaExe env create -n $Name -f $Yml } },
        @{ Label = "yes | env create -n -f"; Script = { "y`ny`ny" | & $script:CondaExe env create -n $Name -f $Yml } },
        @{ Label = "create -y -n -f"; Script = { & $script:CondaExe create -y -n $Name -f $Yml } }
    )
    foreach ($attempt in $attempts) {
        Write-Host "  try: conda $($attempt.Label)"
        if (Test-CondaEnv $Name) {
            Write-Host "  cleaning partial env $Name before retry..."
            try { Remove-CondaEnvCompat $Name } catch { }
        }
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $attempt.Script
            if (($LASTEXITCODE -eq 0) -and (Test-CondaEnv $Name)) {
                Write-Host "[OK] created $Name via: $($attempt.Label)"
                return
            }
        } catch {
            # try next
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }
    $ver = (& $script:CondaExe --version 2>$null | Out-String).Trim()
    throw "All conda env create variants failed for '$Name' from $Yml (conda: $ver)"
}

function New-CondaEnvPythonCompat([string]$Name) {
    # Force plugin skip even if the parent shell cleared CONDA_NO_PLUGINS.
    $env:CONDA_NO_PLUGINS = "true"
    $attempts = @(
        { & $script:CondaExe create -y -n $Name -c conda-forge --override-channels "python=3.10" "pip" },
        { & $script:CondaExe create -y -n $Name -c conda-forge "python=3.10" "pip" },
        { & $script:CondaExe create --yes -n $Name -c conda-forge --override-channels "python=3.10" "pip" },
        { "a`na`na`ny`ny`ny" | & $script:CondaExe create -n $Name -c conda-forge "python=3.10" "pip" }
    )
    foreach ($attempt in $attempts) {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $attempt
            if (($LASTEXITCODE -eq 0) -and (Test-CondaEnv $Name)) { return }
        } catch {
            # try next
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }
    throw "Failed to create conda env '$Name' (python=3.10)"
}

function Install-CoreCondaPackages([string]$Name) {
    Write-Host "Installing core conda packages into $Name (PyQt5 / numpy / scipy / ...)..."
    $env:CONDA_NO_PLUGINS = "true"
    # Specs keep "<"/">="; calling conda.exe (not .bat) so cmd does not treat "<" as redirect.
    & $script:CondaExe install -y -n $Name -c conda-forge --override-channels `
        "python=3.10" `
        "pip>=24" `
        "setuptools" `
        "wheel" `
        "numpy>=1.26,<3" `
        "scipy>=1.11" `
        "pandas>=2.0" `
        "pyqt=5" `
        "pyqt5-sip" `
        "qtpy>=2.4" `
        "pillow>=10" `
        "matplotlib>=3.8" `
        "scikit-image>=0.22" `
        "scikit-learn>=1.3" `
        "tifffile>=2023" `
        "pyyaml>=6" `
        "tqdm" `
        "h5py" `
        "networkx" `
        "imageio" `
        "requests" `
        "lxml"
    if ($LASTEXITCODE -ne 0) {
        throw "conda install of core packages failed for env '$Name' (exit $LASTEXITCODE). If you still see 'The system cannot find the file specified', conda.bat was used; this script must call Scripts\conda.exe."
    }
}

function Install-SandboxCondaPackages([string]$Name) {
    Write-Host "Installing sandbox conda packages into $Name (numpy / scipy / scikit-image / ..., no Qt)..."
    $env:CONDA_NO_PLUGINS = "true"
    & $script:CondaExe install -y -n $Name -c conda-forge --override-channels `
        "python=3.10" `
        "pip>=24" `
        "setuptools" `
        "wheel" `
        "numpy>=1.26,<3" `
        "scipy>=1.11" `
        "pandas>=2.0" `
        "pillow>=10" `
        "matplotlib>=3.8" `
        "scikit-image>=0.22" `
        "scikit-learn>=1.3" `
        "tifffile>=2023" `
        "pyyaml>=6" `
        "tqdm" `
        "h5py" `
        "networkx" `
        "imageio" `
        "requests" `
        "lxml"
    if ($LASTEXITCODE -ne 0) {
        throw "conda install of sandbox packages failed for env '$Name' (exit $LASTEXITCODE)"
    }
}

function Invoke-CondaRun([string]$Name, [string[]]$PythonArgs) {
    & $script:CondaExe run --no-capture-output -n $Name @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "conda run -n $Name failed (exit $LASTEXITCODE): $($PythonArgs -join ' ')"
    }
}

function Repair-Pip([string]$Name) {
    Write-Host "Repairing pip in $Name via conda (no pip self-upgrade)..."
    $spOut = & $script:CondaExe run -n $Name python -c "import site; print(site.getsitepackages()[0])"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($spOut | Out-String))) {
        throw "Could not resolve site-packages for '$Name'"
    }
    if ($spOut -is [array]) {
        $sp = ($spOut | Where-Object { $_ -and $_.ToString().Trim() } | Select-Object -Last 1)
    } else {
        $sp = $spOut
    }
    $sp = "$sp".Trim()
    Write-Host "  cleaning pip leftovers under $sp"
    $pipDir = Join-Path $sp "pip"
    if (Test-Path $pipDir) { Remove-Item -Recurse -Force $pipDir }
    Get-ChildItem -Path $sp -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'pip-*.dist-info' -or $_.Name -like 'pip-*.egg-info' } |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
    & $script:CondaExe install -y -n $Name -c conda-forge --force-reinstall "pip>=24" "setuptools" "wheel"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to repair pip in '$Name'"
    }
    Invoke-CondaRun $Name @("python", "-m", "pip", "--version")
}

function Remove-PipPyQt5Qt5IfPresent([string]$Name) {
    # pip prints "WARNING: Skipping PyQt5-Qt5 as it is not installed" to stderr and may
    # exit non-zero. With $ErrorActionPreference=Stop, PowerShell treats that as fatal
    # and aborts setup before morphagent_sandbox is created. Always ignore failure here.
    Write-Host "Removing pip-bundled Qt runtime PyQt5-Qt5 if present (ignore if absent)..."
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & $script:CondaExe run --no-capture-output -n $Name python -m pip uninstall -y PyQt5-Qt5 2>&1
    } catch {
        # absent package / native stderr noise - safe to ignore
    } finally {
        $ErrorActionPreference = $prevEap
        $global:LASTEXITCODE = 0
    }
}

function Install-PipRequirementsWithoutPyQt([string]$Name, [string]$ReqFile) {
    # Build filtered requirements as UTF-8 (no BOM) in PowerShell - never UTF-16LE,
    # and never multiline `conda run python -c` (broken on conda 4.14).
    $filtered = Join-Path $env:TEMP ("morphagent-req-no-pyqt-" + [guid]::NewGuid().ToString() + ".txt")
    Write-Host "Writing UTF-8 pip requirements (PyQt5 lines skipped - using conda pyqt)..."
    Write-FilteredUiRequirements -Src $ReqFile -Dst $filtered

    Write-Host "Installing pip packages from $(Split-Path $ReqFile -Leaf) (PyQt5 skipped)..."
    $prevPyUtf8 = $env:PYTHONUTF8
    $prevPyIo = $env:PYTHONIOENCODING
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    try {
        Repair-Pip $Name
        Invoke-CondaRun $Name @("python", "-m", "pip", "install", "-r", $filtered)
    } finally {
        if ($null -eq $prevPyUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue } else { $env:PYTHONUTF8 = $prevPyUtf8 }
        if ($null -eq $prevPyIo) { Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue } else { $env:PYTHONIOENCODING = $prevPyIo }
        Remove-Item $filtered -ErrorAction SilentlyContinue
    }
    Remove-PipPyQt5Qt5IfPresent $Name
}

function Ensure-SingleQtStack([string]$Name) {
    Write-Host "Reaffirming single conda Qt stack in $Name..."
    & $script:CondaExe install -y -n $Name -c conda-forge --force-reinstall "pyqt=5" "pyqt5-sip" "qtpy>=2.4"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to reinstall conda pyqt for '$Name'"
    }
    Invoke-CondaPythonScript -Name $Name -ScriptText @'
import PyQt5
from PyQt5 import QtCore
from pathlib import Path
import qtpy, numpy
print("[OK] single Qt stack: PyQt5=%s, qtpy=%s, numpy=%s" % (QtCore.PYQT_VERSION_STR, qtpy.__version__, numpy.__version__))
pip_qt = Path(PyQt5.__file__).resolve().parent / "Qt5" / "lib"
if pip_qt.exists():
    raise SystemExit("Pip PyQt5 Qt binaries still present; uninstall PyQt5-Qt5 and reinstall conda pyqt=5")
'@
}

$scriptExit = 0
try {
    Initialize-Utf8Console
    Start-SetupTranscript
    Set-Location $HandoffRoot
    Write-Host "MorphAgent handoff root: $HandoffRoot"
    Write-Host "[OK] CONDA_NO_PLUGINS=$($env:CONDA_NO_PLUGINS) (avoids Anaconda ToS interactive prompt)"

    Initialize-Conda
    Accept-AnacondaTosBestEffort

    if (-not (Test-Path $Requirements)) {
        throw "Missing requirements file: $Requirements"
    }
    if (-not (Test-Path $SandboxRequirements)) {
        throw "Missing sandbox requirements file: $SandboxRequirements"
    }
    if (-not (Test-Path $Repository)) {
        throw "Missing MorphAgent repository folder: $Repository"
    }

    if (-not (Test-CondaEnv $EnvName)) {
        Write-Host "Creating conda environment: $EnvName"
        New-CondaEnvPythonCompat $EnvName
    } else {
        Write-Host "Using existing conda environment: $EnvName"
    }

    Install-CoreCondaPackages $EnvName
    Install-PipRequirementsWithoutPyQt $EnvName $Requirements
    Invoke-CondaRun $EnvName @("python", "-m", "pip", "install", "-e", $Repository)
    Ensure-SingleQtStack $EnvName

    if (-not (Test-CondaEnv $SandboxEnvName)) {
        Write-Host "Creating code-sandbox conda environment: $SandboxEnvName"
        New-CondaEnvPythonCompat $SandboxEnvName
    } else {
        Write-Host "Using existing conda environment: $SandboxEnvName"
    }
    Install-SandboxCondaPackages $SandboxEnvName
    Write-Host "Installing frozen sandbox pip stack from requirements-sandbox.txt..."
    Repair-Pip $SandboxEnvName
    $sandboxFiltered = Join-Path $env:TEMP ("morphagent-sandbox-req-" + [guid]::NewGuid().ToString("N") + ".txt")
    try {
        Write-AsciiPipRequirements -Src $SandboxRequirements -Dst $sandboxFiltered
        Invoke-CondaRun $SandboxEnvName @("python", "-m", "pip", "install", "-r", $sandboxFiltered)
    } finally {
        if (Test-Path -LiteralPath $sandboxFiltered) {
            Remove-Item -LiteralPath $sandboxFiltered -Force -ErrorAction SilentlyContinue
        }
    }
    Invoke-CondaRun $SandboxEnvName @(
        "python", "-c",
        "import numpy, skimage, cv2, mahotas; print('[OK] sandbox numpy', numpy.__version__, 'skimage', skimage.__version__)"
    )

    $EnvFile = Join-Path $Repository ".env"
    $EnvExample = Join-Path $Repository ".env.example"
    if (-not (Test-Path $EnvFile)) {
        if (Test-Path $EnvExample) {
            Copy-Item $EnvExample $EnvFile
        } else {
            Write-Warning ".env.example missing; skipped creating .env"
        }
    }
    if (Test-Path $EnvFile) {
        try {
            $envText = [System.IO.File]::ReadAllText($EnvFile, [System.Text.UTF8Encoding]::new($false))
        } catch {
            $envText = Get-Content -Raw -Path $EnvFile
        }
        if ($envText -match '(?m)^CONDA_ENV=') {
            $envText = [regex]::Replace($envText, '(?m)^CONDA_ENV=.*$', 'CONDA_ENV=morphagent_sandbox')
        } else {
            if (-not $envText.EndsWith("`n")) { $envText += "`n" }
            $envText += "CONDA_ENV=morphagent_sandbox`n"
        }
        Write-Utf8NoBomFile -Path $EnvFile -Content $envText
    }

    if (-not $SkipAllen) {
        try {
            if (-not (Test-Path $AllenYml)) { throw "Allen environment.yml not found: $AllenYml" }
            if (-not (Test-Path $AllenReq)) { throw "Allen requirements not found: $AllenReq" }
            if (-not (Test-Path $AllenPkg)) { throw "Allen package not found: $AllenPkg" }

            if (-not (Test-CondaEnv $AllenEnvName)) {
                Write-Host "Creating Allen segmentation environment: $AllenEnvName (win-64 / Python 3.6)"
                $ver = (& $script:CondaExe --version 2>$null | Out-String).Trim()
                Write-Host "conda: $ver"
                New-CondaEnvFromYamlCompat -Name $AllenEnvName -Yml $AllenYml
            } else {
                Write-Host "Using existing conda environment: $AllenEnvName"
            }

            Write-Host "Installing Allen scientific stack..."
            # pip<22 MUST go through conda.exe (same "<" redirection trap via conda.bat).
            & $script:CondaExe run --no-capture-output -n $AllenEnvName python -m pip install --upgrade "pip<22" "setuptools<59" "wheel"
            if ($LASTEXITCODE -ne 0) { throw "Allen pip bootstrap failed" }
            # Old pip on py3.6 decodes -r with the locale codec (GBK on Chinese Windows).
            # Always feed an ASCII-only temp requirements file.
            $allenFiltered = Join-Path $env:TEMP ("morphagent-allen-req-" + [guid]::NewGuid().ToString("N") + ".txt")
            try {
                Write-AsciiPipRequirements -Src $AllenReq -Dst $allenFiltered
                & $script:CondaExe run --no-capture-output -n $AllenEnvName python -m pip install -r $allenFiltered
                if ($LASTEXITCODE -ne 0) { throw "Allen requirements install failed" }
            } finally {
                if (Test-Path -LiteralPath $allenFiltered) {
                    Remove-Item -LiteralPath $allenFiltered -Force -ErrorAction SilentlyContinue
                }
            }
            Write-Host "Installing vendored aicssegmentation..."
            & $script:CondaExe run --no-capture-output -n $AllenEnvName python -m pip install -e $AllenPkg --no-deps
            if ($LASTEXITCODE -ne 0) { throw "aicssegmentation editable install failed" }
            Write-Host "Verifying Allen installation..."
            & $script:CondaExe run --no-capture-output -n $AllenEnvName python $AllenCheck
            if ($LASTEXITCODE -ne 0) { throw "Allen check_installation.py failed" }
            Write-Host "[OK] Allen environment $AllenEnvName is ready"
        } catch {
            Write-Warning "Allen environment setup failed: $($_.Exception.Message)"
            Write-Warning "MorphAgent UI install continues. Custom data without masks will skip auto-segmentation and still finish the run."
        }
    } else {
        Write-Host "[INFO] Skipping Allen env (-SkipAllen)."
    }

    Write-Host "Running install verification (UI smoke)..."
    # Do not force QT_QPA_PLATFORM=offscreen here: conda pyqt on Windows often
    # lacks qoffscreen.dll and the process aborts. verify_install.py chooses a
    # working platform (and sets QT_PLUGIN_PATH when needed).
    $prevQt = $env:QT_QPA_PLATFORM
    if ($env:QT_QPA_PLATFORM -eq "offscreen") {
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }
    try {
        & $script:CondaExe run --no-capture-output -n $EnvName python $Verifier --ui-smoke
        if ($LASTEXITCODE -ne 0) {
            throw "verify_install.py failed (exit $LASTEXITCODE)"
        }
    } finally {
        if ($null -eq $prevQt) {
            Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
        } else {
            $env:QT_QPA_PLATFORM = $prevQt
        }
    }

    Write-Host "Installation verified. Double-click scripts\start_ui_windows.bat to launch MorphAgent."
    Write-Host "UI / agent env: $EnvName"
    Write-Host "Code sandbox env (feature extract): $SandboxEnvName"
    Write-Host "Allen segmentation env (optional): $AllenEnvName"
} catch {
    $scriptExit = 1
    Write-Host ""
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    if ($_.ScriptStackTrace) {
        Write-Host $_.ScriptStackTrace -ForegroundColor DarkRed
    }
} finally {
    $scriptExit = Write-InstallChecklist -ExitCode $scriptExit
    Stop-SetupTranscriptSafe
    # When launched via setup_windows.bat, MORPHAGENT_NO_PAUSE=1 and the .bat pauses.
    # When the .ps1 is run directly, always pause so the window does not flash-close.
    Wait-Always -ExitCode $scriptExit
}

exit $scriptExit
