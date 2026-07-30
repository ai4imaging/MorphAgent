# MorphAgent UI setup (Windows only — macOS/Linux use setup.sh).
# Creates:
#   - `morphagent`          — Qt UI + agent
#   - `morphagent_sandbox`  — frozen scientific stack for extract() code
#   - optional `morphagent_allen`
#
# Qt (Windows): conda `pyqt=5` only. Pip PyQt5 from requirements-demo-ui.txt is
# filtered out on purpose — Windows is not meant to follow the Unix pip-Qt path.
#
# Silky path for non-dev machines: double-click scripts\setup_windows.bat
# (bypasses ExecutionPolicy and auto-finds conda). Advanced:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
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

function Wait-IfInteractive {
    param([int]$ExitCode = 0)
    if ($env:MORPHAGENT_NO_PAUSE -eq "1") { return }
    # Keep the window readable when launched via Explorer "Run with PowerShell".
    try {
        $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction SilentlyContinue).ParentProcessId
        $parentName = if ($parent) {
            (Get-CimInstance Win32_Process -Filter "ProcessId=$parent" -ErrorAction SilentlyContinue).Name
        } else { "" }
    } catch {
        $parentName = ""
    }
    $fromExplorer = $parentName -match '(?i)explorer\.exe'
    if ($fromExplorer -or $ExitCode -ne 0) {
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
conda was not found on this machine.

English:
  1. Install Miniconda or Miniforge (one-time):
       https://docs.conda.io/en/latest/miniconda.html
       https://github.com/conda-forge/miniforge
  2. Close this window, then double-click scripts\setup_windows.bat again.
     (Do not use Explorer 'Run with PowerShell' — Windows blocks .ps1 by default.)

中文:
  1. 请先安装 Miniconda 或 Miniforge（只需一次）。
  2. 关闭窗口后，再双击 scripts\setup_windows.bat。
     不要用资源管理器的「使用 PowerShell 运行」——系统默认会拦截 .ps1 脚本。
"@
}

function Test-CondaEnv([string]$Name) {
    # Prefer JSON; fall back to plain `conda env list` (warnings can break JSON on some Windows setups).
    try {
        $jsonText = & conda env list --json 2>$null | Out-String
        $listed = $jsonText | ConvertFrom-Json
        if ($null -ne $listed.envs) {
            return [bool]($listed.envs | Where-Object { (Split-Path $_ -Leaf) -eq $Name })
        }
    } catch {
        # fall through
    }
    $lines = & conda env list 2>$null
    foreach ($line in $lines) {
        if (-not $line -or $line.StartsWith("#")) { continue }
        $token = ($line -split "\s+")[0]
        if ($token -eq $Name) { return $true }
    }
    return $false
}

function Install-CoreCondaPackages([string]$Name) {
    Write-Host "Installing core conda packages into $Name (PyQt5 / numpy / scipy / ...)..."
    # conda-forge PyQt wheels are far more reliable on Windows than pip-only PyQt5.
    & conda install -y -n $Name -c conda-forge `
        "python=3.10" `
        "pip>=24" `
        "numpy>=1.26,<3" `
        "scipy>=1.11" `
        "pandas>=2.0" `
        "pyqt=5" `
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
        throw "conda install of core packages failed for env '$Name' (exit $LASTEXITCODE)"
    }
}

function Install-SandboxCondaPackages([string]$Name) {
    Write-Host "Installing sandbox conda packages into $Name (numpy / scipy / scikit-image / ..., no Qt)..."
    & conda install -y -n $Name -c conda-forge `
        "python=3.10" `
        "pip>=24" `
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
    & conda run --no-capture-output -n $Name @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "conda run -n $Name failed (exit $LASTEXITCODE): $($PythonArgs -join ' ')"
    }
}

function Repair-Pip([string]$Name) {
    Write-Host "Repairing pip in $Name via conda (no pip self-upgrade)..."
    $sp = & conda run -n $Name python -c "import site; print(site.getsitepackages()[0])"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($sp)) {
        throw "Could not resolve site-packages for '$Name'"
    }
    Write-Host "  cleaning pip leftovers under $sp"
    $pipDir = Join-Path $sp "pip"
    if (Test-Path $pipDir) { Remove-Item -Recurse -Force $pipDir }
    Get-ChildItem -Path $sp -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'pip-*.dist-info' -or $_.Name -like 'pip-*.egg-info' } |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
    & conda install -y -n $Name -c conda-forge --force-reinstall "pip>=24" "setuptools" "wheel"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to repair pip in '$Name'"
    }
    Invoke-CondaRun $Name @("python", "-m", "pip", "--version")
}

function Install-PipRequirementsWithoutPyQt([string]$Name, [string]$ReqFile) {
    # PowerShell 5.1 Set-Content / Out-File often write UTF-16LE; pip requires UTF-8.
    # Always build the filtered requirements file with Python (UTF-8, no BOM).
    $filtered = Join-Path $env:TEMP ("morphagent-req-no-pyqt-" + [guid]::NewGuid().ToString() + ".txt")
    $env:MORPHAGENT_REQ_SRC = $ReqFile
    $env:MORPHAGENT_REQ_DST = $filtered
    Write-Host "Writing UTF-8 pip requirements (PyQt5 lines skipped — using conda pyqt)..."
    Invoke-CondaRun $Name @(
        "python", "-c",
        @"
from pathlib import Path
import os, re
src = Path(os.environ['MORPHAGENT_REQ_SRC'])
dst = Path(os.environ['MORPHAGENT_REQ_DST'])
pat = re.compile(r'^\s*PyQt5([=<>!\.].*)?\s*(#.*)?$', re.I)
lines = [ln for ln in src.read_text(encoding='utf-8').splitlines() if not pat.match(ln)]
dst.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('[OK] wrote', dst, 'lines=', len(lines), 'encoding=utf-8')
"@
    )
    Remove-Item Env:MORPHAGENT_REQ_SRC -ErrorAction SilentlyContinue
    Remove-Item Env:MORPHAGENT_REQ_DST -ErrorAction SilentlyContinue

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
    # Only drop pip-bundled Qt runtime; Ensure-SingleQtStack reinstalls conda pyqt.
    Write-Host "Removing pip-bundled Qt runtime PyQt5-Qt5 if present..."
    & conda run --no-capture-output -n $Name python -m pip uninstall -y PyQt5-Qt5 2>$null
}


function Ensure-SingleQtStack([string]$Name) {
    Write-Host "Reaffirming single conda Qt stack in $Name..."
    & conda install -y -n $Name -c conda-forge --force-reinstall "pyqt=5" "pyqt5-sip" "qtpy>=2.4"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to reinstall conda pyqt for '$Name'"
    }
    Invoke-CondaRun $Name @(
        "python", "-c",
        @"
import PyQt5
from PyQt5 import QtCore
from pathlib import Path
import qtpy, numpy
print('[OK] single Qt stack: PyQt5=%s, qtpy=%s, numpy=%s' % (QtCore.PYQT_VERSION_STR, qtpy.__version__, numpy.__version__))
pip_qt = Path(PyQt5.__file__).resolve().parent / 'Qt5' / 'lib'
if pip_qt.exists():
    raise SystemExit('Pip PyQt5 Qt binaries still present; uninstall PyQt5-Qt5 and reinstall conda pyqt=5')
"@
    )
}

$scriptExit = 0
try {
    Set-Location $HandoffRoot
    Write-Host "MorphAgent handoff root: $HandoffRoot"

    Initialize-Conda

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
        & conda create -y -n $EnvName -c conda-forge python=3.10 pip
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create conda env '$EnvName'"
        }
    } else {
        Write-Host "Using existing conda environment: $EnvName"
    }

    Install-CoreCondaPackages $EnvName
    Install-PipRequirementsWithoutPyQt $EnvName $Requirements
    Invoke-CondaRun $EnvName @("python", "-m", "pip", "install", "-e", $Repository)
    Ensure-SingleQtStack $EnvName

    if (-not (Test-CondaEnv $SandboxEnvName)) {
        Write-Host "Creating code-sandbox conda environment: $SandboxEnvName"
        & conda create -y -n $SandboxEnvName -c conda-forge python=3.10 pip
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create conda env '$SandboxEnvName'"
        }
    } else {
        Write-Host "Using existing conda environment: $SandboxEnvName"
    }
    Install-SandboxCondaPackages $SandboxEnvName
    Write-Host "Installing frozen sandbox pip stack from requirements-sandbox.txt..."
    Repair-Pip $SandboxEnvName
    Invoke-CondaRun $SandboxEnvName @("python", "-m", "pip", "install", "-r", $SandboxRequirements)
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
        $envText = Get-Content -Raw -Path $EnvFile
        if ($envText -match '(?m)^CONDA_ENV=') {
            $envText = [regex]::Replace($envText, '(?m)^CONDA_ENV=.*$', 'CONDA_ENV=morphagent_sandbox')
        } else {
            if (-not $envText.EndsWith("`n")) { $envText += "`n" }
            $envText += "CONDA_ENV=morphagent_sandbox`n"
        }
        Set-Content -Path $EnvFile -Value $envText -NoNewline
    }

    if (-not $SkipAllen) {
        try {
            if (-not (Test-Path $AllenYml)) { throw "Allen environment.yml not found: $AllenYml" }
            if (-not (Test-Path $AllenReq)) { throw "Allen requirements not found: $AllenReq" }
            if (-not (Test-Path $AllenPkg)) { throw "Allen package not found: $AllenPkg" }

            # Windows uses native win-64 Python 3.6 builds (no CONDA_SUBDIR / Rosetta).
            if (-not (Test-CondaEnv $AllenEnvName)) {
                Write-Host "Creating Allen segmentation environment: $AllenEnvName (win-64 / Python 3.6)"
                & conda env create -y -n $AllenEnvName -f $AllenYml
                if ($LASTEXITCODE -ne 0) {
                    throw "conda env create failed for '$AllenEnvName'"
                }
            } else {
                Write-Host "Using existing conda environment: $AllenEnvName"
            }

            Write-Host "Installing Allen scientific stack..."
            & conda run --no-capture-output -n $AllenEnvName python -m pip install --upgrade "pip<22" "setuptools<59" "wheel"
            if ($LASTEXITCODE -ne 0) { throw "Allen pip bootstrap failed" }
            & conda run --no-capture-output -n $AllenEnvName python -m pip install -r $AllenReq
            if ($LASTEXITCODE -ne 0) { throw "Allen requirements install failed" }
            Write-Host "Installing vendored aicssegmentation..."
            & conda run --no-capture-output -n $AllenEnvName python -m pip install -e $AllenPkg --no-deps
            if ($LASTEXITCODE -ne 0) { throw "aicssegmentation editable install failed" }
            Write-Host "Verifying Allen installation..."
            & conda run --no-capture-output -n $AllenEnvName python $AllenCheck
            if ($LASTEXITCODE -ne 0) { throw "Allen check_installation.py failed" }
            Write-Host "[OK] Allen environment $AllenEnvName is ready"
        } catch {
            Write-Warning "Allen environment setup failed: $($_.Exception.Message)"
            Write-Warning "MorphAgent UI install continues. Custom data without masks will skip auto-segmentation and still finish the run."
        }
    } else {
        Write-Host "[INFO] Skipping Allen env (-SkipAllen)."
    }

    Write-Host "Running install verification (UI smoke, offscreen)..."
    $prevQt = $env:QT_QPA_PLATFORM
    $env:QT_QPA_PLATFORM = "offscreen"
    try {
        & conda run --no-capture-output -n $EnvName python $Verifier --ui-smoke
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
    Wait-IfInteractive -ExitCode $scriptExit
}

exit $scriptExit
