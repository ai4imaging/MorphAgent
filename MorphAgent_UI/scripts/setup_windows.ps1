# MorphAgent UI setup (Windows only — macOS/Linux use setup.sh).
# Creates `morphagent` with Qt + scientific stack, then optional `morphagent_allen`.
#
# Qt (Windows): conda `pyqt=5` only. Pip PyQt5 from requirements-demo-ui.txt is
# filtered out on purpose — Windows is not meant to follow the Unix pip-Qt path.
#
# Prefer running from Anaconda Prompt / Miniforge Prompt:
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
param(
    [string]$EnvName = "morphagent",
    [string]$AllenEnvName = "morphagent_allen",
    [switch]$SkipAllen
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HandoffRoot = Split-Path -Parent $ScriptDir
$Repository = Join-Path $HandoffRoot "MorphAgent"
$Requirements = Join-Path $HandoffRoot "dependencies\requirements-demo-ui.txt"
$Verifier = Join-Path $HandoffRoot "scripts\verify_install.py"
$AllenYml = Join-Path $Repository "envs\environment_allen.yml"
if (-not (Test-Path $AllenYml)) {
    $AllenYml = Join-Path $HandoffRoot "dependencies\environment-allen-optional.yml"
}
$AllenReq = Join-Path $Repository "envs\requirements-allen.txt"
$AllenPkg = Join-Path $Repository "segmentation_allen"
$AllenCheck = Join-Path $AllenPkg "check_installation.py"

function Assert-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. Open this script from an Anaconda/Miniforge Prompt."
    }
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

function Invoke-CondaRun([string]$Name, [string[]]$PythonArgs) {
    & conda run --no-capture-output -n $Name @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "conda run -n $Name failed (exit $LASTEXITCODE): $($PythonArgs -join ' ')"
    }
}

function Install-PipRequirementsWithoutPyQt([string]$Name, [string]$ReqFile) {
    $filtered = Join-Path $env:TEMP ("morphagent-req-no-pyqt-" + [guid]::NewGuid().ToString() + ".txt")
    Get-Content $ReqFile | Where-Object { $_ -notmatch '^\s*PyQt5([=<>!].*)?\s*(#.*)?$' } | Set-Content -Path $filtered
    Write-Host "Installing pip packages from $(Split-Path $ReqFile -Leaf) (PyQt5 lines skipped — using conda pyqt)..."
    Invoke-CondaRun $Name @("python", "-m", "pip", "install", "--upgrade", "pip")
    Invoke-CondaRun $Name @("python", "-m", "pip", "install", "-r", $filtered)
    Remove-Item $filtered -ErrorAction SilentlyContinue
    Write-Host "Removing any pip-installed PyQt5 wheels that conflict with conda pyqt..."
    & conda run --no-capture-output -n $Name python -m pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip 2>$null
}

function Ensure-SingleQtStack([string]$Name) {
    Write-Host "Reaffirming single conda Qt stack in $Name..."
    & conda install -y -n $Name -c conda-forge "pyqt=5" "qtpy>=2.4"
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
    raise SystemExit('Pip PyQt5 Qt binaries still present; uninstall PyQt5/PyQt5-Qt5/PyQt5-sip and reinstall conda pyqt=5')
"@
    )
}

Assert-Command "conda"

if (-not (Test-Path $Requirements)) {
    throw "Missing requirements file: $Requirements"
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

$EnvFile = Join-Path $Repository ".env"
$EnvExample = Join-Path $Repository ".env.example"
if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile
    } else {
        Write-Warning ".env.example missing; skipped creating .env"
    }
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

Write-Host "Installation verified. Run scripts\start_ui.ps1 to launch MorphAgent."
Write-Host "Allen segmentation env (optional): $AllenEnvName"
