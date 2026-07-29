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

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found. Open this script from an Anaconda/Miniforge Prompt."
}

$KnownEnvs = (conda env list --json | ConvertFrom-Json).envs
$Exists = $KnownEnvs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName }
if (-not $Exists) {
    conda create -y -n $EnvName python=3.10 pip
}

conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install -r $Requirements
conda run -n $EnvName python -m pip install -e $Repository
$EnvFile = Join-Path $Repository ".env"
$EnvExample = Join-Path $Repository ".env.example"
if (-not (Test-Path $EnvFile)) {
    Copy-Item $EnvExample $EnvFile
}

if (-not $SkipAllen) {
    try {
        if (-not (Test-Path $AllenYml)) { throw "Allen environment.yml not found: $AllenYml" }
        if (-not (Test-Path $AllenReq)) { throw "Allen requirements not found: $AllenReq" }
        $AllenExists = $KnownEnvs | Where-Object { (Split-Path $_ -Leaf) -eq $AllenEnvName }
        if (-not $AllenExists) {
            Write-Host "Creating Allen segmentation environment: $AllenEnvName"
            conda env create -y -n $AllenEnvName -f $AllenYml
        } else {
            Write-Host "Using existing conda environment: $AllenEnvName"
        }
        Write-Host "Installing Allen scientific stack..."
        conda run -n $AllenEnvName python -m pip install --upgrade "pip<22" "setuptools<59" "wheel"
        conda run -n $AllenEnvName python -m pip install -r $AllenReq
        Write-Host "Installing vendored aicssegmentation..."
        conda run -n $AllenEnvName python -m pip install -e $AllenPkg --no-deps
        Write-Host "Verifying Allen installation..."
        conda run -n $AllenEnvName python $AllenCheck
        Write-Host "[OK] Allen environment $AllenEnvName is ready"
    } catch {
        Write-Warning "Allen environment setup failed: $($_.Exception.Message)"
        Write-Warning "MorphAgent UI install continues. Custom data without masks will skip auto-segmentation and still finish the run."
    }
} else {
    Write-Host "[INFO] Skipping Allen env (-SkipAllen)."
}

conda run -n $EnvName python $Verifier --ui-smoke

Write-Host "Installation verified. Run scripts\start_ui.ps1 to launch MorphAgent."
Write-Host "Allen segmentation env (optional): $AllenEnvName"
