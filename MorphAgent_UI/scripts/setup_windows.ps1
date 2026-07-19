param(
    [string]$EnvName = "morphagent"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$HandoffRoot = Split-Path -Parent $ScriptDir
$Repository = Join-Path $HandoffRoot "MorphAgent"
$Requirements = Join-Path $HandoffRoot "dependencies\requirements-demo-ui.txt"
$Verifier = Join-Path $HandoffRoot "scripts\verify_install.py"

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
conda run -n $EnvName python $Verifier --ui-smoke

Write-Host "Installation verified. Run scripts\start_ui.ps1 to launch MorphAgent."
