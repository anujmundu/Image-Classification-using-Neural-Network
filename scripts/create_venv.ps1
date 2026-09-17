# PowerShell script to create the virtual environment 'IC' for this project
# ------------------------------------------------------------
# Usage (run from repository root):
#   powershell -ExecutionPolicy Bypass -File scripts\create_venv.ps1
# ------------------------------------------------------------

$venvName = "IC"
# Resolve the directory where this script lives and move to the repository root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path $scriptDir -Parent
Set-Location $projectRoot

Write-Host "🔧 Creating virtual environment '$venvName'..."
# Determine the global Python executable that is on the PATH (typically Anaconda's python)
$globalPython = (Get-Command python -ErrorAction Stop).Source

# Create the venv (let Python handle pip installation automatically)
& $globalPython -m venv $venvName

# Path to the venv's python interpreter
$venvPython = Join-Path $projectRoot "$venvName\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "❌ Unable to locate venv python at $venvPython"
    exit 1
}

# Ensure pip is up‑to‑date using the bundled ensurepip (no external download)
Write-Host "🔧 Bootstrapping pip inside the venv..."
& $venvPython -m ensurepip --upgrade

# Upgrade pip itself (still offline – will succeed if pip is already present)
Write-Host "🔧 Upgrading pip..."
& $venvPython -m pip install --upgrade pip

# Install project requirements if they exist
$requirementsPath = Join-Path $projectRoot "requirements.txt"
$apiReqPath = Join-Path $projectRoot "api\requirements.txt"
if (Test-Path $requirementsPath) {
    Write-Host "📦 Installing core dependencies..."
    & $venvPython -m pip install -r $requirementsPath
} else {
    Write-Warning "⚠️ Core requirements.txt not found – skipping core install."
}
if (Test-Path $apiReqPath) {
    Write-Host "📦 Installing API dependencies..."
    & $venvPython -m pip install -r $apiReqPath
} else {
    Write-Warning "⚠️ API requirements.txt not found – skipping API install."
}

Write-Host "✅ Virtual environment '$venvName' is ready."
Write-Host "To activate:"
Write-Host "  # PowerShell"
Write-Host "  . $venvName\\Scripts\\Activate.ps1"
Write-Host "  # CMD"
Write-Host "  $venvName\\Scripts\\activate.bat"
Write-Host "  # Git‑Bash / WSL"
Write-Host "  source $venvName/bin/activate"
