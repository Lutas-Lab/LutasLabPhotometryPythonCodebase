$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$environmentPath = Join-Path $projectRoot ".venv-gui"
$environmentPython = Join-Path $environmentPath "Scripts\python.exe"
$uvVersion = "0.12.21"
$installerTemp = Join-Path $projectRoot ".gui-install-temp"
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
New-Item -ItemType Directory -Force -Path $installerTemp | Out-Null
$env:TEMP = $installerTemp
$env:TMP = $installerTemp

function Test-Python312 {
    param(
        [string]$Command,
        [string[]]$PrefixArguments = @()
    )

    try {
        & $Command @PrefixArguments -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Test-Pip {
    param([string]$PythonCommand)

    try {
        & $PythonCommand -m pip --version *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

$pythonCommand = $null
$pythonPrefixArguments = @()
$projectEnvironmentPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if ((Get-Command "py.exe" -ErrorAction SilentlyContinue) -and
    (Test-Python312 -Command "py.exe" -PrefixArguments @("-3.12"))) {
    $pythonCommand = "py.exe"
    $pythonPrefixArguments = @("-3.12")
}
elseif ((Test-Path -LiteralPath $projectEnvironmentPython) -and
    (Test-Python312 -Command $projectEnvironmentPython)) {
    $pythonCommand = $projectEnvironmentPython
}
elseif ((Get-Command "python.exe" -ErrorAction SilentlyContinue) -and
    (Test-Python312 -Command "python.exe")) {
    $pythonCommand = "python.exe"
}

if (-not $pythonCommand) {
    Write-Host "Python 3.12 was not found." -ForegroundColor Red
    Write-Host "Install a 64-bit Python 3.12.x release from https://www.python.org/downloads/windows/"
    Write-Host "During installation, enable 'Add python.exe to PATH', then run install_gui.bat again."
    exit 1
}

Write-Host "Using Python 3.12 to create an isolated GUI environment."
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $pythonCommand @pythonPrefixArguments -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Python could not create the .venv environment."
    }
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    throw "The environment was not created at $environmentPath."
}
if (-not (Test-Python312 -Command $environmentPython)) {
    Write-Host "The existing .venv-gui does not use Python 3.12." -ForegroundColor Red
    Write-Host "Rename or remove only the .venv-gui folder, then run install_gui.bat again."
    exit 1
}

if (-not (Test-Pip -PythonCommand $environmentPython)) {
    Write-Host "Finishing pip setup in the GUI environment..."
    & $environmentPython -m ensurepip --upgrade --default-pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip could not be initialized in .venv-gui. Rename or remove that folder and try again."
    }
}

Write-Host "Installing the pinned environment manager..."
& $environmentPython -m pip install --disable-pip-version-check "uv==$uvVersion"
if ($LASTEXITCODE -ne 0) {
    throw "uv $uvVersion could not be installed. Check the internet connection and try again."
}

Write-Host "Installing the locked conventional, FluoPulse, and iFLIP3 GUI dependencies..."
$previousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
$previousUvCache = $env:UV_CACHE_DIR
try {
    $env:UV_PROJECT_ENVIRONMENT = $environmentPath
    $env:UV_CACHE_DIR = Join-Path $environmentPath ".uv-cache"
    & $environmentPython -m uv sync `
        --directory $projectRoot `
        --frozen `
        --inexact `
        --all-packages `
        --extra gui `
        --extra forecasting `
        --extra events `
        --python $environmentPython
    if ($LASTEXITCODE -ne 0) {
        throw "The locked photometry GUI dependencies could not be installed."
    }
}
finally {
    if ($null -eq $previousProjectEnvironment) {
        Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    }
    else {
        $env:UV_PROJECT_ENVIRONMENT = $previousProjectEnvironment
    }
    if ($null -eq $previousUvCache) {
        Remove-Item Env:UV_CACHE_DIR -ErrorAction SilentlyContinue
    }
    else {
        $env:UV_CACHE_DIR = $previousUvCache
    }
}

Write-Host "Checking the installation..."
& $environmentPython -c "import pandas, sklearn, streamlit, lutaslab_photometry, fluopulse_analysis, iflip3; print('All GUI workflow dependencies imported successfully.')"
if ($LASTEXITCODE -ne 0) {
    throw "The installation completed, but its import check failed."
}

Write-Host "Loading the Streamlit application as a smoke test..."
$appPath = Join-Path $projectRoot "streamlit_app.py"
$smokeTest = @'
import sys
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(sys.argv[2]).run(timeout=30)
if app.exception:
    raise RuntimeError(str(app.exception))
print("Streamlit application loaded successfully.")
'@
$encodedSmokeTest = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($smokeTest))
& $environmentPython -c "import base64,sys;exec(base64.b64decode(sys.argv[1]))" $encodedSmokeTest $appPath
if ($LASTEXITCODE -ne 0) {
    throw "The dependencies installed, but the Streamlit application did not load successfully."
}

if ($null -eq $previousTemp) {
    Remove-Item Env:TEMP -ErrorAction SilentlyContinue
}
else {
    $env:TEMP = $previousTemp
}
if ($null -eq $previousTmp) {
    Remove-Item Env:TMP -ErrorAction SilentlyContinue
}
else {
    $env:TMP = $previousTmp
}
Remove-Item -LiteralPath $installerTemp -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "The GUI environment is ready at $environmentPath." -ForegroundColor Green
exit 0
