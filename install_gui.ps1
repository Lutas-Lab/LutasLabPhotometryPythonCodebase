$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$environmentPath = Join-Path $projectRoot ".venv"
$environmentPython = Join-Path $environmentPath "Scripts\python.exe"

function Test-Python312 {
    param(
        [string]$Command,
        [string[]]$PrefixArguments = @()
    )

    try {
        & $Command @PrefixArguments -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

$pythonCommand = $null
$pythonPrefixArguments = @()

if ((Get-Command "py.exe" -ErrorAction SilentlyContinue) -and
    (Test-Python312 -Command "py.exe" -PrefixArguments @("-3.12"))) {
    $pythonCommand = "py.exe"
    $pythonPrefixArguments = @("-3.12")
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

Write-Host "Using Python 3.12 to create an isolated environment."
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
    Write-Host "The existing .venv does not use Python 3.12." -ForegroundColor Red
    Write-Host "Rename or remove only the .venv folder, then run install_gui.bat again."
    exit 1
}

Write-Host "Updating pip..."
& $environmentPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip could not be updated."
}

Write-Host "Installing the photometry application and GUI dependencies..."
$guiTarget = "${projectRoot}[gui]"
& $environmentPython -m pip install -e $guiTarget
if ($LASTEXITCODE -ne 0) {
    throw "The photometry GUI dependencies could not be installed."
}

Write-Host "Checking the installation..."
& $environmentPython -c "import pandas, streamlit, src; print('GUI dependencies imported successfully.')"
if ($LASTEXITCODE -ne 0) {
    throw "The installation completed, but its import check failed."
}

Write-Host "The GUI environment is ready at $environmentPath." -ForegroundColor Green
exit 0
