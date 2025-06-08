# PowerShell Installation Script for k-NN Project
# This script installs all required packages for the k-NN analysis project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "k-NN Project Setup for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Function to check if command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check if Python is installed
if (-not (Test-Command python)) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Python found:" -ForegroundColor Green
python --version

# Check if pip is available
if (-not (Test-Command pip)) {
    Write-Host "ERROR: pip is not available" -ForegroundColor Red
    Write-Host "Please ensure pip is installed with Python" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "pip found:" -ForegroundColor Green
pip --version

# Check if we're in a virtual environment
$InVenv = $env:VIRTUAL_ENV -or $env:CONDA_DEFAULT_ENV
if (-not $InVenv) {
    Write-Host ""
    Write-Host "WARNING: You're not in a virtual environment" -ForegroundColor Yellow
    Write-Host "It's recommended to create one first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Gray
    Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -notmatch "^[Yy]") {
        exit 0
    }
}

# Upgrade pip first
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Blue
try {
    python -m pip install --upgrade pip
    Write-Host "✓ pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "Warning: Failed to upgrade pip" -ForegroundColor Yellow
}

# Install packages from requirements.txt
Write-Host ""
Write-Host "Installing packages from requirements.txt..." -ForegroundColor Blue

try {
    pip install -r requirements.txt
    Write-Host "✓ Packages installed successfully" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Some packages failed to install" -ForegroundColor Red
    Write-Host "You may need to install them individually" -ForegroundColor Yellow
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Test import of key packages
Write-Host ""
Write-Host "Testing key package imports..." -ForegroundColor Blue

try {
    python -c "import numpy, pandas, sklearn, matplotlib, seaborn; print('✓ All core packages imported successfully')"
    Write-Host "✓ Package installation verified" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Some packages may not have installed correctly" -ForegroundColor Yellow
}

# Check if Jupyter is working
Write-Host ""
Write-Host "Checking Jupyter installation..." -ForegroundColor Blue
try {
    jupyter --version | Out-Null
    Write-Host "✓ Jupyter is ready" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run the analysis:" -ForegroundColor Green
    Write-Host "  jupyter lab notebooks/knn_analysis.ipynb" -ForegroundColor Gray
    Write-Host "  or" -ForegroundColor Gray
    Write-Host "  jupyter notebook notebooks/knn_analysis.ipynb" -ForegroundColor Gray
} catch {
    Write-Host "Warning: Jupyter may not be properly installed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete! Press Enter to exit..." -ForegroundColor Green
Read-Host
