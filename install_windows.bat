@echo off
REM Windows Installation Script for k-NN Project
REM This script installs all required packages for the k-NN analysis project

echo ========================================
echo k-NN Project Setup for Windows
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found:
python --version

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo pip found:
pip --version

REM Upgrade pip first
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install packages from requirements.txt
echo.
echo Installing packages from requirements.txt...
pip install -r requirements.txt

REM Check if installation was successful
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Some packages failed to install
    echo You may need to install them individually
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================

REM Optional: Create virtual environment reminder
echo.
echo REMINDER: It's recommended to use a virtual environment
echo To create one, run:
echo   python -m venv .venv
echo   .venv\Scripts\activate
echo   pip install -r requirements.txt
echo.

REM Test import of key packages
echo Testing key package imports...
python -c "import numpy, pandas, sklearn, matplotlib, seaborn; print('✓ All core packages imported successfully')"

if %errorlevel% neq 0 (
    echo WARNING: Some packages may not have installed correctly
) else (
    echo ✓ Package installation verified
)

echo.
echo You can now run:
echo   jupyter lab notebooks/knn_analysis.ipynb
echo.
pause
