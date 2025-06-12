@echo off
REM Banknote Authentication GUI Setup Script for Windows
REM ====================================================

echo 🏦 Setting up Banknote Authentication GUI...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Install required packages
echo.
echo 📦 Installing required packages...
pip install -r gui_requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install required packages
    pause
    exit /b 1
)

echo ✅ Packages installed successfully

REM Generate sample images
echo.
echo 🖼️ Generating sample banknote images...
python generate_sample_images.py

if errorlevel 1 (
    echo ❌ Failed to generate sample images
    pause
    exit /b 1
)

echo ✅ Sample images generated successfully

REM Launch the GUI application
echo.
echo 🚀 Launching Banknote Authentication GUI...
echo.
python gui_banknote_classifier.py

echo.
echo 👋 Thanks for using the Banknote Authentication System!
pause
