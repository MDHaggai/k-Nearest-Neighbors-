#!/bin/bash

# Banknote Authentication GUI Setup Script
# =========================================

echo "🏦 Setting up Banknote Authentication GUI..."
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Install required packages
echo ""
echo "📦 Installing required packages..."
pip install -r gui_requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install required packages"
    exit 1
fi

echo "✅ Packages installed successfully"

# Generate sample images
echo ""
echo "🖼️ Generating sample banknote images..."
python generate_sample_images.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate sample images"
    exit 1
fi

echo "✅ Sample images generated successfully"

# Launch the GUI application
echo ""
echo "🚀 Launching Banknote Authentication GUI..."
echo ""
python gui_banknote_classifier.py

echo ""
echo "👋 Thanks for using the Banknote Authentication System!"
