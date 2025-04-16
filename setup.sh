#!/bin/bash

echo "🚀 Setting up your Python environment..."

# Step 1: Create virtual environment
python3 -m venv venv
echo "✅ Virtual environment created in ./venv"

# Step 2: Activate it (for Unix/macOS users)
source venv/bin/activate
echo "✅ Virtual environment activated"

# Step 3: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed from requirements.txt"

echo "🎉 Setup complete. You're ready to go!"
