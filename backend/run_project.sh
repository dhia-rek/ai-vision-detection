#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting AI Vision Detection Setup..."

# Determine python command
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi
echo "Using $PYTHON_CMD"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# 2. Activate venv and install dependencies
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn ultralytics opencv-python python-multipart

echo "✅ Setup complete. Starting server..."
# 3. Run the application
python backend/app/main.py --reload
