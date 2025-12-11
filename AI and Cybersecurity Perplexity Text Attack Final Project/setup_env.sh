#!/bin/bash
set -e

echo "1. Creating Virtual Environment..."
python3 -m venv env

echo "2. Activating Environment..."
source env/bin/activate

echo "3. Installing Dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "4. Setup Complete."
echo ""
echo "To start working, please run the following commands:"
echo "  source env/bin/activate"
echo "  export PYTHONPATH=\$PYTHONPATH:$(pwd)/src"
