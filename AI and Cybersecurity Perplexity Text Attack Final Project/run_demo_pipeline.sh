#!/bin/bash
set -e

echo "=========================================================="
echo "      AI & CYBERSECURITY FINAL PROJECT: DEMO RUN"
echo "      Processing 50 samples (approx. 5 minutes)"
echo "=========================================================="

# 1. Setup Environment
echo "[1/5] Activating Virtual Environment..."
if [ -d "env" ]; then
    source env/bin/activate
else
    echo "Error: Virtual environment 'env' not found. Please run setup_env.sh first."
    exit 1
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export DEMO_MODE=1

# 2. Run Verification Tests
echo "[2/5] Verification: Running Unit Tests..."
pytest tests/test_perplexity.py

# 3. Run the Attack
echo "[3/5] Running Adversarial Attack (TextAttack)..."
python experiments/run_attack.py

# 4. Measure Perplexity
echo "[4/5] Measuring Perplexity (Clean vs Attacked)..."
python experiments/run_perplexity_on_clean.py
python experiments/run_perplexity_on_attacked.py

# 5. Execute Notebook
echo "[5/5] Updating Notebook Visualizations..."
jupyter nbconvert --to notebook --execute --inplace experiments/comparison_notebook.ipynb

echo "=========================================================="
echo "SUCCESS: Demo completed."
echo "  - Results saved in 'tests/data/measures/'"
echo "  - Charts updated in 'experiments/comparison_notebook.ipynb'"
echo "=========================================================="
