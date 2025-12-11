[README.md](https://github.com/user-attachments/files/24093681/README.md)
# AI & Cybersecurity: Perplexity Text Attack Analysis

**University of Luxembourg**  
**Course:** AI & Cybersecurity  
**Project:** Adversarial Attack Analysis using Perplexity Metric

---

## Project Overview

This project evaluates how adversarial text attacks affect the quality and naturalness of text in NLP systems. We use perplexity as a measure of text fluency - the idea is that adversarial examples, while effective at fooling models, often produce unnatural text that can be detected through higher perplexity scores.

### Core Components
1.  **Perplexity Metric:** A general-purpose implementation using the pre-trained `distilgpt2` model. This metric can evaluate any text dataset - whether clean, machine-generated, or adversarially modified.
2.  **Attack Generation:** Uses TextAttack's `WordSwapEmbedding` transformation to create adversarial examples from the SQuAD validation set.
3.  **Comparative Analysis:** Measures perplexity on both clean and attacked text to quantify the quality degradation caused by adversarial modifications.

---

## Repository & Resources

* **Public Repository:** [https://github.com/memikj-eldar-unilu/Perplexity-Text-Attack---Final-Project](https://github.com/memikj-eldar-unilu/Perplexity-Text-Attack---Final-Project)
* **Private Project Repository:** 
    * *Note: Professor access has been granted to the private repository.*
* **Full Project Backup (OneDrive):** 
    * *Note: This link is provided as a backup. It contains the fully configured environment and all datasets pre-installed, should you prefer not to run the setup script.*

---

## Installation & Setup

### Prerequisites
*   **OS:** Linux, MacOS, or Windows with WSL
*   **Python:** 3.8 or higher

### Setup Instructions
Run the following command from the project root:

```bash
bash setup_env.sh
```

This script will create a python virtual environment (`env/`) and install the required libraries from `requirements.txt`.

---

## Automated Demo

For a one-click demonstration of the entire pipeline (tests, attack, measurement, visualization) on a small subset of data (50 rows), you can run:

```bash
bash run_demo_pipeline.sh
```

This script will:
1.  Activate the environment.
2.  Run unit tests to verify correctness.
3.  Generate a small adversarial dataset.
4.  Calculate perplexity scores.
5.  Update the visualization notebook (`experiments/comparison_notebook.ipynb`).

---

## How to Run the Experiments

**Important:** Before running any script, you must activate the environment and set the python path.

```bash
# 1. Activate Environment
source env/bin/activate

# 2. Set Python Path (Required for imports to work)
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Step 1: Generate Adversarial Examples
This script loads the clean SQuAD data and generates adversarial examples.

```bash
python experiments/run_attack.py
```

*   **Output:** The script saves the resulting adversarial dataset to `tests/data/squad_date_val_attacked.parquet`.
*   **Note:** If you run this script multiple times, it will overwrite the existing parquet file. By default, it processes the full dataset, which can take several hours. 
*   **Fast Demo Mode:** To verify functionality quickly (e.g., in < 5 mins), you can run the script with `DEMO_MODE=1`. This will limit execution to the first 50 rows:
    ```bash
    export DEMO_MODE=1 && python experiments/run_attack.py
    ```

### Step 2: Measure Baseline Perplexity (Clean Data)
This script calculates the perplexity scores for the original, unmodified text to establish a baseline.

```bash
python experiments/run_perplexity_on_clean.py
```

*   **Output:** Results are saved to `tests/data/measures/perplexity_data.csv`. This CSV file contains the raw text and its corresponding perplexity score.
*   **Data Handling:** Running this script again will overwrite the previous CSV file, ensuring results are fresh.
*   **Demo Mode:** Supports `DEMO_MODE=1` for quick verification.

### Step 3: Measure Attack Perplexity (Adversarial Data)
This script calculates the perplexity scores for the attacked text generated in Step 1.

```bash
python experiments/run_perplexity_on_attacked.py
```

*   **Output:** Results are saved to `tests/data/measures/perplexity_attacked.csv`.
*   **Data Handling:** Like the clean data script, this will overwrite the existing CSV file.
*   **Demo Mode:** Supports `DEMO_MODE=1` for quick verification.

### Step 4: Visualize Results
We use a Jupyter Notebook to visualize the difference in text quality.

```bash
jupyter notebook experiments/comparison_notebook.ipynb
```

*   **How it works:** The notebook reads the two CSV files generated in Steps 2 and 3 (`perplexity_data.csv` and `perplexity_attacked.csv`). It then plots overlapping histograms to show the shift in perplexity distribution.
*   **Persistence:** Since the data is saved to CSVs on disk, you can close and reopen the notebook without losing the underlying data (unless you re-run the experiment scripts).
*   **Full Report:** We also provide `experiments/comparison_notebook_FULL_REPORT.ipynb`. This notebook contains the pre-computed analysis and visualizations for the full 10,000+ sample dataset, allowing you to view the comprehensive results immediately without running the time-consuming full experiment.

---

## Project Structure

```text
.
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── setup_env.sh                        # Installation script
├── run_demo_pipeline.sh                # Automated one-click demo script (runs on subset)
├── experiments/                        # Scripts for the experiment pipeline
│   ├── run_attack.py                   # Generates adversarial data using TextAttack
│   ├── run_perplexity_on_clean.py      # Measures baseline perplexity on clean text
│   ├── run_perplexity_on_attacked.py   # Measures perplexity on attacked text
│   ├── comparison_notebook.ipynb       # Visualizes results (for demo/subset runs)
│   └── comparison_notebook_FULL_REPORT.ipynb # Visualizes results (for full dataset runs)
├── src/
│   └── a4s_eval/                       # Core package code
│       └── metrics/model_metrics/      # Contains the generic Perplexity metric implementation
└── tests/
    └── data/                           # Data storage
        ├── squad_date_val.parquet      # Original clean dataset (used as input)
        ├── squad_date_val_attacked.parquet # Attacked dataset (subset for DEMO_MODE)
        ├── squad_date_val_attacked_FULL.parquet # Attacked dataset (full 10k+ rows)
        └── measures/                   # Output folder for CSV results
            ├── perplexity_data.csv             # Perplexity scores for clean data (subset for DEMO_MODE)
            ├── perplexity_attacked.csv         # Perplexity scores for attacked data (subset for DEMO_MODE)
            ├── perplexity_data_FULL.csv        # Perplexity scores for clean data (full 10k+ rows)
            └── perplexity_attacked_FULL.csv    # Perplexity scores for attacked data (full 10k+ rows)
```

## Running Tests
To verify that the metric implementation works correctly (unit tests):

```bash
pytest tests/
```
