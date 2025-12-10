# AI & Cybersecurity: Perplexity Text Attack Analysis

**University of Luxembourg**  
**Course:** AI & Cybersecurity  
**Project:** Adversarial Attack Analysis using Perplexity Metric

---

## Project Overview

This project evaluates the robustness and quality of Natural Language Processing (NLP) models against adversarial attacks. Specifically, we analyze how adversarial text attacks affect the perplexity (fluency and naturalness) of text.

Our hypothesis is that adversarial examples, while successful at fooling a target model, often result in degraded text quality that can be detected via higher perplexity scores compared to legitimate text.

### Core Components
1.  **Metric Implementation:** We implemented a general-purpose Perplexity Metric using a pre-trained `distilgpt2` model. This metric is designed to be model-agnostic and can evaluate the fluency of *any* text dataset, whether it is clean human-written text, machine-generated text, or adversarially modified text.
2.  **Attack Generation:** We use `TextAttack` (specifically the `WordSwapEmbedding` recipe) to generate a dataset of adversarial examples from the SQuAD validation set.
3.  **Analysis:** We run our general perplexity metric on both the clean baseline data and the generated adversarial data to quantify the degradation caused by the attack.

---

## Repository Links

*   **Public Repository:** [https://github.com/memikj-eldar-unilu/Perplexity-Text-Attack---Final-Project](https://github.com/memikj-eldar-unilu/Perplexity-Text-Attack---Final-Project)
*   **Private Project Repository (Professor/Staff Access):** [https://github.com/memikj-eldar-unilu/Ai-and-Cybersecurity-Perplexity-Text-Attack-Final-Project-](https://github.com/memikj-eldar-unilu/Ai-and-Cybersecurity-Perplexity-Text-Attack-Final-Project-)
    *   *Note: The Professor has been added as a collaborator to this private repository.*

---

## Installation & Setup

We provide a script to create a virtual environment and install all dependencies (PyTorch, Transformers, TextAttack, etc.).

### Prerequisites
*   **OS:** Linux / MacOS (or WSL on Windows)
*   **Python:** 3.8+

### Setup
Run the following command from the root of this folder:

```bash
bash setup_env.sh
```

This script will create a python virtual environment (`env/`) and install the required libraries from `requirements.txt`.

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
*   **Note:** If you run this script multiple times, it will overwrite the existing parquet file. By default, it processes the full dataset, which can take several hours. To verify functionality quickly (e.g., in < 5 mins), you can edit `experiments/run_attack.py` to uncomment the line that limits the dataframe (e.g., `df = df.head(50)`).

### Step 2: Measure Baseline Perplexity (Clean Data)
This script calculates the perplexity scores for the original, unmodified text to establish a baseline.

```bash
python experiments/run_perplexity_on_clean.py
```

*   **Output:** Results are saved to `tests/data/measures/perplexity_data.csv`. This CSV file contains the raw text and its corresponding perplexity score.
*   **Data Handling:** Running this script again will overwrite the previous CSV file, ensuring results are fresh.

### Step 3: Measure Attack Perplexity (Adversarial Data)
This script calculates the perplexity scores for the attacked text generated in Step 1.

```bash
python experiments/run_perplexity_on_attacked.py
```

*   **Output:** Results are saved to `tests/data/measures/perplexity_attacked.csv`.
*   **Data Handling:** Like the clean data script, this will overwrite the existing CSV file.

### Step 4: Visualize Results
We use a Jupyter Notebook to visualize the difference in text quality.

```bash
jupyter notebook experiments/comparison_notebook.ipynb
```

*   **How it works:** The notebook reads the two CSV files generated in Steps 2 and 3 (`perplexity_data.csv` and `perplexity_attacked.csv`). It then plots overlapping histograms to show the shift in perplexity distribution.
*   **Persistence:** Since the data is saved to CSVs on disk, you can close and reopen the notebook without losing the underlying data (unless you re-run the experiment scripts).

---

## Project Structure

```text
.
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── setup_env.sh                        # Installation script
├── experiments/                        # Scripts for the experiment pipeline
│   ├── run_attack.py                   # Generates adversarial data using TextAttack
│   ├── run_perplexity_on_clean.py      # Measures baseline perplexity on clean text
│   ├── run_perplexity_on_attacked.py   # Measures perplexity on attacked text
│   └── comparison_notebook.ipynb       # Visualizes results from the CSVs
├── src/
│   └── a4s_eval/                       # Core package code
│       └── metrics/model_metrics/      # Contains the generic Perplexity metric implementation
└── tests/
    └── data/                           # Data storage
        ├── squad_date_val.parquet      # Original clean dataset
        ├── squad_date_val_attacked.parquet # Generated adversarial dataset
        └── measures/                   # Output folder for CSV results
```

## Running Tests
To verify that the metric implementation works correctly (unit tests):

```bash
pytest tests/
```