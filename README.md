#  Perplexity Metric Implementation and Testing

This document explains the perplexity metric we implemented, the files we created, and how to run the code.

## 1. The Perplexity Metric

**What it is:**
Perplexity measures how "surprised" a language model is when it sees a piece of text. Think of it as a confusion score.

*   A **low** perplexity score means the model was not surprised and predicted the text well.
*   A **high** score means the model was very surprised and did not expect that text.

**Assumptions:**
*   **Data Type:** This metric only works on text data.
*   **Model:** It uses a pre-trained language model called `distilgpt2` as a reference to judge the text. It does not evaluate the project's classification model, but rather the text data itself.

## 2. What Each File Does

Here are the files we added and what they are for:

*   `a4s_eval/metrics/model_metrics/perplexity_metric.py`: This is the main file. It contains the Python code that calculates the perplexity score for a given text.

*   `tests/metrics/model_metrics/test_perplexity.py`: This is a simple unit test to make sure the `perplexity_metric.py` code works correctly and doesn't crash with bad data.

*   `run_attack.py`: This script takes the original dataset (`squad_date_val.parquet`) and uses a simple "adversarial attack" to change some of the words. It saves the modified data as `squad_date_val_attacked.parquet`.

*   `run_perplexity_on_attacked.py`: This script calculates the perplexity score for both the original and the attacked datasets and saves the results into `.csv` files.

*   `comparison_notebook.ipynb`: This is a Jupyter Notebook that you can use to see the results. It loads the scores from the `.csv` files and shows a chart comparing the perplexity of the original text versus the attacked text.

## 3. Workflow Diagram

This chart shows how the files and scripts work together.

**Step A: The Attack**
The `run_attack.py` script reads the original data and creates a new, "attacked" version of it.

    [squad_date_val.parquet]
             |
             | run_attack.py
             v
    [squad_date_val_attacked.parquet]


**Step B: Perplexity Calculation**
The `run_perplexity_on_attacked.py` script reads **both** the original and the attacked data files and calculates the perplexity for each one, creating two separate result files.

    [squad_date_val.parquet] -------->+
                                     |
                                     | run_perplexity_on_attacked.py
                                     |
    [squad_date_val_attacked.parquet] ->+
                                     |
                                     +--> [perplexity_data.csv]
                                     |
                                     +--> [perplexity_attacked.csv]


**Step C: Visualization**
The Jupyter Notebook reads the two result files to create the comparison chart.

    [perplexity_data.csv] ------>+
                                 |
                                 | comparison_notebook.ipynb
                                 |
    [perplexity_attacked.csv] -->+
                                 |
                                 v
                          (Chart Visualization)


---
**Unit Test (Separate Process)**
The unit test directly checks the `perplexity_metric.py` code.

    [perplexity_metric.py] <--- pytest tests/metrics/model_metrics/test_perplexity.py



## 4. How to Run Everything

To see the whole process run from start to finish, follow these steps in your terminal from the `a4s/a4s-eval/` directory.

**Step 1: Install Dependencies**
Make sure you have installed the project's dependencies. You will also need `tqdm`, `transformers`, and `textattack`.
```bash
pip install tqdm transformers textattack
```

**Step 2: Run the Attack**
This will create the `squad_date_val_attacked.parquet` file. It will take some time to run.
```bash
python run_attack.py
```

**Step 3: Run the Perplexity Calculation**
This will calculate the perplexity for the attacked file. It will also take some time.
```bash
python run_perplexity_on_attacked.py
```

**Step 4: See the Results**
Open the `comparison_notebook.ipynb` in Jupyter. Run the cells in the notebook to see a histogram comparing the perplexity scores before and after the attack.

**How to Run the Unit Test:**
If you want to just run the basic test for the metric, use `pytest`.
```bash
pytest tests/metrics/model_metrics/test_perplexity.py
```
This will confirm that the metric code itself is working as expected.
