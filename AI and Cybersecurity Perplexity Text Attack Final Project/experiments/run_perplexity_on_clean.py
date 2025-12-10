import pandas as pd
import torch
import csv
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_perplexity_on_clean_data():
    """
    Loads the ORIGINAL (clean) dataset, runs the perplexity metric, and saves 
    the measures to a CSV. This provides the 'Baseline' for comparison.
    """
    # 1. Define file paths
    input_path = "tests/data/squad_date_val.parquet" # <--- ORIGINAL DATA
    progress_csv_path = "tests/data/measures/perplexity_data.csv" # <--- BASELINE OUTPUT
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(progress_csv_path), exist_ok=True)

    # 2. Load the original dataset
    print(f"Loading CLEAN dataset from {input_path}...")
    df = pd.read_parquet(input_path)

    # --- PERFORMANCE NOTE ---
    # To verify this script quickly without processing the entire 10k+ dataset,
    # you can uncomment the line below to limit the number of samples.
    # df = df.head(50)
    # ------------------------

    texts = df['context'].tolist()
    print(f"Clean dataset loaded. Found {len(texts)} samples.")

    # 3. Initialize the reference model
    print("Initializing reference model (distilgpt2)...")
    ref_model_name = "distilgpt2"
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)

    # 4. Calculate perplexity
    print(f"Calculating baseline perplexity and writing to {progress_csv_path}...")
    with open(progress_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['text', 'score']) # Notebook expects 'score' column

        for text in tqdm(texts, desc="Calculating Baseline"):
            if not isinstance(text, str) or not text.strip():
                score = float("inf")
            else:
                try:
                    # Added truncation=True to prevent crashes on long texts
                    input_ids = ref_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
                    with torch.no_grad():
                        outputs = ref_model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    score = torch.exp(loss).item()
                except Exception as e:
                    score = -1.0 
                    print(f"Error: {e}")

            csv_writer.writerow([text, score])
    
    print("Baseline calculation complete.")

if __name__ == "__main__":
    run_perplexity_on_clean_data()