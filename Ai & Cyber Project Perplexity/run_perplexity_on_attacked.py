import uuid
import pandas as pd
import torch
import csv
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_perplexity_on_attacked_data():
    """
    Loads the attacked dataset, runs the perplexity metric, and saves the 
    measures incrementally to a CSV for progress tracking.
    """
    # 1. Define file paths
    input_path = "tests/data/squad_date_val_attacked.parquet"
    progress_csv_path = "tests/data/measures/perplexity_progress.csv"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(progress_csv_path), exist_ok=True)

    # 2. Load the attacked dataset
    print(f"Loading attacked dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    texts = df['context'].tolist()
    print("Attacked dataset loaded.")

    # 3. Initialize the reference model and tokenizer for perplexity calculation
    print("Initializing reference model (distilgpt2) for perplexity...")
    ref_model_name = "distilgpt2"
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
    print("Model initialized.")

    # 4. Open the progress CSV file and calculate perplexity row-by-row
    print(f"Calculating perplexity and writing progress to {progress_csv_path}...")
    with open(progress_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['text', 'perplexity_score'])

        for text in tqdm(texts, desc="Calculating perplexity"):
            if not isinstance(text, str) or not text.strip():
                score = float("inf")
            else:
                try:
                    input_ids = ref_tokenizer.encode(text, return_tensors="pt")
                    with torch.no_grad():
                        outputs = ref_model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    score = torch.exp(loss).item()
                except Exception as e:
                    # Catch potential errors with very long texts or other issues
                    score = -1.0 
                    print(f"Error processing text: {text[:100]}... Error: {e}")

            csv_writer.writerow([text, score])
    
    print("Perplexity calculation and progress logging complete.")

if __name__ == "__main__":
    run_perplexity_on_attacked_data()