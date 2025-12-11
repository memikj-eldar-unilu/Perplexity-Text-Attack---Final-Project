import pandas as pd
import torch
import csv
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_perplexity_on_attacked_data():
    """
    Loads the attacked dataset, runs the perplexity metric, and saves the 
    measures to a CSV file.
    """
    # 1. Define file paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    input_path = PROJECT_ROOT / "tests" / "data" / "squad_date_val_attacked.parquet"
    output_csv_path = PROJECT_ROOT / "tests" / "data" / "measures" / "perplexity_attacked.csv"
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"CRITICAL ERROR: Attacked dataset not found at {input_path}\n"
            f"Please run 'python experiments/run_attack.py' first."
        )
    
    # Ensure the output directory exists
    os.makedirs(output_csv_path.parent, exist_ok=True)
    
    # 2. Load the attacked dataset
    print(f"Loading attacked dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Note: We don't check DEMO_MODE here because the attacked file
    # already has the correct size (50 or full) from run_attack.py
    
    texts = df['context'].tolist()
    print(f"Starting perplexity measurement on {len(texts)} samples...")
    
    # 3. Initialize the reference model
    print("Loading distilgpt2 model...")
    ref_model_name = "distilgpt2"
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
    ref_model.eval()
    
    # 4. Calculate perplexity
    print("Calculating perplexity scores...")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['text', 'score'])
        
        for text in tqdm(texts, desc="Processing texts"):
            if not isinstance(text, str) or not text.strip():
                score = float("inf")
            else:
                try:
                    input_ids = ref_tokenizer.encode(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    )
                    with torch.no_grad():
                        outputs = ref_model(input_ids, labels=input_ids)
                        loss = outputs.loss
                    score = torch.exp(loss).item()
                except Exception as e:
                    print(f"Warning: Error processing text (length {len(text)}): {e}")
                    score = float("inf")
            
            csv_writer.writerow([text, score])
    
    print(f"Done. Perplexity scores saved to {output_csv_path}")


if __name__ == "__main__":
    run_perplexity_on_attacked_data()
