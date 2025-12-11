import pandas as pd
from textattack.transformations import WordSwapEmbedding
from textattack.shared import AttackedText
from tqdm import tqdm
import csv
import os
from pathlib import Path


def run_attack():
    """
    Loads a dataset, applies an adversarial transformation, saves the transformed
    dataset to a parquet file, and writes progress to a CSV file row-by-row.
    """
    # 1. Define file paths robustly
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    input_path = PROJECT_ROOT / "tests" / "data" / "squad_date_val.parquet"
    
    # ALWAYS write to regular files (no _FULL suffix)
    output_parquet_path = PROJECT_ROOT / "tests" / "data" / "squad_date_val_attacked.parquet"
    progress_csv_path = PROJECT_ROOT / "tests" / "data" / "measures" / "attack_progress.csv"
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Data file not found at {input_path}")
    
    # Ensure the output directory exists
    os.makedirs(output_parquet_path.parent, exist_ok=True)
    os.makedirs(progress_csv_path.parent, exist_ok=True)
    
    # 2. Load the original dataset
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # 3. Check for demo mode (limits samples but output path stays the same)
    demo_mode = os.environ.get("DEMO_MODE") == "1"
    
    if demo_mode:
        print(f"Note: DEMO MODE active - limiting to 50 of {len(df)} samples")
        df = df.head(50)
    else:
        print(f"Processing full dataset: {len(df)} samples")
    
    original_texts = df['context'].tolist()
    print(f"Starting attack on {len(original_texts)} samples...")
    
    # 4. Apply the adversarial transformation
    print("Applying adversarial transformation (WordSwapEmbedding)...")
    transformation = WordSwapEmbedding(max_candidates=10)
    attacked_texts = []
    
    # Open the progress CSV file for writing
    with open(progress_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['original_text', 'attacked_text'])
        
        # Process each text with a progress bar
        for text in tqdm(original_texts, desc="Attacking texts"):
            attacked_text_obj = AttackedText(text)
            transformed_texts = transformation(attacked_text_obj)
            
            if transformed_texts:
                result_text = transformed_texts[0].text
            else:
                result_text = text
            
            attacked_texts.append(result_text)
            csv_writer.writerow([text, result_text])
    
    # 5. Create a new dataframe with the attacked text, preserving other columns
    attacked_df = df.copy()
    attacked_df['context'] = attacked_texts
    print("Transformation complete.")
    
    # 6. Save the final attacked dataframe to a parquet file
    print(f"Saving attacked dataset to {output_parquet_path}...")
    attacked_df.to_parquet(output_parquet_path)
    print(f"Done. Attacked dataset saved to {output_parquet_path}")
    print(f"Progress log written to {progress_csv_path}")


if __name__ == "__main__":
    run_attack()
