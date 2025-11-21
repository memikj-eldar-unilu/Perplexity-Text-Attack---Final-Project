import pandas as pd
from textattack.transformations import WordSwapEmbedding
from textattack.shared import AttackedText
from tqdm import tqdm
import csv
import os

def run_attack():
    """
    Loads a dataset, applies an adversarial transformation, saves the transformed
    dataset to a parquet file, and writes progress to a CSV file row-by-row.
    """
    # 1. Define file paths
    input_path = "tests/data/squad_date_val.parquet"
    output_parquet_path = "tests/data/squad_date_val_attacked.parquet"
    progress_csv_path = "tests/data/measures/attack_progress.csv"
    
    # Ensure the output directory for the progress file exists
    os.makedirs(os.path.dirname(progress_csv_path), exist_ok=True)

    # 2. Load the original dataset
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    original_texts = df['context'].tolist()

    # 3. Apply the adversarial transformation
    print("Applying adversarial transformation (WordSwapEmbedding)...")
    transformation = WordSwapEmbedding(max_candidates=10)
    attacked_texts = []

    # Open the progress CSV file for writing
    with open(progress_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
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
            
            # Write the current result to the progress CSV
            csv_writer.writerow([text, result_text])

    # 4. Create a new dataframe with the attacked text, preserving other columns
    attacked_df = df.copy()
    attacked_df['context'] = attacked_texts
    print("Transformation complete.")

    # 5. Save the final attacked dataframe to a parquet file
    print(f"Saving final attacked dataset to {output_parquet_path}...")
    attacked_df.to_parquet(output_parquet_path)
    print(f"Attacked dataset saved successfully to {output_parquet_path}")
    print(f"Live progress was written to {progress_csv_path}")

if __name__ == "__main__":
    run_attack()