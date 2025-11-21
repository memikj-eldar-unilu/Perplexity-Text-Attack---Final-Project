import uuid
import pandas as pd
from textattack.transformations import WordSwapEmbedding
from textattack.shared import AttackedText

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    Feature,
    FeatureType,
)
from a4s_eval.metrics.model_metrics.perplexity_metric import perplexity
from a4s_eval.service.functional_model import TabularClassificationModel
from tests.save_measures_utils import save_measures

def attack_and_recompute():
    """
    Loads a dataset, applies an adversarial transformation, and re-runs the
    perplexity metric on the transformed data, saving the new results.
    """
    # 1. Load the original dataset
    df = pd.read_parquet("tests/data/squad_date_val.parquet")

    original_texts = df['context'].tolist()

    # 2. Apply the adversarial transformation
    print("Applying adversarial transformation (WordSwapEmbedding)...")
    transformation = WordSwapEmbedding(max_candidates=10)
    attacked_texts = []
    for text in original_texts:
        attacked_text = AttackedText(text)
        transformed_texts = transformation(attacked_text)
        # Use the first transformed text if available, otherwise keep original
        if transformed_texts:
            attacked_texts.append(transformed_texts[0].text)
        else:
            attacked_texts.append(text)
    
    attacked_df = pd.DataFrame({'context': attacked_texts})
    print("Transformation complete.")

    # 3. Create A4S data structures for the attacked data
    features = [
        Feature(
            pid=uuid.uuid4(),
            name="context",
            feature_type=FeatureType.TEXT,
            min_value=None,
            max_value=None,
        )
    ]
    attacked_datashape = DataShape(features=features, target=None)
    attacked_dataset = Dataset(pid=uuid.uuid4(), shape=attacked_datashape, data=attacked_df)
    
    dummy_model = Model(pid=uuid.uuid4(), dataset=attacked_dataset)
    dummy_predict = lambda x: x
    dummy_functional_model = TabularClassificationModel(
        predict_class=dummy_predict, predict_proba=None
    )

    # 4. Run perplexity metric on the attacked data
    print("Running perplexity metric on attacked data...")
    measures = perplexity(
        attacked_datashape, dummy_model, attacked_dataset, dummy_functional_model
    )
    print("Metric calculation complete.")

    # 5. Save the new measures
    output_filename = "perplexity_attacked"
    save_measures(output_filename, measures)
    print(f"Results saved to 'tests/data/measures/{output_filename}.csv'")

if __name__ == "__main__":
    attack_and_recompute()
