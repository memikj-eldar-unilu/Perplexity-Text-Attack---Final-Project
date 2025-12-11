from datetime import datetime
import torch

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel

# Perplexity is a metric that is calculated with a reference model.
# As such, we are importing the modules required to do so.
from transformers import AutoModelForCausalLM, AutoTokenizer


@model_metric(name="perplexity")
def perplexity(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> list[Measure]:
    """
    Calculates the perplexity of text samples in a dataset.

    Perplexity is a measure of how well a probability model predicts a sample. In NLP,
    it's used to evaluate language models. A lower perplexity score indicates that the
    language model is better at predicting the text sample.

    This metric uses a pre-trained language model (distilgpt2) as a reference to calculate
    the perplexity of each text sample in the input dataset. It does not use the
    `functional_model` argument, as it is evaluating the dataset content itself,
    not the model's predictions.
    """
    measures = []

    # This metric is only applicable to text features.
    # We will check the feature type and return an empty list if it's not text.
    if not datashape.features or datashape.features[0].feature_type != "text":
        # Or log a warning: get_logger().warning("Perplexity metric is not applicable to non-text features.")
        return []

    text_column = datashape.features[0].name

    if text_column not in dataset.data.columns:
        raise ValueError(
            f"Text column '{text_column}' not found in the dataset."
        )

    # Initialize the reference model and tokenizer for perplexity calculation.
    # We use 'distilgpt2' as it is smaller and faster than 'gpt2', while providing
    # a reliable perplexity measure.
    ref_model_name = "distilgpt2"
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)

    texts = dataset.data[text_column].tolist()

    # Calculate perplexity for each text sample
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            score = float("inf")
        else:
            input_ids = ref_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                outputs = ref_model(input_ids, labels=input_ids)
            loss = outputs.loss
            score = torch.exp(loss).item()

        measures.append(
            Measure(
                name="perplexity",
                score=score,
                time=datetime.now(),
            )
        )

    return measures
