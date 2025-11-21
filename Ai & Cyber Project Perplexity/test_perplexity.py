import pandas as pd
import pytest
import uuid

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model, Feature, FeatureType
from a4s_eval.metrics.model_metrics.perplexity_metric import perplexity
from a4s_eval.service.functional_model import TabularClassificationModel


@pytest.fixture
def sample_data():
    """Provides a sample dataset and datashape for testing."""
    data = {"text": ["this is a sample sentence", "this is another one"]}
    df = pd.DataFrame(data)

    features = [
        Feature(
            pid=uuid.uuid4(),
            name="text",
            feature_type=FeatureType.TEXT,
            min_value=None,
            max_value=None,
        )
    ]
    datashape = DataShape(features=features, target=None)

    dataset = Dataset(pid=uuid.uuid4(), shape=datashape, data=df)

    model = Model(pid=uuid.uuid4(), dataset=dataset)

    # The functional_model is not used, but we need a valid object to satisfy the type hint.
    dummy_predict = lambda x: x
    functional_model = TabularClassificationModel(
        predict_class=dummy_predict, predict_proba=None
    )

    return datashape, model, dataset, functional_model


def test_perplexity_calculation(sample_data):
    """
    Tests that perplexity metric returns the correct number of measures
    with valid scores.
    """
    datashape, model, dataset, functional_model = sample_data
    
    measures = perplexity(datashape, model, dataset, functional_model)
    
    # Check that we have one measure per input text
    assert len(measures) == len(dataset.data)
    
    # Check that the measure objects are correctly formed
    for measure in measures:
        assert measure.name == "perplexity"
        assert isinstance(measure.score, float)
        assert measure.score > 0.0

def test_perplexity_invalid_text(sample_data):
    """
    Tests that perplexity metric handles invalid text entries gracefully.
    """
    datashape, model, dataset, functional_model = sample_data
    
    # Add an invalid entry to the dataset
    dataset.data.loc[len(dataset.data)] = {"text": ""}
    
    measures = perplexity(datashape, model, dataset, functional_model)
    
    assert len(measures) == len(dataset.data)
    
    # The score for the invalid text should be infinity
    invalid_measure = measures[-1]
    assert invalid_measure.score == float('inf')

def test_perplexity_missing_column(sample_data):
    """
    Tests that perplexity raises a ValueError if the text column is missing.
    """
    datashape, model, dataset, functional_model = sample_data
    
    # Rename the column to something else
    dataset.data = dataset.data.rename(columns={"text": "wrong_column"})
    
    with pytest.raises(ValueError, match="Text column 'text' not found in the dataset."):
        perplexity(datashape, model, dataset, functional_model)

