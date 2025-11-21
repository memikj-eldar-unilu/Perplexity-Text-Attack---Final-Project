import uuid
import pandas as pd
import pytest

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


@pytest.fixture
def squad_datashape() -> DataShape:
    """Creates a DataShape for the SQuAD dataset."""
    features = [
        Feature(
            pid=uuid.uuid4(),
            name="context",  # SQuAD dataset has a 'context' column
            feature_type=FeatureType.TEXT,
            min_value=None,
            max_value=None,
        )
    ]
    return DataShape(features=features, target=None)


@pytest.fixture
def squad_dataset(squad_datashape: DataShape) -> Dataset:
    """Loads the SQuAD validation dataset with dates."""
    df = pd.read_parquet("tests/data/squad_date_val.parquet")
    return Dataset(pid=uuid.uuid4(), shape=squad_datashape, data=df)


@pytest.fixture
def dummy_model(squad_dataset: Dataset) -> Model:
    """Creates a dummy model for the test."""
    return Model(pid=uuid.uuid4(), dataset=squad_dataset)


@pytest.fixture
def dummy_functional_model() -> TabularClassificationModel:
    """Creates a dummy functional model."""
    dummy_predict = lambda x: x
    return TabularClassificationModel(
        predict_class=dummy_predict, predict_proba=None
    )


def test_execute_perplexity_on_squad(
    squad_datashape: DataShape,
    dummy_model: Model,
    squad_dataset: Dataset,
    dummy_functional_model: TabularClassificationModel,
):
    """
    Runs the perplexity metric on the SQuAD dataset and saves the measures.
    """
    measures = perplexity(
        squad_datashape, dummy_model, squad_dataset, dummy_functional_model
    )

    assert len(measures) > 0
    assert len(measures) == len(squad_dataset.data)

    save_measures("perplexity_data", measures)
