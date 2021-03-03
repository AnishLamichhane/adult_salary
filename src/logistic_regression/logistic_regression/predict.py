import pandas as pd
import joblib

from logistic_regression.processing.data_management import load_pipeline
from logistic_regression.processing.data_management import validate_inputs
from logistic_regression.config import config
from logistic_regression import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_adult_salary_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """ Make a prediction using saved model pipeline.
    The result is list with two columns.
    First columns is prediction class.
    Second column is the probability of true class."""

    # data = pd.read_csv(input_data)
    validated_data = validate_inputs(input_data)

    output = [_adult_salary_pipe.predict(validated_data[config.FEATURES]),
              _adult_salary_pipe.predict_proba(validated_data[config.FEATURES])[:, 1]]

    results = {'prediction': output, 'version': _version}
    _logger.info(
        f"Making predictions with model version: {_version}"
        f"Inputs: {validated_data}"
        f"Predictions: {results}"
    )

    return results
