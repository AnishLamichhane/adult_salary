import pandas as pd
import joblib

from logistic_regression.processing.data_management import load_pipeline
from logistic_regression.processing.data_validation import validate_inputs
from logistic_regression.config import config

pipeline_file_name = config.PIPELINE_NAME
_adult_salary_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: pd.DataFrame) -> list:
    """ Make a prediction using saved model pipeline.
    The result is list with two columns.
    First columns is prediction class.
    Second column is the probability of true class."""

    # data = pd.read_csv(input_data)
    input_data=validate_inputs(input_data)
    results = [_adult_salary_pipe.predict(input_data[config.FEATURES]),
               _adult_salary_pipe.predict_proba(input_data[config.FEATURES])[:, 1]]

    return results

