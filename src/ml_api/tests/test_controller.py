from logistic_regression.config import config as model_config
from logistic_regression.processing.data_management import   load_dataset
from logistic_regression import __version__ as _version

import json
import math
import pandas as pd

def test_health_endpoint_returns_200(flask_test_client):
    # when
    response = flask_test_client.get('/health')

    # then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # given
    # load the tests data from the logistic regression package.
    test_data = load_dataset(file_name=model_config.TEST_DATA_FILE)
    post_json = test_data.head(5).to_json(orient='records')

    # when
    response = flask_test_client.post('/v1/predict/logistic_regression',
                                      json=post_json)

    # then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert response_version ==_version

