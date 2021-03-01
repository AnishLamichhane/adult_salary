import json

from logistic_regression.config import config
from logistic_regression.processing.data_management import load_dataset


def test_prediction_endpoint_validation_200(flask_test_client):
    # Given
    # Load the test data from the regression_model package.
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = load_dataset(file_name=config.TEST_DATA_FILE)
    sample_size = 10
    post_json = test_data.head(sample_size).to_json(orient='records')


    # When
    response = flask_test_client.post('/v1/predict/logistic_regression',
                                      json=post_json)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)

    # Check correct number of errors removed
    assert len(response_json.get('predictions')) + len(
        response_json.get('errors')) == sample_size