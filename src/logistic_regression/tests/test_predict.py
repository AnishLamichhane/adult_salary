import math
import numpy as np
from logistic_regression.predict import make_prediction
from logistic_regression.processing.data_management import load_dataset


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='adult_test.txt')
    single_test = test_data.head(1)

    # When
    subject = make_prediction(input_data=single_test)

    # Then
    assert subject is not None
    print (type(subject.get('prediction')[0][0]))

    assert isinstance(subject.get('prediction')[1][0], np.float)

#test_make_single_prediction()