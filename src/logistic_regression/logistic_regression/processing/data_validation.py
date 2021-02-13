from logistic_regression.config import config

import pandas as pd

def validate_inputs (input_data)-> pd.DataFrame:
    validated_data = input_data.copy()

    if isinstance(validated_data, pd.Series):
        validated_data= validated_data.to_frame()

    return validated_data
