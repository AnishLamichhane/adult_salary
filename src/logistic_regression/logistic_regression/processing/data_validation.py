from logistic_regression.config import config
import json
import pandas as pd

def validate_inputs (input_data)-> pd.DataFrame:

    validated_data = input_data.copy()
    if isinstance(validated_data,list):
        return pd.DataFrame(validated_data)

    if isinstance(validated_data, pd.Series):
        return validated_data.to_frame()


    return validated_data

def is_json(data):
    try:
        json_object = json.load(data[0])
    except ValueError as e:
        return False
    return True