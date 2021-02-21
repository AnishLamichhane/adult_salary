import json

from flask import Blueprint, request,jsonify
from logistic_regression.predict import make_prediction
import pandas as pd
from logistic_regression import __version__ as _version
from api.config import get_logger
from api import __version__ as api_version
from api.validation import validate_inputs

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app',__name__)

@prediction_app.route('/health',methods=['GET'])
def health():
    if request.method =='GET':
        return 'ok'


@prediction_app.route('/v1/predict/logistic_regression',methods=['POST'])
def predict():
    if request.method == 'POST':
        # Step1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        # Step 2: validate the input using marshmallow schema
        input_data = json.loads (json_data)
        input_data,errors = validate_inputs(input_data=input_data)

        # _data = pd.read_json(json_data)
        result = make_prediction(input_data=input_data)
        _logger.info(f'Outputs: {result}')

        predictions = result.get('prediction')[0].tolist()
        version = result.get('version')

        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors
                        })


@prediction_app.route('/version',methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                       'api_version': api_version
                        })




