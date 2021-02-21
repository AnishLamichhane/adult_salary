import typing as t

from marshmallow import Schema, fields
from marshmallow import ValidationError

from api import config


class InvalidInputError(Exception):
    """Invalid model input."""


SYNTAX_ERROR_FIELD_MAP = {
    'education-num': 'education_num',
    'marital-status': 'marital_status',
    'capital-gain': 'capital_gain',
    'hours-per-week': 'hours_per_week',
    'native-country': 'native_country'
}


class Adult_Salary(Schema):
    age = fields.Integer()
    workclass = fields.Str()
    fnlwgt = fields.Integer()
    education = fields.Str()
    education_num = fields.Integer()
    marital_status = fields.Str()
    occupation = fields.Str()
    relationship = fields.Str()
    race = fields.Str()
    sex = fields.Str()
    capital_gain = fields.Float()
    capital_loss = fields.Float()
    hours_per_week = fields.Float()
    native_country = fields.Str()



def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(*,input_data:list):
    """Check prediction inputs against schema."""

    # set many=True to allow passing in a list
    schema = Adult_Salary(strict=True, many=True)

    '''
    # convert syntax error field names (beginning with numbers)
    for dict in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict[value] = dict[key]
            del dict[key]
    '''
    errors = None

    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages

    # convert syntax error field names back
    # this is a hack - never name your data
    # fields with numbers as the first letter.
    '''
    for dict in input_data:
        for key, value in SYNTAX_ERROR_FIELD_MAP.items():
            dict[key] = dict[value]
            del dict[value]
    '''
    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data
        errors=[]

    return validated_input, errors


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS