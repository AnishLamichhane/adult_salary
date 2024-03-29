import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from logistic_regression.config import config
from logistic_regression import __version__ as _version

import logging
import typing as t

_logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """ loads data give filename from the dataset directory"""
    _logger.info('Loading data')

    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}", header=None)
    _data.columns = config.DATA_COLUMNS

    _logger.info('Data loaded successfully.')

    return _data


def validate_inputs(input_data) -> pd.DataFrame:
    _logger.info('Validating input format.')
    validated_data = input_data.copy()
    if isinstance(validated_data, list):
        _logger.info('Converting list to dataframe.')
        return pd.DataFrame(validated_data)

    if isinstance(validated_data, pd.Series):
        _logger.info('Converting series to dataframe.')
        return validated_data.to_frame()

    return validated_data


def save_pipeline(*, pipeline_to_presist) -> None:
    """ Presist the pipeline
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is published,
    there is only one trained model that can be called, and we know
    exactly how it was build."""

    _logger.info('Saving fitted pipeline.')

    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_presist, save_path)
    _logger.info(f"Saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """ Load a persisted pipeline."""
    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    _logger.info('Loaded saved pipeline.')
    return saved_pipeline


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """ Removes old model pipelines."""
    # we need to preserve the __init__ files for module access
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

    _logger.info('Deleted older pipelines.')
