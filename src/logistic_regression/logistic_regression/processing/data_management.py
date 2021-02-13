import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from logistic_regression.config import config

def load_dataset(*,file_name: str)-> pd.DataFrame:
    """ loads data give filename from the dataset directory"""
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}", header=None)
    _data.columns = config.DATA_COLUMNS
    return _data

def save_pipeline(*, pipeline_to_presist)-> None:
    """ Presist the pipeline
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is published,
    there is only one trained model that can be called, and we know
    exactly how it was build."""

    save_file_name = 'logistic_regression.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_presist, save_path)
    print("Saved pipeline")

def load_pipeline(*, file_name: str)-> Pipeline:
    """ Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR/file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline

