import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from logistic_regression import pipeline
from logistic_regression.config import config
from logistic_regression.processing.data_management import save_pipeline
from logistic_regression.processing.data_management import load_dataset

def run_training():
    """ Train the model"""

    #data = pd.read_csv(config.DATASET_DIR/config.TRAINING_DATA_FILE)
    _data = load_dataset(file_name =config.TRAINING_DATA_FILE)
    # assign column labels to the imported data
    #_data.columns = config.DATA_COLUMNS

    # divide train and tests
    x_train, x_test, y_train, y_test = train_test_split(_data[config.FEATURES], _data[config.TARGET], test_size=0.2,
                                                        random_state=0, stratify=_data[config.TARGET])

    # transform the target
    y_train = [config.TARGET_DICT[x] for x in y_train]
    y_test = [config.TARGET_DICT[x] for x in y_test]

    pipe=pipeline.adult_salary_pipe.fit(x_train[config.FEATURES],y_train)

    save_pipeline(pipeline_to_presist=pipe)

if __name__ == '__main__':
    run_training()
