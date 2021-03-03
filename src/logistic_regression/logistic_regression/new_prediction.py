from sklearn.metrics import roc_auc_score, f1_score

from logistic_regression import config
import pandas as pd
import joblib
from logistic_regression.processing.data_management import  load_dataset
from logistic_regression.predict import make_prediction

_data = load_dataset(file_name=config.TRAINING_DATA_FILE)


# extract target variable
y_test = _data[config.TARGET]
y_test = [config.TARGET_DICT[x.split('.')[0]] for x in y_test]

results = make_prediction(input_data=_data)

print('tests F1 score: {}'.format(f1_score(y_test, results.get('prediction')[0])))
print('tests roc-auc: {}'.format(roc_auc_score(y_test, results.get('prediction')[1])))