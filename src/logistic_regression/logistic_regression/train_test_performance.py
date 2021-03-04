from sklearn.metrics import roc_auc_score, f1_score

from logistic_regression import config
import pandas as pd
import joblib
from logistic_regression.processing.data_management import  load_dataset
from logistic_regression.predict import make_prediction

_data_train = load_dataset(file_name=config.TRAINING_DATA_FILE)
_data_test = load_dataset(file_name=config.TEST_DATA_FILE)

# extract target variable
y_train = _data_train[config.TARGET]
y_train = [config.TARGET_DICT[x.split('.')[0]] for x in y_train]
y_test = _data_test[config.TARGET]
y_test = [config.TARGET_DICT[x.split('.')[0]] for x in y_test]

results_train = make_prediction(input_data=_data_train)
results_test = make_prediction(input_data=_data_test)

print('Train F1 score: {}'.format(f1_score(y_train, results_train.get('prediction')[0])))
print('Train roc-auc: {}'.format(roc_auc_score(y_train, results_train.get('prediction')[1])))
print('Tests F1 score: {}'.format(f1_score(y_test, results_test.get('prediction')[0])))
print('Tests roc-auc: {}'.format(roc_auc_score(y_test, results_test.get('prediction')[1])))