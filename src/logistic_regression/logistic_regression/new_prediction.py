from sklearn.metrics import roc_auc_score, f1_score

import config
import pandas as pd
import joblib
from predict import make_prediction

data = pd.read_csv(config.DATA_FOLDER_PATH + config.TEST_DATA_FILE,skiprows=1)
data.columns = config.DATA_COLUMNS

# extract target variable
y_test = data[config.TARGET]
y_test = [config.TARGET_DICT[x.split('.')[0]] for x in y_test]

results = make_prediction(data[config.FEATURES])

print('tests F1 score: {}'.format(f1_score(y_test, results[0])))
print('tests roc-auc: {}'.format(roc_auc_score(y_test, results[1])))