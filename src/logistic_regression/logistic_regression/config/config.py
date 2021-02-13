import pathlib
import logistic_regression

PACKAGE_ROOT =pathlib.Path(logistic_regression.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models/"
DATASET_DIR= PACKAGE_ROOT /'datasets/'

# data

TRAINING_DATA_FILE = "adult_data.txt"
TEST_DATA_FILE = "adult_test.txt"

# random state for reproduction
RANDOM_STATE = 0

# saved piplines
PIPELINE_NAME = 'logistic_regression.pkl'


# target variables
TARGET = 'class'
TARGET_DICT = {' <=50K': 1, ' >50K': 0}

# input variables
DATA_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']

FEATURES = ['workclass', 'age', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# this variable is to calculate the temporal variable,
# must be dropped afterwards
DROP_FEATURES = 'fnlwgt'

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = []

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['workclass', 'occupation', 'native-country']

TEMPORAL_VARS = []

# variables to DISCREET
NUMERICAL_VARS_TO_DISCREET = ['age', 'education-num']

# variables to Scale
NUMERICALS_TO_SCALE = ['capital-loss', 'capital-gain', 'hours-per-week']

# categorical variables to encode
CATEGORICAL_VARS = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'native-country', 'age_group', 'edu-group']

# selected featues for model training
SELECTED_VARS = ['capital-gain', 'capital-loss', 'hours-per-week','workclass_ Self-emp-not-inc', 'education_ 7th-8th',
                 'marital-status_ Divorced', 'marital-status_ Married-civ-spouse',
                 'marital-status_ Never-married', 'marital-status_ Separated',
                 'occupation_ Exec-managerial', 'occupation_ Farming-fishing',
                 'occupation_ Handlers-cleaners', 'occupation_ Other-service',
                 'relationship_ Other-relative', 'relationship_ Own-child',
                 'relationship_ Wife', 'race_Rare', 'sex_ Female',
                 'native-country_ Mexico', 'age_group_<=25', 'edu-group_<=8.5']

