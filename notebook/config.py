# data
DATA_FOLDER_PATH = "E:/DataScience/Projects/adult_salary/data/"
TRAINING_DATA_FILE = "/raw/adult_data.txt"

PIPELINE_NAME = 'logistic_regression'

TARGET = 'class'

# input variables 
FEATURES = ['workclass', 'age', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']

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
CATEGORICAL_VARS = ['marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'hours-per-week', 'native-country']
