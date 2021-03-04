from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


from logistic_regression.processing import preprocessor as pp
from logistic_regression.config import config

adult_salary_pipe=Pipeline(
    [
        ('categorical_imputer',
         pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),
        ('numerical_discretizer',
         pp.NumericalDiscretizer(variables=config.NUMERICAL_VARS_TO_DISCREET)),
        ('rare_label_encoder',
         pp.RareLabelCategoricalEncoder(
             tol=config.TOLERANCE,
             variables=config.CATEGORICAL_VARS)),
        ('categorical_encoder',
         pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('filter_feature',pp.FilterFeatures(variables=config.SELECTED_VARS)),
        ('scaler',MinMaxScaler()),
        ('Logistic_regression',LogisticRegression(C=0.05, random_state=0, penalty='l2'))

    ])
