import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

_logger = logging.getLogger(__name__)

# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # need fit statement to accommodate sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].replace(' ?', 'missing')

        return X


# discretize variable
def get_age_group(x):
    if x <= 25:
        return '<=25'
    elif x <= 30:
        return '25-30'
    elif x <= 35:
        return '30-35'
    elif x <= 45:
        return '35-45'
    elif x <= 60:
        return '45-60'
    elif x <= 70:
        return '<=70'
    else:
        return '>70'


def get_education_group(edu):
    if edu <= 8.5:
        return '<=8.5'
    elif edu <= 12.5:
        return '<=12.5'
    elif edu <= 14.5:
        return '<=14.5'
    else:
        return '>14.5'


class NumericalDiscretizer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = variables
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # need fit statement to accomodate sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            if feature == 'age':
                # age discretization
                test=X['age']
                X['age_group'] = X['age'].apply(lambda x: get_age_group(x))
            elif feature == 'education_num':
                # education-num discretization
                X['edu_group'] = X[feature].apply(lambda x: get_education_group(x))
        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):

        self.tol = tol

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                                                      feature]), X[feature], 'Rare')

        return X


class FilterFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.variables]
        return X



# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=False).columns
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        X = pd.concat([X,
                       pd.get_dummies(X[self.variables], drop_first=False)],
                       axis=1)

        #X.drop(labels=self.variables, axis=1, inplace=False)

        # add missing dummies if any
        missing_vars = [var for var in self.dummies if var not in X.columns]

        if len(missing_vars) != 0:
            for var in missing_vars:
                X[var] = 0

        return X
