"""
Implements Naive Bayes algorithm following sklearn API design.

Allows mixing of both categorical and numerical predictors.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BaseNB
# possibly columntransformer

class GeneralNB(BaseNB):
    """
    Uses pandas dataframe for X to allow mixed data types.
    
    Categorical features can be used as-is with no encoding.
    Numerical features are assumed to follow a Gaussian distribution.

    Attributes
    ----------

    X : pandas dataframes
        Features used in model training

    y : array-like, or pandas series/dataframe
        Contains target predicted against

    n_samples : int
        Number of samples in training set

    n_features : int
        Number of features in training set

    classes_ : list[Categorical]
        Unique labels or classes in target vector y

    n_classes : int
        Number of unique values in target vector y

    categorical_columns : List[string]
        Names of categorical columns in X

    numerical_columns : List[string]
        Name of numerical columns in X

    """

    def fit(self, X, y):
        self.X = X
	self.y = y

	self.n_samples = X.shape[0]
	self.n_features = X.shape[1]
	self.classes_ = np.unique(y)
	self.n_classes = len(self.classes_)

        self.categorical_columns = X.select_dtypes(exclude=('number',))\
	                            .columns.tolist()
        self.numerical_columns = X.select_dtypes(include=('number',))\
	                          .columns.tolist()


    def _joint_log_likelihood(self, X):
        
	# calculate class probabilities
        pass

    def predict(self, X, top_n=1):
        jll = self._joint_log_likelihood(X)

	return np.argmax(jll, axis=1)[:top_n]

    def _mark_categorical_colums(self, X):
	return X.select_dtypes(include=['category', 'object', 'bool'])

if __name__ == '__main__':
    gnb = GeneralNB()
    print(gnb)
