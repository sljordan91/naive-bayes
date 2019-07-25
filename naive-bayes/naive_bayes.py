"""
Implements Naive Bayes algorithm following sklearn API design.

Allows mixing of both categorical and numerical predictors.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BaseNB
# possibly columntransformer

class GeneralNaiveBayes(BaseNB):
    """
    Uses pandas dataframe for X to allow mixed data types.
    """

    def fit(self, X, y):
        pass

    def _joint_log_likelihood(self, X):
        pass

    def predict(self, X, top_n=1):
        jll = self._joint_log_likelihood(X)

	return np.argmax(jll, axis=1)[:top_n]

