# -*- coding: utf-8 -*-

"""
Implements Naive Bayes algorithm following sklearn API design.

Allows mixing of both categorical and numerical predictors.
Also permits NaNs in both fitting and scoring by ignoring the value.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BaseNB
# possibly columntransformer

from utils import gaussian_pdf


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

        self.categorical_columns = X.select_dtypes(exclude=('number',)).columns.tolist()
        self.numerical_columns = X.select_dtypes(include=('number',)).columns.tolist()
        
        # calculate class and conditional probabilities
        self.class_probs = dict()
        self.cond_probs = dict()
        for label in self.classes_:
            self.class_probs[label] = sum(self.y == label) / self.n_samples
        
            # set conditional probabilities by feature
            self.cond_probs[label] = dict()
            
            # numerical columns predicted using Gaussian based on empirical mean, std
            for col in self.numerical_columns:
                self.cond_probs[label][col] = dict()
                cond_df = self.X.loc[self.y == label, col].values
                self.cond_probs[label][col]['mean'] = np.nanmean(cond_df)
                self.cond_probs[label][col]['std'] = np.nanstd(cond_df)
            
            for col in self.categorical_columns:
                # don't factor untrained values into class likelihoods (probability of 1 is no-op)
                self.cond_probs[label][col] = defaultdict(lambda: 1)
                
                for value in self.df[col].unique():
                    cond_df = self.y[self.X[col] == value]
                    self.cond_probs[label][col][value] = sum(cond_df == label) / len(cond_df)
                
                # don't factor NaNs into class likelihoods (probability of 1 is no-op)
                self.cond_probs[label][col][np.nan] = 1
                    


    def _joint_log_likelihood(self, X):
        """
        Calculates the log likelihood of each class.
        
        Output: 2d array, shape (n_records, n_classes)
        """
        # TODO: check that model is fitted
        
        jll = 

    def predict(self, X, top_n=1):
        jll = self._joint_log_likelihood(X)

        return np.argmax(jll, axis=1)[:top_n]

    def _mark_categorical_colums(self, X):
        return X.select_dtypes(include=['category', 'object', 'bool'])

if __name__ == '__main__':
    gnb = GeneralNB()
    print(gnb)
