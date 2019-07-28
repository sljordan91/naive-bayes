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
        
        Output: 2d array, shape (n_samples, n_classes)
        """
        assert hasattr(self, 'cond_probs'), 'Model not fit' 
        
        def _vectorized_get(key, dict_):
            return np.vectorize(dict_.__getitem__)(key)

        class_probs_arr = np.atleast_2d([val for val in self.class_probs.values()])
        jll = np.repeat(np.log(class_probs_arr), self.n_samples, axis=0)

        for col in self.categorical_columns:
            log_probs = np.log(np.stack([_vectorized_get(self.X[col].values, self.cond_probs[i][col])\
                                         for i in self.classes_], axis=1))

            # impute log cond prob to 0 in case of 0% cond prob in training set
            # TODO: this should probably be a toggle option in case user wants to throw an error on unseen values
            log_probs[log_probs == -np.inf] = 0

            jll += log_probs

        for col in self.numerical_columns:
            log_probs_by_class = []
            # willing to allow a O(n_features * n_classes) loop for now
            for label in self.classes_:
                mean = self.cond_probs[label][col]['mean']
                std = self.cond_probs[label][col]['std']

                # impute missing values to 0
                log_probs = np.log(gaussian_pdf(self.X[feat].values, mean, std))
                log_probs_0filled = np.nan_to_num(log_probs)

                log_probs_by_class.append(log_probs_0filled)

            log_probs_array = np.stack(log_probs_by_class, axis=1)
            jll += log_probs_array
        
        return jll

    def predict(self, X, top_n=1):
        jll = self._joint_log_likelihood(X)

        return np.argmax(jll, axis=1)[:top_n]

if __name__ == '__main__':
    gnb = GeneralNB()
    print(gnb)
