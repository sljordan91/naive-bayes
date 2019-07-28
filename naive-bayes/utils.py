# -*- coding: utf-8 -*-

"""
Small utility functions.
"""

import numpy as np

def gaussian_pdf(x, mu, sigma):
    """
    Probability density function of a Gaussian distribution.
    """
    var = sigma ** 2
    num = np.exp(-(x - mu) ** 2 / (2 * var))
    denom = np.sqrt(2 * np.pi * var)
    
    return num / denom

print(gaussian_pdf(7, 5, 5))