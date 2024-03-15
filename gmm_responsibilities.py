#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:17:58 2024

@author: pzc
"""

import numpy as np
import random
from scipy.stats import multivariate_normal

# Define the means of the Gaussians
mu = [np.array([1e-2, 1e-2]),
      np.array([10, 10]),
      np.array([-10, 10]),
      np.array([-10, -10]),
      np.array([10, -10])]

# Define a shared covariance matrix (assuming a diagonal covariance)
sigma = np.array([[1, 0], [0, 1]])

# Function to perform variational inference
def calculate_responsibilities(data_point, means, covariance):
    """
    Perform variational inference to infer the categorical probability distribution
    of a given data point belonging to each component of a Gaussian mixture model.

    Parameters:
    - data_point: A 2D continuous data point (numpy array).
    - means: A list of means (numpy arrays) for each Gaussian in the mixture.
    - covariance: A shared diagonal covariance matrix (numpy array).

    Returns:
    - A numpy array representing the categorical probability distribution.
    """
    # Compute the unnormalized responsibilities (likelihoods * priors)
    # Assuming equal priors since not specified
    responsibilities = [multivariate_normal.pdf(data_point, mean=mean, cov=covariance) for mean in means]
    
    # Normalize the responsibilities to get probabilities
    total = sum(responsibilities)
    probabilities = np.array([resp / total for resp in responsibilities])
    
    return probabilities

# Example data point
data_point = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])

# Infer the categorical probability distribution
probabilities = calculate_responsibilities(data_point, mu, sigma)

print(data_point)
print(np.argmax(probabilities))
