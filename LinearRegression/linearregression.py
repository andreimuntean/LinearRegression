#!/usr/bin/python3

"""linearregression.py: Finds relationships between dependent and independent variables."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def train(self, x, y, maximum_iterations = 100000):
        pass

    @staticmethod
    def get_cost(x, y, theta, regularization_term = 0.1):
        """Computes the cost of linear regression using the values of theta as its parameters."""

        # The number of training examples.
        m = x.shape[0]
        predictions = x @ theta.transpose()
        squared_errors = np.power(predictions - y.transpose(), 2)

        return 1 / (2 * m) * np.sum(squared_errors)