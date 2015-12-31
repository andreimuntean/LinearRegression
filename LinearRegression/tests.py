#!/usr/bin/python3

"""tests.py: Tests the Linear Regression model."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
import matplotlib.pyplot as plt
from datahelpers import read_data
from linearregression import LinearRegression


def run():
    model = LinearRegression()
    data = read_data('data/data-1.csv')

    # Trains the model.
    model.train(data['training_x'], data['training_y'])

    cost = LinearRegression.get_cost(np.array([[1, 2], [1, 3], [2, 4]]),
                              np.array([[3, 4, 6]]),
                              np.array([[1, 1]]))

    print(cost)


if __name__ == '__main__':
    run()