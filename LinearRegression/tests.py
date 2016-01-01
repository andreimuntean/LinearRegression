#!/usr/bin/python3

"""tests.py: Tests the Linear Regression model."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
import matplotlib.pyplot as plt
from datahelpers import read_data
from linearregression import LinearRegression


def test(model, x, y):
    for index in range(0, x.shape[0]):
        prediction = model.predict(x[index, :])

        print('Predicted value: {0}\nActual value: {1}\n'.format(prediction, y[index]))

    print('Error cost: {0}'.format(model.get_cost(x, y)))


def run():
    model = LinearRegression()
    data = read_data('data/data-2.csv')

    # Trains the model.
    model.train(data['training_x'], data['training_y'])

    # Tests the model.
    test(model, data['test_x'], data['test_y'])


if __name__ == '__main__':
    run()