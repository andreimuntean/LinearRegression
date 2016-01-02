#!/usr/bin/python3

"""tests.py: Tests the Linear Regression model."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
import matplotlib.pyplot as plt
from datahelpers import read_data, split_data
from linearregression import LinearRegression


def test(model, x, y):
    # Iterates through a few examples.
    for index in range(0, min(x.shape[0], 10)):
        prediction = model.predict(x[index])

        # Displays the prediction.
        print('Predicted value: {0}'.format(prediction))
        print('Actual value: {0}\n'.format(y[index]))

    print('Error cost: {0}'.format(model.get_cost(x, y)))


def run():
    model = LinearRegression()
    x, y = read_data('data/data-1.csv')
    training_data, test_data = split_data(x, y)

    # Trains the model.
    model.train(training_data['x'], training_data['y'])

    # Tests the model.
    test(model, test_data['x'], test_data['y'])
    
    # Plots the results if they are two dimensional.
    if x.shape[1] == 1:
        predictions = np.apply_along_axis(model.predict, 1, x)

        plt.plot(training_data['x'], training_data['y'], 'bo',
            test_data['x'], test_data['y'], 'go',
            x, predictions, '--r')

        plt.show()


if __name__ == '__main__':
    run()