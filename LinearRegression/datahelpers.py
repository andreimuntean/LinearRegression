#!/usr/bin/python3

"""datahelpers.py: Provides functions for handling data."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


def separate_data(x, y, threshold = 0.7):
    """Splits the specified data into training and test sets."""

    pivot_index = threshold * x.shape[0]

    return {
        'training_x': x[1 : pivot_index],
        'training_y': y[1 : pivot_index],
        'test_x': x[pivot_index :],
        'test_y': y[pivot_index :]
    }


def read_data(path):
    """Reads csv-formatted data from the specified path."""

    data = np.loadtxt(path, delimiter = ',')

    # Gets the dependent variables. They're stored in the first column.
    y = data[:, 0]

    # Gets the independent variables.
    x = data[:, 1:]

    return separate_data(x, y)