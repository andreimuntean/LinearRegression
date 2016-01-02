#!/usr/bin/python3

"""datahelpers.py: Provides functions for handling data."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


def shuffle_data(x, y):
    random_indexes = np.random.permutation(x.shape[0])
    shuffled_x = np.empty_like(x)
    shuffled_y = np.empty_like(y)

    for index in range(0, shuffled_x.shape[0]):
        random_index = random_indexes[index]
        shuffled_x[index] = x[random_index]
        shuffled_y[index] = y[random_index]

    return x, y


def split_data(x, y, threshold = 0.7, shuffle = True):
    """Generates training and tests sets from the specified data.""" 

    if shuffle:
        x, y = shuffle_data(x, y)
    
    pivot_index = round(threshold * x.shape[0])

    training_data = {
        'x': x[0 : pivot_index],
        'y': y[0 : pivot_index]
    }

    test_data = {
        'x': x[pivot_index:],
        'y': y[pivot_index:]
    }

    return training_data, test_data


def read_data(path):
    """Reads csv-formatted data from the specified path."""

    data = np.loadtxt(path, delimiter = ',')

    # Gets the dependent variables. They're stored in the first column.
    y = data[:, 0]

    # Gets the independent variables.
    x = data[:, 1:]

    return x, y