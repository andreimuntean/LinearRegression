#!/usr/bin/python3

"""linearregression.py: Finds relationships between dependent and independent variables."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


class LinearRegression:
    def train(self, x, y):
        self.mean = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)
        x, y = self.__preprocess_data(x, y)

        self.parameters = LinearRegression.__gradient_descent(x, y)

    def predict(self, x):
        x = self.__preprocess_data(x, None)[0]

        return x @ self.parameters

    def get_cost(self, x, y):
        """Evaluates the performance of the model. A smaller value represents a higher degree of accuracy."""
        
        x, y = self.__preprocess_data(x, y)

        return LinearRegression.__get_cost(x, y, self.parameters)[0]

    def __preprocess_data(self, x, y):
        # Normalizes the features to hasten convergence.
        x = self.__normalize_features(x)

        # Adds the bias term.
        x = LinearRegression.__add_bias(x)
        x = np.atleast_2d(x)

        # Converts y into a column vector.
        y = np.atleast_2d(y).T

        return x, y

    def __normalize_features(self, x):
        return (x - self.mean) / self.std
  
    @staticmethod
    def __add_bias(x):
        """Prepends the bias term to the specified array."""

        if x.shape[0] == 1:
            return np.append(1, x)
        else:
            return np.append(np.ones([x.shape[0], 1]), x, axis = 1)

    @staticmethod
    def __gradient_descent(x, y, maximum_iterations = 100000, epsilon = 1e-7):
        """Trains the model using gradient descent."""

        # Determines the number of training examples.
        m = x.shape[0]

        # Determines the extent to which parameters are updated.
        learning_rate = 0.001

        # Keeps a history of outputs from the cost function.
        cost_history = np.zeros([maximum_iterations, 1])

        # Initializes the parameters.
        parameters = np.zeros([x.shape[1], 1])

        # Iterates through the training set multiple times.
        for iteration in range(0, maximum_iterations):
            errors = x @ parameters - y
            parameters -= learning_rate / m * x.T @ errors

            # Records the cost for this iteration.
            cost_history[iteration, 0] = LinearRegression.__get_cost(x, y, parameters)[0]

            # Stops if the performance gains become negligible.
            if iteration > 0 and cost_history[iteration - 1, 0] - cost_history[iteration, 0] < epsilon:
                break

        return parameters

    @staticmethod
    def __get_cost(x, y, parameters, regularization_term = 0):
        # Determines the number of training examples.
        m = x.shape[0]

        # Determines the number of incorrect predictions.
        errors = x @ parameters - y
        sum_of_squared_errors = np.sum(np.power(errors, 2))

        # Determines the regularization applied to the cost to prevent overfitting.
        regularization_values = regularization_term * np.sum(np.power(parameters[1:, :], 2))

        # Calculates and regularizes the cost.
        cost = (sum_of_squared_errors + regularization_values) / 2 * m

        # Calculates the gradient -- the derivative of the cost function.
        gradient = errors.T @ x / m
        gradient[:, 1:] += regularization_term * parameters[1:, :].T

        return cost, gradient