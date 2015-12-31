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