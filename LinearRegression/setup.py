#!/usr/bin/python3

"""setup.py: Installs the modules required to run linearregression.py."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

from setuptools import setup


setup(
    name = 'Linear Regression',
    version = '0.1.0',
    description = 'Finds relationships between dependent and independent variables.',
    author = 'Andrei Muntean',
    license = 'MIT',
    keywords = 'linear regression multiple machine learning ml predict',
    install_requires = ['numpy', 'matplotlib']
)