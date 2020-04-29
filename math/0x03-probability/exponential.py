#!/usr/bin/env python3
"""Exponential distribution"""


pi = 3.1415926536
e = 2.7182818285


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Exponential"""
        self.data = data

        if data is None:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)

        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))
