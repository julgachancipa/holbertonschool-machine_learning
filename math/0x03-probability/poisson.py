#!/usr/bin/env python3
"""Poisson distribution"""


pi = 3.1415926536
e = 2.7182818285


def factorial(n):
    """Finds the factorial of a given number"""
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


class Poisson:
    """Represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson"""

        if data is None:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)

        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of successes (k)"""

        if k < 0:
            return 0
        if type(k) != int:
            k = int(k)
        return (e**(-self.lambtha) * self.lambtha**k) / factorial(k)

    def cdf(self, k):
        """Calculates the value of the CDF
        for a given number of successes (k)"""

        if k < 0:
            return 0

        if type(k) != int:
            k = int(k)

        p = 0
        for i in range(k + 1):
            p += ((self.lambtha**i) / factorial(i))

        return e**(-self.lambtha) * p
