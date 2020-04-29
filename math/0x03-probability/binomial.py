#!/usr/bin/env python3
"""Binomial distribution"""


pi = 3.1415926536
e = 2.7182818285


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial"""

        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            s_dif = []
            for d in data:
                s_dif.append((d - mean)**2)

            stddev = (sum(s_dif) / len(s_dif))**(1/2)
            var = stddev**2
            p = -((var / mean) - 1)
            self.n = round(mean / p)
            self.p = mean / self.n
