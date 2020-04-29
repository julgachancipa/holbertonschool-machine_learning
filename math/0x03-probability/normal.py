#!/usr/bin/env python3
"""Normal distribution"""


pi = 3.1415926536
e = 2.7182818285


class Normal:
    """Represents an normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal"""
        self.data = data

        if data is None:
            if stddev < 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            self.mean = float(mean)

        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)

            s_dif = []
            for d in data:
                s_dif.append((d - self.mean)**2)

            self.stddev = (sum(s_dif) / len(s_dif))**(1/2)
