#!/usr/bin/env python3
"""Normal distribution"""


pi = 3.1415926536
e = 2.7182818285


def erf(x):
    """error function encountered in
    integrating the normal distribution"""
    a = (2 / (pi**(1/2)))
    b = (x - ((x**3)/3) + ((x**5)/10) - ((x**7)/42) + ((x**9)/216))
    return a * b


class Normal:
    """Represents an normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal"""
        self.data = data

        if data is None:
            if stddev <= 0:
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

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""

        return self.stddev * z + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        aux = ((x - self.mean) / self.stddev)**2

        return (1 / (self.stddev * (2 * pi)**(1/2))) * e**((-1/2) * aux)

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        aux = (x - self.mean)/(self.stddev * (2**(1/2)))
        return (1/2) * (1 + erf(aux))
