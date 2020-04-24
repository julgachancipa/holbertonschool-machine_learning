#!/usr/bin/env python3
"""Sigma Notation"""


def summation_i_squared(n):
    """Calculates the summation of i²
    in a [1, n] range"""

    range_l = list(range(1, n+1))
    return sum(map(lambda i: i**2, range_l))
