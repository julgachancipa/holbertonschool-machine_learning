#!/usr/bin/env python3
"""Sigma Notation"""


def summation_i_squared(n):
    """Calculates the summation of i²
    in a [1, n]"""

    result = 0

    for i in range(1, n + 1):
        result += i**2

    return result
