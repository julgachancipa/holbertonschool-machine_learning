#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if poly == []:
        return None
    result = []
    for i in range(1, len(poly)):
        result.append(i * poly[i])
    """
    if result == [0] * len(result):
        return [0]
    """
    return result
