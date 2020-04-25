#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if type(poly) != list or type(C) != int:
        return None
    if poly == []:
        return None
    result = [C]
    for i in range(len(poly)):
        if poly[i] == 0:
            result.append(0)
        else:
            result.append(poly[i] / (i+1))

    return result
