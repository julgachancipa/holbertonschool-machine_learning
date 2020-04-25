#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if type(poly) != list or type(C) != int:
        return None
    for p in poly:
        if type(p) != int and type(p) != float:
            return None
    if poly == []:
        return None
    result = [C]
    for i in range(len(poly)):
        r = poly[i] / (i+1)
        if r.is_integer():
            r = int(r)
        result.append(r)
    return result
