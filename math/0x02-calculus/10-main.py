#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))
print(poly_derivative([]))
print(poly_derivative([9]))
print(poly_derivative(5))
print(poly_derivative(['a', 'b']))
