#!/usr/bin/env python3
"""
RMSProp
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the
    RMSProp optimization algorithm
    """
    s = beta2 * s + (1 - beta2) * grad**2
    var = var - alpha * (grad / (s**(1/2) + epsilon))

    return var, s
