#!/usr/bin/env python3
"""
Adam
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    Updates a variable using the
    Adam optimization algorithm
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    v_c = v / (1-beta1**t)
    s_c = s / (1-beta2**t)

    var = var - alpha * (v_c / (s_c**(1/2) + epsilon))

    return var, v, s
