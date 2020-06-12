#!/usr/bin/env python3
"""
Pooling Back Prop
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network
    :param dA: is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    :param A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c)
    containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
    :param kernel_shape: is a tuple of (kh, kw) containing the size of the
    kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    :param stride: is a tuple of (sh, sw) containing the strides for the
    pooling
        sh is the stride for the height
        sw is the stride for the width
    :param mode: is a string containing either max or avg, indicating
    whether to perform maximum or average pooling, respectively
    :return: the partial derivatives with respect to the previous layer
    (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for img in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for cn in range(c_new):
                    tmp_dA = dA[img, i, j, cn]
                    tmp_A_p = A_prev[img, i * sh:i *
                                     sh + kh, j * sw:j * sw + kw, cn]
                    if mode is 'max':
                        aux = (tmp_A_p == np.max(tmp_A_p))
                    else:
                        aux = np.ones((kh, kw))
                        aux /= (kh * kw)
                    dA_prev[img, i * sh:i * sh + kh, j *
                            sw:j * sw + kw, cn] = aux * tmp_dA

    return dA_prev
