#!/usr/bin/env python3
"""
Pooling Forward Prop
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    :param kernel_shape: tuple of (kh, kw) containing the size of the kernel
    for the pooling
        kh is the kernel height
        kw is the kernel width
    :param stride: tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    :param mode: string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    :return: output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = ((h_prev - kh) // sh) + 1
    ow = ((w_prev - kw) // sw) + 1
    pool = np.zeros((m, oh, ow, c_prev))

    for i in range(oh):
        for j in range(ow):
            aux = np.reshape(A_prev[:, i * sh:i * sh + kh,
                             j * sw:j * sw + kw, :],
                             (m, kh * kw, c_prev))
            if mode is 'max':
                pool[:, i, j, :] = np.amax(aux, axis=1)
            elif mode is 'avg':
                pool[:, i, j, :] = np.sum(aux, axis=1) / (kh * kw)

    return pool
