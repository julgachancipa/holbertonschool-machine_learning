#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a
    convolutional layer of a neural network
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    :param W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    :param b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    :param activation: activation function applied to the convolution
    :param padding: string that is either same or valid, indicating the type of
    padding used
    :param stride: tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    :return: output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        ph = int(np.ceil(ph / 2))
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        pw = int(np.ceil(pw / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding[0], padding[1]

    oh = ((h_prev - kh + (2 * ph)) // sh) + 1
    ow = ((w_prev - kw + (2 * pw)) // sw) + 1

    A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    conv = np.zeros((m, oh, ow, c_new))
    for k in range(c_new):
        for i in range(oh):
            for j in range(ow):
                aux = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw] \
                      * W[:, :, :, k]
                conv[:, i, j, k] = np.sum(aux, axis=(1, 2, 3)) + b[:, :, :, k]
    return activation(conv)
