#!/usr/bin/env python3
"""
Convolutional Back Prop
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Cerforms back propagation over a convolutional layer of a neural network
    :param dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated output of the
    convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    :param A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    :param W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
        kh is the filter height
        kw is the filter width
    :param b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    :param padding: string that is either same or valid, indicating the type
    of padding used
    :param stride: tuple of (sh, sw) containing the strides for the
    convolution
        sh is the stride for the height
        sw is the stride for the width
    :return: partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding is 'same':
        ph = int(np.ceil(max((h_prev - 1) * sh + kh - h_prev, 0) / 2))
        pw = int(np.ceil(max((w_prev - 1) * sw + kw - w_prev, 0) / 2))
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    else:
        ph, pw = 0, 0

    dW = np.zeros(W.shape)
    dA_prev = np.zeros(A_prev.shape)
    db = np.zeros(b.shape)
    db[:, :, 0, :] = np.sum(dZ, axis=(0, 1, 2))

    m, h_prev, w_prev, c_prev = A_prev.shape
    for img in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for cn in range(c_new):
                    tmp_W = W[:, :, :, cn]
                    tmp_dZ = dZ[img, i, j, cn]
                    dA_prev[img, i * sh:i * sh + kh, j * sw:j * sw + kw] += \
                        (tmp_W * tmp_dZ)
                    dW[:, :, :, cn] += (A_prev[img, i * sh:i * sh + kh,
                                        j * sw:j * sw + kw] * tmp_dZ)

    dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dW, db
