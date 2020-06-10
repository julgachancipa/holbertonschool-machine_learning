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
    print('W ', W.shape)
    print('b ', b.shape)
    print('dZ ', dZ.shape)
    print('A_p ', A_prev.shape)

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    dW = np.zeros(W.shape)

    for i in range(kh):
        for j in range(kw):
            A_p = A_prev[:, i * sh:i * sh + h_new, j * sw:j * sw + w_new]
            aux = A_p * dZ
            dW[i, j] = np.sum(aux, axis=(0, 1, 2))
    db = np.sum(dZ, axis=(0, 1, 2))

    if padding == 'same':
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        ph = int(np.ceil(ph / 2))
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        pw = int(np.ceil(pw / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding[0], padding[1]

    ih = ((h_new - kh + (2 * ph)) // sh) + 1
    iw = ((w_new - kw + (2 * pw)) // sw) + 1

    new_dZ = np.pad(dZ, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    dA_prev = np.zeros((m, ih, iw, c_prev))
    for c in range(c_prev):
        for i in range(ih):
            for j in range(iw):
                dZ = new_dZ[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
                aux = dZ * W[:, :, c, :]
                dA_prev[:, i, j, c] = np.sum(aux, axis=(1, 2, 3))

    # print('>', dA_prev.shape)
    return dA_prev, dW, db
