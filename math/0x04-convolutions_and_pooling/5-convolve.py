#!/usr/bin/env python3
"""
Multiple Kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels
    :param images: contain multiple images
    :param kernels: contain the kernels for the convolution
    :param padding: padding
    :param stride: tuple of (sh, sw)
    :return: numpy.ndarray containing the convolved images
    """
    if padding == 'same':
        images = np.pad(images, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)),
                        mode='constant', constant_values=0)
    elif padding != 'valid':
        ph = padding[0]
        pw = padding[1]
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant', constant_values=0)
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    c = images.shape[3]
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nk = kernels.shape[2]
    hc = (ih - kh + 1)
    wc = (iw - kw + 1)

    conv = np.zeros((m, hc // stride[0], wc // stride[1], nk))

    i = 0
    for h in range(0, hc, stride[0]):
        j = 0
        for w in range(0, wc, stride[1]):
            for k in range(nk):
                aux = np.multiply(images[:, h:h + kh, w:w + kw],
                                  kernels[:, :, :, k])
                aux = np.reshape(aux, (m, kh * kw * c))
                conv[:, i, j, k] = np.sum(aux, axis=1)
            j += 1
        i += 1
    return conv
