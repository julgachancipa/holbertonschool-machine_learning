#!/usr/bin/env python3
"""
Performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    :param images: contain multiple images
    :param kernel_shape: contain the kernel shape for the pooling
    :param stride: tuple of (sh, sw)
    :param mode: type of pooling
    :return: numpy.ndarray containing the pooled images
    """
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    hc = (ih - kh)
    wc = (iw - kw)

    conv = np.zeros((m, (hc // stride[0]) + 1, (wc // stride[1]) + 1, c))
    i = 0
    for h in range(0, hc+1, stride[0]):
        j = 0
        for w in range(0, wc+1, stride[1]):
            if mode is 'max':
                aux = images[:, h:h + kh, w:w + kw].max
            elif mode is 'avg':
                aux = np.reshape(images[:, h:h + kh, w:w + kw, :],
                                 (m, kw*kh, c))
                aux = np.sum(aux, axis=1) / (kh * kw)
            conv[:, i, j, :] = aux
            j += 1
        i += 1
    return conv
