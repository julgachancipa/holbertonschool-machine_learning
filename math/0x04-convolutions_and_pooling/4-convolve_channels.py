#!/usr/bin/env python3
"""
Convolution with Channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels
    :param images: contain multiple images
    :param kernel: contain the kernel for the convolution
    :param padding: padding
    :param stride: tuple of (sh, sw)
    :return: numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        ph = max((h - 1) * sh + kh - h, 0)
        pt = int(np.ceil(ph / 2))
        pb = pt
        pw = max((w - 1) * sw + kw - w, 0)
        pl = int(np.ceil(pw / 2))
        pr = pl
    elif padding == 'valid':
        pt, pb, pl, pr = 0, 0, 0, 0
    else:
        pt, pb = padding[0], padding[0]
        pl, pr = padding[1], padding[1]

    oh = ((h - kh + pt + pb) // sh) + 1
    ow = ((w - kw + pl + pr) // sw) + 1

    images = np.pad(images, pad_width=((0, 0), (pt, pb), (pl, pr), (0, 0)),
                    mode='constant', constant_values=0)

    conv = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            aux = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel
            conv[:, i, j] = np.sum(aux, axis=(1, 2, 3))
    return conv
