#!/usr/bin/env python3
"""
Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images
    :param images: contain multiple grayscale images
    :param kernel: contain the kernel for the convolution
    :param padding: either a tuple of (ph, pw), ‘same’, or ‘valid’
    :param stride: tuple of (sh, sw)
    :return:
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        # ph = int(((h - 1) * sh + kh - h) / 2) + 1
        # pw = int(((w - 1) * sw + kw - w) / 2) + 1
        ph = max((h - 1) * stride[0] + kh - h, 0)
        pt = ph // 2
        pb = ph - pt
        pw = max((w - 1) * stride[1] + kw - w, 0)
        pl = pw // 2
        pr = pw - pl
    elif padding == 'valid':
        pt, pb, pl, pr = 0, 0, 0, 0
        # ph = 0
        # pw = 0
    else:
        # ph = padding[0]
        # pw = padding[1]
        pt, pb = padding[0], padding[0]
        pl, pr = padding[1], padding[1]

    oh = ((h - kh + pt + pb) // sh) + 1
    ow = ((w - kw + pl + pr) // sw) + 1

    images = np.pad(images, pad_width=((0, 0), (pt, pb), (pl, pr)),
                    mode='constant', constant_values=0)

    conv = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            aux = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel
            conv[:, i, j] = np.sum(aux, axis=(1, 2))
    return conv
