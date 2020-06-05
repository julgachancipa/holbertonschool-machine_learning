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
        oh = int(np.ceil(h/sh))
        ow = int(np.ceil(w/sw))
        ph = max((oh - 1) * sh + kh - h, 0)
        pw = max((ow - 1) * sw + kw - w, 0)
        ph = int(np.floor(ph/2))
        pw = int(np.floor(pw/2))
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
        hc = int(np.ceil(h / sh))
        wc = int(np.ceil(w / sw))

    elif padding == 'valid':
        hc = (h - kh + 1) // sh
        wc = (w - kw + 1) // sw

    else:
        ph = padding[0]
        pw = padding[1]
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                        mode='constant', constant_values=0)
        hc = ((h - kh + 2 * ph) // sh) + 1
        wc = ((w - kw + 2 * pw) // sw) + 1

    conv = np.zeros((m, hc, wc))
    row = 0
    for i in range(0, h - kh + 1, sh):
        col = 0
        for j in range(0, w - kw + 1, sw):
            aux = np.multiply(images[:, i:i + kh, j:j + kw], kernel)
            conv[:, row, col] = np.sum(aux, axis=(1, 2))
            col += 1
        row += 1
    return conv
