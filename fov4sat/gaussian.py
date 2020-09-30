"""Module for adding Gaussian blobs to heatmap"""
import math
import numpy as np


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, sigma, k=1, mode='max'):
    diameter = sigma * 6
    radius = int(((diameter - 1) / 2))
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        if mode == 'max':
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        elif mode == 'sum':
            np.add(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
