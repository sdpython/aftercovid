"""
Loads grid images.
"""
import os
import numpy
import skimage.io


def load_grid_image(name='france_bin.bmp'):
    """
    Loads a black and white picture and makes a grid.
    :param name: picture name
    """
    if not os.path.exists(name):
        this = os.path.dirname(__file__)
        name2 = os.path.join(this, name)
        if os.path.exists(name2):
            name = name2

    img = skimage.io.imread(name, as_gray=True)
    img //= 255
    img = 1 - img
    return img.astype(numpy.int32)
