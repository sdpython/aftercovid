"""
Loads grid images.
"""
import numpy
from skimage.transform import resize


def reduce_image(img, new_size):
    """
    Reduces an image.
    :param name: numpy array
    :param new_size: new_size
    :return: new image
    """
    img = resize(img.astype(numpy.float64), new_size,
                 mode='constant', anti_aliasing=True)
    img = img >= 0.5
    return img.astype(numpy.int32)
