"""
    Module with GAC segmentation functions
"""
import numpy as np
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)


def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def gac(img):
    """
    Morphological GAC segmentation function. Takes an image and returns
    a segmentation.

    Args:
        img (numpy.ndarray): Image to be segmented.

    Returns:
        numpy.ndarray: Segmented image.
    """

    gimage = inverse_gaussian_gradient(img)
    # Initial level set
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
                                               smoothing=4, balloon=-1,
                                               threshold=0.69,
                                               iter_callback=callback)

    return ls
