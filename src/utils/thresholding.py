"""
    Functions based on thresholding of images.
"""
import numpy as np
import matplotlib.pyplot as plt


def thresholding(img, threshold):
    """
    Thresholding function. Takes an image and a threshold value 
    and returns a binary image. Values above the threshold are
    set to 1 and values below are set to 0.

    Args:
        img (numpy.ndarray): Image to be thresholded.
        threshold (int): Threshold value.

    Returns:
        numpy.ndarray: Thresholded image.
    """
    return np.where(img >= threshold, 1, 0)

def plot_thresholding(img, threshold):
    """
    Plots the original image and the thresholded image side by side.

    Args:
        img (numpy.ndarray): Image to be thresholded.
        threshold (int): Threshold value.
    """

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(thresholding(img, threshold), cmap="gray")
    ax1.set_title("Thresholded image")
    ax1.axis('off')
    ax2 = fig.add_subplot(122)
    ax2.imshow(img, cmap="gray")
    ax2.set_title("Original image")
    ax2.axis('off')
    plt.show()
