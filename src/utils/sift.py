"""
    Module for SIFT feature extraction
"""
import cv2 as cv


def grey_sift(img):
    """
    Extracts SIFT features from a grayscale image.

    Args:
        img (numpy.ndarray): Image to extract features from.

    Returns:
        numpy.ndarray: SIFT features.
    """
    sift = cv.SIFT_create()

    kp = sift.detect(img, None)
    img = cv.drawKeypoints(
        image=img,
        keypoints=kp,
        outImage=img,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img
