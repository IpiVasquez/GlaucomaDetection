"""Form descriptors"""
import cv2
import numpy as np


def form_descriptors(img):
    """Calculates form descriptors.

    1. Perimeter
    2. Area
    3. Compacity
    4. X centroid
    5. Y centroid
    """
    features = np.zeros(5)
    # Perimeter
    cnts, _ = cv2.findContours(img, 1, 2)
    features[0] = cnts[0].sum()
    # Area
    features[1] = img.sum()
    # Compacity
    features[2] = features[0] ** 2 / features[1]
    # X & Y centroid
    M = cv2.moments(img)
    features[3] = int(M['m10'] / M['m00'])
    features[4] = int(M['m01'] / M['m00'])

    return features
