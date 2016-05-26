import numpy as np
import cv2
import sys

features_images = np.load(sys.argv[1])

for features_image in features_images:
    image = np.reshape(features_image, (28, 28)) * 255
    cv2.imshow('win', image)
    cv2.waitKey(0)

