""" This file explores template matching for the segmentation and FR tasks.

Google drive link for labelled images:
https://drive.google.com/drive/folders/1YdQIGYjrGnJRZfsoqLq_Nz0OVfOXcqUh?usp=sharing
 """
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# Download the export folder from drive link
IMG_DIR = os.getcwd() + '\\export'

########################################
# PART-1 Obtaining a binary map of the image

for file in os.listdir(IMG_DIR):
    if file.endswith(".jpeg"):
        img2 = cv2.imread(os.path.join(IMG_DIR,file))
        if max(img2.shape)>1000:
            if max(img2.shape)<2400: SHRINKING_FACTOR = 2
            else: SHRINKING_FACTOR = 6
            # print('Before resizing:', img2.shape)
            img2 = cv2.resize(img2, (int(img2.shape[1]//SHRINKING_FACTOR), int(img2.shape[0]//SHRINKING_FACTOR)))
            # print('After resizing:', img2.shape)

        # Read the image
        img = img2.copy()
        # Assume the center of image always contains fabric. 2*k is length of square fabric pattern we crop from center. On increasing k i.e. size of pattern we get fewer matches. 
        k = 15

        h, w,_ = img.shape
        template = img[int(h / 2 - k):int(h / 2 + k), int(w / 2 - k):int(w / 2 + k), :]

        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        # Threshold above which we consider a positive result. On increasing threshold we get fewer matches. We need to tune k (pattern size) and threshold to get good results.
        THRESHOLD = 0.3
        loc = np.where(result>=THRESHOLD)
        # print(loc)

        # Construct a new black image. On this image add white points at template matches.
        binary_map = np.zeros_like(result)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(binary_map, pt, (pt[0]+1, pt[1]+1), (255, 255, 255), -1)

        cv2.imshow("Image", img)

        # Erosion
        kernel = np.ones((3,3),np.uint8)
        binary_map = cv2.erode(binary_map, kernel,iterations = 1)

        # Visualization of the binary template map
        cv2.imshow("Template map"+f'_thresh{THRESHOLD}', binary_map)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()


