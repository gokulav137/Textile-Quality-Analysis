""" This file uses ORB feature detection on the pickglass with the goal of segmenting the fabric inside the pickglass.

Google drive link for labelled images:
https://drive.google.com/drive/folders/1YdQIGYjrGnJRZfsoqLq_Nz0OVfOXcqUh?usp=sharing
 """
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

# Download the export folder from drive link
IMG_DIR = os.getcwd() + '\\export'

SHRINKING_FACTOR = 8
img1 = cv2.imread("pickglass.jpeg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (int(img1.shape[1]//SHRINKING_FACTOR), int(img1.shape[0]//SHRINKING_FACTOR)))
# cv2.imshow('resized image', img1)

#region generate mask
f = "pickglass.xml"
xmlTree = ET.parse(f)
root = xmlTree.getroot()
true_label, all_labels = list(), list()
for polygon in root.iter('polygon'):
    i, temp = 0, []
    for coordinate in polygon.iter():
        if ' ' not in coordinate.text:  # To ignore the <polygon> tag
            temp.append(int(int(coordinate.text)//SHRINKING_FACTOR))
            i += 1
            if i%2 == 0:
                true_label.append(temp)
                temp = []
    all_labels.append(true_label)
    true_label = []

outer_label = np.array(all_labels[0])
inner_label = np.array(all_labels[1])

outer_mask, inner_mask = np.zeros(img1.shape, np.uint8), np.zeros(img1.shape, np.uint8)
cv2.drawContours(outer_mask, [outer_label], -1, 255, -1)
cv2.drawContours(inner_mask, [inner_label], -1, 255, -1)

final_mask = cv2.bitwise_xor(outer_mask, inner_mask)

# cv2.imshow('mask', final_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#endregion


#region ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, final_mask)
NUM_KEY_POINTS = 200

# Loop through images
for file in os.listdir(IMG_DIR):
    if file.endswith(".jpeg"):
        img2 = cv2.imread(os.path.join(IMG_DIR,file), cv2.IMREAD_GRAYSCALE)
        if max(img2.shape)>1000:
            if max(img2.shape)<2400: SHRINKING_FACTOR = 3
            else: SHRINKING_FACTOR = 8
            print('Before resizing:', img2.shape)
            print(SHRINKING_FACTOR)
            img2 = cv2.resize(img2, (int(img2.shape[1]//SHRINKING_FACTOR), int(img2.shape[0]//SHRINKING_FACTOR)))
            print('After resizing:', img2.shape)
        # cv2.imshow('resized image', img2)

        # Create second image mask
        height, width = img2.shape
        upper_left = (int(width / 3), int(height / 3))
        bottom_right = (int(width * 2/3), int(height * 2/3))
        img2_mask = np.zeros(img2.shape, np.uint8)
        img2_mask[:, :] =255
        img2_mask[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = 0
        img2_mask[:int(height/8),:] = 0
        img2_mask[int(height*8/9):, :] = 0
        img2_mask[:, :int(width/8)] = 0
        img2_mask[:, int(width*8/9):] = 0
        # print(img2_mask.shape)
        # cv2.imshow('Image1-pickglass mask', final_mask)
        cv2.imshow('Image2 mask', img2_mask)

        kp2, des2 = orb.detectAndCompute(img2, img2_mask)

        # Brute Force Matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:NUM_KEY_POINTS], None, flags=2)

        key_point_map = np.zeros(img2.shape, np.uint8)
        for m in matches[:NUM_KEY_POINTS]:
            # print(kp2[m.trainIdx].pt, kp1[m.queryIdx].pt)
            nonzero_pt = [int(i) for i in kp2[m.trainIdx].pt]
            key_point_map[nonzero_pt[1], nonzero_pt[0]] = 255
        # cv2.imshow("Img1-pickglass template", img1)
        # cv2.imshow("Img2-example", img2)
        cv2.imshow("Matching result", matching_result)
        key_point_map = cv2.dilate(key_point_map, np.ones((5, 5)))
        cv2.imshow("Keypoint map", key_point_map)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()
        #endregion