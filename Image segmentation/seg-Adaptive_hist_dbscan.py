""" This file uses adaptive thresholding followed by histogram point removal, DBSCAN outlier detection and lastly finds the minimum rectangle for segmenting the image. (Not being used)"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy
from numpy import matlib

# ---------------------------------------
#region PART-1 Obtaining a binary map of the input image
# Method 1: Adaptive thresholding + fabric pattern convolution

# Read the image
FNAME = '6.jpeg'

img = cv2.imread(FNAME)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Apply adaptive threshold and set grayscale pixels > thresh to 1, otherwise 0. 
# We take a nbhd of size 21. We do not subtract anything from the calculated weighted mean i.e. last argument C = 0.
thresh = cv2.adaptiveThreshold(imgray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)

# Visualizing the adaptive thresholding performed
ret,threshvis=cv2.threshold(thresh,.5, 255, cv2.THRESH_BINARY)
# cv2.imshow('thresh',threshvis)

## Obtaining a filter for the fabric pattern derived from image.
# Assume the center of image always contains fabric (Good assumption). 2*k is length of square fabric pattern we crop from center.
k = 15
h, w = thresh.shape

# The filter is a matrix of -1s and 1s. It is then normalized.
filter = thresh[int(h / 2 - k):int(h / 2 + k), int(w / 2 - k):int(w / 2 + k)]*2-np.ones((2 * k, 2 * k))
filter = filter*255/(4*k*k)     # normalizationo

# The image after adaptive thresholding is a matrix of 0s and 1s. On convoluting with 0 we lose information, thus it is converted to -1s and 1s.
thresh=thresh*2-np.ones(np.shape(thresh))

# Convolve the thresholded image with the extracted fabric filter
filtered_img = cv2.filter2D(thresh, cv2.CV_8UC3, filter)
# cv2.imshow('0-Filtered image (before histogram noise removal)',filtered_img)


## Histogram noise removal

# Histogram of filtered image
hist = cv2.calcHist([filtered_img],[0],None,[255],[int(1),int(255)])

# let NUM_POINTS be the first grayscale value for which number of points is less than 500.
# All points beyond that grayscale value 'm' are what we desire.

m = 0   # m is the desired grayscale value
NUM_POINTS = 500
for i,j in enumerate(hist):
    if j<NUM_POINTS:
        m=i
        break

print('grayscale threshold value for histogram noise removal:', m)
# plt.plot(hist)
# plt.show()

# Saving image before applying threshold 'm'
# cv2.imwrite(FNAME.split('.')[0] + '_before_threshold.' + FNAME.split('.')[1], filtered_img)

# Apply threshold 'm' and save the image
ret, filtered_thresh_img = cv2.threshold(filtered_img, m, 255, cv2.THRESH_BINARY)
# cv2.imwrite(FNAME.split('.')[0] + '_after_threshold.' + FNAME.split('.')[1], filtered_thresh_img)

# Display binary image after histogram noise removal
# cv2.imshow('1-Filtered image (after histogram noise removal)', filtered_thresh_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
#endregion
# ---------------------------------------
#region PART-2 Removing noise from binary map/image
#   region Method-1: DBSCAN

import itertools
from sklearn.cluster import DBSCAN
from scipy import stats

# Finds all points with non-zero grayscale values in binary map
nonzero = cv2.findNonZero(filtered_thresh_img)
# print(nonzero.shape)
nonzero = np.squeeze(nonzero, axis = 1)
# print(nonzero.shape)

# DBSCAN for outlier detection we set MinPts = 1, i.e. every point belongs to a cluster. We then select the largest cluster of points and discard the rest of the points. We only need to determine EPS(epsilon) parameter for DBSCAN.

print('Obtaining KDTree')
kdt = scipy.spatial.cKDTree(nonzero) 
k = 40 # number of nearest neighbors for whom distance we want to determine
dists, neighs = kdt.query(nonzero, k+1)


mean_dist_closest = np.mean(dists[:, 1:], axis=0)
median_dist_closest = np.median(dists[:, 1:], axis=0)

def finding_knee_index(curve):
    """ Finds the index of point which represents the 'knee' of the curve. This is the point which is farthest away from the line joining the end points of a curve. """
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint

# Plot the change in mean/median distances
plt.plot(mean_dist_closest,'b', label = 'mean')
plt.plot(median_dist_closest, 'r', label = 'median')
plt.axvline(x=median_dist_closest[finding_knee_index(median_dist_closest)], color = 'r', linestyle='dashed', label = f'maximum median curvature at k')
plt.axvline(x=mean_dist_closest[finding_knee_index(mean_dist_closest)], color = 'b', linestyle='dashed', label = f'maximum mean curvature at k')
plt.legend(loc = 'lower right')
plt.xlabel('k')
plt.ylabel('Distance to k nearest neighbour')
plt.title('Mean/Median distances (of binary map) to k th nearest point')
plt.show()


# Heuristic: EPS = 2 x (median distance of nearest k points, where k is the knee of the curve)
# Try to find better heuristic using the above plots. Maybe employ both the mean and median.

VALUE = 2*median_dist_closest[finding_knee_index(median_dist_closest)]
print('EPS:', VALUE)

# DBSCAN algorithm
clustering = DBSCAN(eps = VALUE, min_samples=1, n_jobs=-1).fit(nonzero)
labels = clustering.labels_

# Obtain the mode of the list labels. We are finding the label of the largest cluster.
mode_value = stats.mode(labels)[0][0]

# Obtain all the noise points in my_array
print(f'Obtained {sum(labels != mode_value)} noise pointsmout of {len(labels)} using DBSCAN')
my_array = nonzero[labels != mode_value]

# The noise points are set to 0 i.e. black colour, they were initially white in the binary map.
filtered_thresh_img[my_array[:, 1], my_array[:, 0]] = 0

# Display image after DBSCAN noise removal
# cv2.imshow('2-Image(after DBSCAN noise removal before dilating)', filtered_thresh_img)
# cv2.waitKey(0)
#endregion

#   region Method-2: x-y histogram count 
# Plot the histogram of number of nonzero points along x axis and y axis. When the count is suddenly falls to 0, we obtain the edges of the fabric. 
# Note: It cannot be used properly is the fabric is rotated.

def find_edgelines(curve):
    """ Returns the index where the the histogram counts suddenly skyrocket and crash """
    window_size = 10
    i = 0
    l = len(curve)
    # Fine-tune this threshold
    THRESHOLD = sum(curve[l//2-window_size//2:l//2+window_size//2])//10
    print('THRESHOLD for finding edges in histogram:', THRESHOLD)
    # Finds the sudden skyrocket
    while True:
        window_difference = sum(curve[i + window_size//2: i + window_size]) - sum(curve[i:i+window_size//2])
        if window_difference < 10: 
            i += window_size//2
        else:
            i += 1
        if window_difference>THRESHOLD:
            break
    skyrocket = i + window_size//2
    # Finds the sudden crash
    i = len(curve-1)
    while True:
        window_difference = sum(curve[i - window_size: i - window_size//2]) - sum(curve[i-window_size//2 : i])
        if window_difference < 10: 
            i -= window_size//2
        else:
            i -= 1
        if window_difference>THRESHOLD:
            break
    crash = i - window_size//2
    return skyrocket, crash

histogramx = np.sum(filtered_thresh_img, axis = 0)
skyrocketx, crashx = find_edgelines(histogramx)
histogramy = np.sum(filtered_thresh_img, axis = 1)
skyrockety, crashy = find_edgelines(histogramy)

plt.plot(histogramx, label='x-axis hist', color = 'orange')
plt.axvline(x= skyrocketx, color = 'orange', linestyle='dashed', label = f'x-axis edgeline')
plt.axvline(x= crashx, color = 'orange', linestyle='dashed', label = f'x-axis edgeline')

plt.plot(histogramy, label='y-axis hist', color = 'blue')
plt.axvline(x= skyrockety, color = 'b', linestyle='dashed', label = f'y-axis edgeline')
plt.axvline(x= crashy, color = 'b', linestyle='dashed', label = f'y-axis edgeline')
plt.legend(loc = 'upper left')
plt.show()

# Display the bounding box predicted using histogram method on binary map
hist_filtered = filtered_thresh_img.copy()

print(hist_filtered[:skyrocketx, :].shape,hist_filtered[:, :skyrockety].shape )
hist_filtered[:, :skyrocketx] = 0
hist_filtered[:, crashx:] = 0
hist_filtered[:skyrockety, :] = 0
hist_filtered[crashy:, :] = 0
print('Corner point 1:', skyrocketx, skyrockety)
print('Corner point 2:', crashx, crashy)
# cv2.imshow('Histogram refinement on binary map', hist_filtered)
# cv2.waitKey(0)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
#endregion
#endregion
# ---------------------------------------
#region PART-3 Obtaining the bounding box

# Dilate the binary map, to help us obtain the bounding box. The parameters over here are heuristics.


# Find bounding box with DBSCAN only
kernel = np.ones((3, 3),np.uint8)
final_image = cv2.morphologyEx(filtered_thresh_img, cv2.MORPH_DILATE, kernel, iterations=2)

                                # (OR) #

## Find bounding box with DBSCAN + histogram refinement (uncomment below 2 lines)
# kernel = np.ones((5, 5),np.uint8)
# final_image = cv2.morphologyEx(hist_filtered, cv2.MORPH_DILATE, kernel, iterations=2)


# cv2.imshow('3-Image(after dilating)',final_image)

# Obtain the nonzero points for finding the bounding box
nonzero = cv2.findNonZero(final_image)

# Minimum rectangle for image
final_rect = final_image.copy()
rect = cv2.minAreaRect(nonzero)
# corner points of the rectangle
box = cv2.boxPoints(rect)
box = np.int0(box)
print('Corner points of bounding box:\n', box)
# Visualize the rectange on binary map
cv2.drawContours(final_rect, [box], 0, 255)
# cv2.imshow('Minimum rectangle box after DBSCAN', final_rect)

# Visualize the rectange on input image
cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
# cv2.imshow('Final prediction', img)

name = FNAME.split('.')[0]
cv2.imwrite(f'{name}_{VALUE}_{NUM_POINTS}.jpeg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# endregion
# ---------------------------------------
#region Extra bits of code
## Discrete 2nd order derivative for finding knee of a curve
# l = len(mean_dist_closest)
# second_derivative_mean = mean_dist_closest[2:l] + mean_dist_closest[0:l-2] - 2*mean_dist_closest[1:l-1]
# second_derivative_median = median_dist_closest[2:l] + median_dist_closest[0:l-2] - 2*median_dist_closest[1:l-1]
# plt.plot(second_derivative_mean, 'b--', label = '2nd derivative mean')
# plt.plot(second_derivative_median, 'r--', label = '2nd derivative median')
# plt.axvline(x=np.argmax(second_derivative_median) + 1, color = 'green', linestyle='dashed', label = f'maximum median curvature at k = {np.argmax(second_derivative_median) + 1}')
# plt.axvline(x=np.argmax(second_derivative_median) + 1, color = 'yellow', linestyle='dashed', label = f'maximum mean curvature at k = {np.argmax(second_derivative_mean) + 1}')



# # Approximate polygon(requires the figure to be connected, does not work)
# epsilon = 0.01 * cv2.arcLength(nonzero, True)
# approx = cv2.approxPolyDP(nonzero, epsilon, True)
# final_poly = final_image.copy()
# cv2.drawContours(final_poly, [approx], -1, 255, 1)
# cv2.imshow('Approximate polygon', final_poly)

## Convex hull of the nonzero points
# hull = cv2.convexHull(nonzero)
# final_hull = final_image.copy()
# cv2.drawContours(final_hull, [hull], -1, 255, 1)
# cv2.imshow('Convex hull', final_hull)
#endregion