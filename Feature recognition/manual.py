
import cv2
import numpy as np
from datetime import datetime
#from matplotlib import pyplot as plt

fname = 'square2.jpeg'
img = cv2.imread(fname)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(img,100,200)
thresh = cv2.adaptiveThreshold(imgray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 0)

# Visualizing the threshold
ret,threshvis=cv2.threshold(thresh,.5, 255, cv2.THRESH_BINARY)
# cv2.imshow('thresh',threshvis)


## Obtaining a filter derived from fabric in image.
# Assume the center of image always contains fabric. 2*k is length of square fabric pattern we crop from center.
k = 15
h, w = thresh.shape


# The filter is a matrix of -1s and 1s. It is then normalized.
filter = thresh[int(h / 2 - k):int(h / 2 + k), int(w / 2 - k):int(w / 2 + k)]*2-np.ones((2 * k, 2 * k)) 
filter = filter*255/(4*k*k)

# Thresholded image is a matrix of 0s and 1s. On convoluting with 0 we lose information, thus it is converted to -1.
thresh=thresh*2-np.ones(np.shape(thresh))

# Convolution of thresholded image with custom fabric filter
filtered_img = cv2.filter2D(thresh, cv2.CV_8UC3, filter)
# cv2.imshow('filtered image',filtered_img)


## Thresholding the filtered image
# Histogram of filtered image
hist = cv2.calcHist([filtered_img],[0],None,[255],[int(1),int(255)])
for i,j in enumerate(hist):
    if j<500:
        m=i
        break

ret, filter = cv2.threshold(filtered_img, m, 255, cv2.THRESH_BINARY)
kernel = np.ones((5,5), np.uint8)
#filter = cv2.dilate(filter,kernel,iterations = 1)
#filter = cv2.erode(filter,kernel,iterations = 1)
a,b = np.shape(filter)
#print(a,b)

n=50 # number of 
z=0
count = np.zeros(2*n)

t0 = datetime.now()
for i in range(0,a-1,int(a/n)):   #iterate over random n rows
    e =int( i/int(a/n) )   # index of a row(0 - n-1)
    print(i)
    for j in range(1,b-1):
        
        if filter[i,j] == filter[i,j-1] :
            continue
        else:
            count[e] = count[e]+1

    
count = count[(count>10)]  
warp1 = np.max(count) - 1   
warp2 = np.median(count)
#count= np.sort(count,axis=0)  
t1 = datetime.now() - t0 
#len = np.sum(count) / n
        
print(warp1,warp2,t1.total_seconds(),count)

cv2.imshow('i',filter)

cv2.waitKey(0)
cv2.destroyAllWindows() 
