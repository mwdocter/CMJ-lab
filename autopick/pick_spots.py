# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:08:00 2019

@author: mwdocter

find spots and make transformation, see Shirani's report
"""
# # NOTES # #
# Install the packages opencv and opencv-contrib.
# Both versions of OpenCV have to be lower than 3.4.3.
# This is because SIFT and SURF algorithms are both patented and removed from newer versions
# Range of uint16: [0, 65535]
# code via Shirani Bisnajak (BEP 2018-2019)

import cv2 #computer vision?
from PIL import Image # Python Imaging Library (PIL)
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters #  Image processing in Python — scikit-image
import numpy as np
import bisect #This module provides support for maintaining a list in sorted order without having to sort the list after each insertion.

def imghist(img): #img is a np.array
    binrange = [np.min(img), np.max(img)]
    binlength = binrange[1] - binrange[0]
    hist,bins = np.histogram(img.flatten(),binlength, binrange) #img.flatten changes RGB into one channel
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(), binlength, binrange, color = 'r')
    plt.xlim(binrange)
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    
def imadjust(src, tol=1, vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    vin = [np.min(src), np.max(src)]
    vout = [0, 65535] # 65535=16 bits
    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(vin[1] - vin[0])),range=tuple(vin))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(0, vin[1]-vin[0]-1): cum[i] = cum[i - 1] + hist[i] # why not hist.cumsum() here?

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0 #everything below zero becomes 0
    vd = vs*scale+0.5 + vout[0] # why +0.5?
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst.astype(np.uint16)


def im_binarize(img, f):
    temp = img.copy()
    temp[temp<f] = 0
    return temp.astype(np.uint8)

def enhance_blobies(image, f):
    l, r = image[:, :image.shape[1]//2], image[:, image.shape[1]//2:]
    l_adj, r_adj = imadjust(l.copy()), imadjust(r.copy())
    l_bin, r_bin = im_binarize(l_adj, f).astype(np.uint8), im_binarize(r_adj,f).astype(np.uint8)
    return l, r, l_bin, r_bin

# Open image
image = tiff.imread('tetraspeck.tif')

# default for 16 bits 50000, for 8 bits 200 (=256*50000/64000)
f=50000

# left, right, enhanced left and enhanced right image for keypoint detection
l, r, l_enh, r_enh = enhance_blobies(image,f)

### Initiate SIFT or SURF detector
# detector = cv2.xfeatures2d.SIFT_create()
detector = cv2.xfeatures2d.SURF_create()

# Find the keypoints and descriptors with SIFT or SURF
kp1, des1 = detector.detectAndCompute(l_enh,None)
kp2, des2 = detector.detectAndCompute(r_enh,None)

# # BFMatcher with default params # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2) #k=2 means it returns the two best matches
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        
print("Keypoints found: {}, {}".format(len(kp1),len(kp2)))
print("Good matches found: {}".format(len(good)))

# cv.drawMatchesKnn expects list of lists as matches.
matches = cv2.drawMatchesKnn(l_enh, kp1, r_enh, kp2, good, outImg=None, flags=2)
plt.figure(num=None, figsize=(15, 15), dpi=100)
plt.imshow(matches)
plt.show()

pts1, pts2 = [], []
for m in good:
    pts1.append(kp1[m[0].queryIdx].pt)
    pts2.append(kp2[m[0].trainIdx].pt)

pts1 = np.array(pts1).astype(np.float32)
pts2 = np.array(pts2).astype(np.float32)

tranformation_matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
print("Transformation matrix is:\n")
print(tranformation_matrix)