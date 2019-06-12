# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:04:10 2019

@author: mwdocter

pts1&2 are found, as well as the transformation between them, to be adapted
"""

from do_before import clear_all
clear_all()
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

image_tetra = tiff.imread('E://CMJ trace analysis/autopick/tetraspeck.tif')
    
    # default for 16 bits 50000, for 8 bits 200 (=256*50000/64000)
if np.max(image_tetra)>256:
    f=50000
else:
    f=200
    
    # left, right, enhanced left and enhanced right image for keypoint detection
l, r, l_enh, r_enh = enhance_blobies(image_tetra,f)
    
gray1 = l_enh
gray2 = r_enh 
    
    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
detector = cv2.AKAZE_create()

(kps1, descs1) = detector.detectAndCompute(gray1, None);
(kps2, descs2) = detector.detectAndCompute(gray2, None);
    
print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))    
    
    # Match the features
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed
    
    # Apply ratio test
pts1, pts2 = [], []
for m in matches:
    pts1.append(kps1[m[0].queryIdx].pt)
    pts2.append(kps2[m[0].trainIdx].pt)
        
pts1 = np.array(pts1).astype(np.float32) # xy position
pts2 = np.array(pts2).astype(np.float32)
#AA=cv2.KeyPoint_convert(kps1);
        
transformation_matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC,20)