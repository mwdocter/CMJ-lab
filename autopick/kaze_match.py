# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:33:48 2019

@author: mwdocter

draw matches between keypoints (which apparently include more than just location)
"""
#https://stackoverflow.com/questions/36796025/how-do-you-use-akaze-in-open-cv-on-python
import cv2

def kaze_match(im1_path, im2_path):
    # load the image and convert it to grayscale
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)    

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))    

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3 = cv2.drawMatchesKnn(im1, kps1, im2, kps2, good[1:20], None, flags=2)
    cv2.imshow("AKAZE matching", im3)
    cv2.waitKey(0)