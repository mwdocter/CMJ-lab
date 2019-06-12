# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:42:16 2019

@author: mwdocter

compare warpperspective with perspective transform --> same
"""

import numpy as np

import matplotlib.pyplot as plt

import cv2
import autopick.pick_spots_akaze_final

file_tetra='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\mapping\\rough_ave.tif'
transform_matrix=autopick.pick_spots_akaze_final.mapping(file_tetra,show=1,bg=None, tol=0,f=10000)[0]


# make image with spot
imDonor=np.zeros((50,50))
#im[24:27,24:27]=1
imDonor[20:23,10:13]=1
#im[4:7,10:13]=1

# find location spot
M = cv2.moments(imDonor)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
ptsG=np.array([np.float(cx),np.float(cy)])

#transform this image
if 0:
    TM=np.zeros((3,3))
    TM[0,0]=1
    TM[1,1]=1
    TM[2,2]=1
    TM[0,2]=5
    TM[1,2]=-3
else: #real values
#    TM=([[ 1.00079718e+00, -3.87412160e-03,  3.10086596e+00],
#       [-7.02025017e-03,  9.99395388e-01,  2.87317236e+00],
#       [-2.85380283e-07,  5.84705682e-07,  1.00000000e+00]])
    TM=np.array([ [1.,0,0],
        [0,1.,0],
        [0,0,1.] ])

array_size=np.shape(im)
imAcceptor=cv2.warpPerspective(imDonor.astype(float), TM,array_size[::-1])


#imALL=np.zeros((50,100))
#imALL[:,0:50]=im
#imALL[:,50:]=imT
#imALL=imALL*300

#find location transformed spot
MT = cv2.moments(imT)
cxT = int(MT['m10']/MT['m00'])
cyT = int(MT['m01']/MT['m00'])

#transform spot location
dstG = cv2.perspectiveTransform(ptsG.reshape(-1,1,2),TM) #reshape needed for persp.transform
dstG= dstG.reshape(-1,2) #reshape needed to bring them back to 2D

print([cx,cy],[cxT,cyT],dstG)
plt.subplot(1,3,1), plt.imshow(im)
plt.subplot(1,3,2), plt.imshow(imT)

