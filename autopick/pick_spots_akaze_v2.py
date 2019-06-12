# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:08:00 2019

@author: mwdocter

find spots and make transformation, see Shirani's report
version 2
"""
# # NOTES # #
# Install the packages opencv and opencv-contrib.
# Both versions of OpenCV have to be lower than 3.4.3.
# This is because SIFT and SURF algorithms are both patented and removed from newer versions
# Range of uint16: [0, 65535]
# code via Shirani Bisnajak (BEP 2018-2019)

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
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

def mapping(file_tetra='tetraspeck.tif'): #'E:\CMJ trace analysis\\autopick\\tetraspeck.tif'
    # Open image
    image_tetra = tiff.imread(file_tetra)
    
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
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)
    position1=cv2.KeyPoint_convert(kps1);
    position2=cv2.KeyPoint_convert(kps2);
    
    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))    
    
    # Match the features

    if 1:
         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
         matches = bf.match(descs1,descs2)
         matches = sorted(matches, key = lambda x:x.distance)
         # these matches are sorted to the distance you get from bf.match
         # I want to sort them with respect to their xy distance
         
         RR_dist=[]
         MMatch=[]
         for ii in range(len(matches)):
             RR_dist.append([np.linalg.norm(position1[ii]-position2[ii]), (position1[ii][0]-position2[ii][0]),(position1[ii][1]-position2[ii][1])])
             MMatch.append([ matches[ii], RR_dist[ii][0] , position1[ii][:],position2[ii][:] ])
#             print(matches[ii].distance, position1[ii],position2[ii],RR_dist[ii])
         Mastermatches = sorted(MMatch, key = lambda x:x[1]) 
         N,M = np.shape(Mastermatches) 
#mRR = [ [Mastermatches[row][column] for row in range(N)] for column in range(M) ]
         mmatches=[Mastermatches[row][0] for row in range(N)]# np.array(Mastermatches)[...,0]
         mRR=[Mastermatches[row][1] for row in range(N)]
         mdX=[(Mastermatches[row][2][0]-Mastermatches[row][3][0]) for row in range(N)]
         mdY=[(Mastermatches[row][2][1]-Mastermatches[row][3][1]) for row in range(N)]
         mX=[Mastermatches[row][2][:].tolist() for row in range(N)]
         mY=[Mastermatches[row][3][:].tolist() for row in range(N)]
         
# check out histogram of RR_dist
         plt.figure(5)
        
         plt.subplot(1,3,1)
         plt.hist(mRR,bins=20)
         plt.title('median={:6.0f}, \nmean={:6.0f}'.format(np.median(mRR),np.mean(mRR)))
         plt.subplot(1,3,2)
         plt.hist(mdX,bins=20)
         plt.title('median={:6.0f}, \nmean={:6.0f}'.format(np.median(mdX),np.mean(mdX)))
         plt.subplot(1,3,3)
         plt.hist(mdY,bins=20)
         plt.title('median={:6.0f}, \nmean={:6.0f}'.format(np.median(mdY),np.mean(mdY)))
         plt.show() 
         Mastermatches = sorted(MMatch, key = lambda x:x[1]) 
         mmatches=np.array(Mastermatches)[...,0]
         
         im3 = cv2.drawMatches(gray1, kps1, gray2, kps2,mmatches[0:90], None, flags=2)
         plt.figure(1)
         plt.imshow(im3)
         plt.show

         pts1 = np.array(mX).astype(np.float32)
         pts2 = np.array(mY).astype(np.float32)
         transformation_matrixA, mask = cv2.findHomography(pts2,pts1, cv2.RANSAC,20)
         
         mX_select,mY_select=[],[]
         dd=3
         for ii in range(len(RR_dist)):
             if (mdX[ii]>np.median(mdX)-dd) and (mdX[ii]<np.median(mdX)+dd) and (mdY[ii]>np.median(mdY)-dd) and (mdY[ii]<np.median(mdY)+dd): 
                 mX_select.append(Mastermatches[ii][2][:])
                 mY_select.append(Mastermatches[ii][3][:])
         transformation_matrixB, mask = cv2.findHomography(np.array(mY_select), np.array(mX_select), cv2.RANSAC,20)        
         
    if 1: #this part is working porperly according to the overlayed images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
        matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed
        # Apply ratio test
        pts1, pts2 = [], []
        size_im=np.shape(gray1)
        heigh,widt=np.shape(gray1)
        
        for m in matches: # BF Matcher already found the matched pairs, no need to double check them
            pts1.append(kps1[m[0].queryIdx].pt)
            pts2.append(kps2[m[0].trainIdx].pt)
#        for m,n in matches: # BF Matcher already found the matched pairs, no need to double check them
#            if m.distance<0.75*n.distance:
#                print(m.distance)
#                pts1.append(kps1[m[0].queryIdx].pt)
#                pts2.append(kps2[m[0].trainIdx].pt)
                 
        pts1 = np.array(pts1).astype(np.float32)
        pts2 = np.array(pts2).astype(np.float32)
            
        transformation_matrixC, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC,20)
              
        # cv2.drawMatchesKnn expects list of lists as matches.
        A=pts1[0:len(matches) : int(len(matches)/15)]
        im3 = cv2.drawMatchesKnn(gray1, kps1, gray2, kps2,matches[1:20] , None, flags=2)
        #cv2.imshow("AKAZE matching", im3)
        #cv2.waitKey(1000)
        #cv2.destroyAllWindows()
        plt.figure(1)
        plt.imshow(im3)
        plt.show
    
    
    # produce an image in which the overlay between two channels is shown
    array_size=np.shape(gray2)
    imA=cv2.warpPerspective(gray2, transformation_matrixA, array_size[::-1] )
    imB=cv2.warpPerspective(gray2, transformation_matrixB, array_size[::-1] )
    imC=cv2.warpPerspective(gray2, transformation_matrixC, array_size[::-1] )
    
    #cv2.imshow("transformed ", im4)
    plt.figure(2)
    plt.subplot(1,3,1),
    plt.imshow(gray1, extent=[0,array_size[1],0,array_size[0]], aspect=1)
        
    plt.subplot(1,3,2),
    plt.imshow(gray2, extent=[0,array_size[1],0,array_size[0]], aspect=1)
        
    plt.subplot(1,3,3),
    plt.imshow(imC, extent=[0,array_size[1],0,array_size[0]], aspect=1)
    plt.show()
    
    plt.figure(3)
    plt.subplot(1,1,1)
    plt.subplot(1,4,1),
    plt.imshow((gray1>0)+2*(gray2>0), extent=[0,array_size[1],0,array_size[0]], aspect=1)
    plt.colorbar()
        
    plt.subplot(1,4,2),
    plt.imshow((gray1>0)+2*(imA>0), extent=[0,array_size[1],0,array_size[0]], aspect=1)
    plt.colorbar()
    plt.show()
    
    plt.subplot(1,4,3),
    plt.imshow((gray1>0)+2*(imB>0), extent=[0,array_size[1],0,array_size[0]], aspect=1)
    plt.colorbar()
    plt.show()
    
    plt.subplot(1,4,4),
    plt.imshow((gray1>0)+2*(imC>0), extent=[0,array_size[1],0,array_size[0]], aspect=1)
    plt.colorbar()
    plt.show()

    bestZERO=sum(sum((gray1>0)&(gray2>0)))
    bestA=sum(sum((gray1>0)&(imA>0)))
    bestB=sum(sum((gray1>0)&(imB>0)))
    bestC=sum(sum((gray1>0)&(imC>0)))
    print([bestZERO,bestA,bestB,bestC])
    return transformation_matrixA, transformation_matrixB, transformation_matrixC, plt