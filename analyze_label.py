# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:50:42 2019

@author: https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python

returns the number of spots, and per spot its centroid and the number of pixels (can be used to discard too large spots)
"""
import cv2
import numpy as np

def analyze(src):
    ret, thresh = cv2.threshold(src.astype(np.uint8),0,1,cv2.THRESH_BINARY)
    
    # You need to choose 4 or 8 for connectivity type
    connectivity = 4  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    
    size_label=np.zeros(num_labels)
    for jj in range(num_labels):
        size_label[jj]=np.sum(labels==jj)
    
    # The third cell is the stat matrix
    #stats = output[2]
    # The fourth cell is the centroid matrix
    ctrd = output[3]
    
#    for ii in num_labels:
#        LL=labels.copy()
#        LL[LL!=ii]=0
#        xx,yy=np.where(LL>0)
#        centroids[ii,0]=np.mean(xx)
#        centroids[ii,1]=np.mean(yy)
    #apparentely centroids is already np.mean
    
    # remove all pixels at the edge (within 10 pix) 
    hdim,vdim=np.shape(src)
    for ii in range(num_labels-1,-1,-1):
        discard=ctrd[ii,0]<10 or ctrd[ii,1]<10 or ctrd[ii,0]>hdim/2-10 or ctrd[ii,1]>vdim-10 or size_label[ii]>100
        if discard:
            ctrd=np.delete(ctrd,ii, axis=0)
            size_label=np.delete(size_label,ii, axis=0)
    num_labels=   len(ctrd) 

      
    return num_labels,size_label, ctrd