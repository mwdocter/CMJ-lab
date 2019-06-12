# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:16:25 2019

@author: mwdocter
this code takes an images, calculates the rolling_ball background, and subtracts it
"""
#http://imagejdocu.tudor.lu/doku.php?id=gui:process:subtract_background
#Based on the a „rolling ball“ algorithm described in Stanley Sternberg's article, „Biomedical Image Processing“, IEEE Computer, January 1983. 
import scipy.ndimage as scim
import skimage
from skimage.morphology import ball
#import matplotlib.pyplot as plt

def rollingball(*args): # Matlab[im_out,im_bg]=rollingball(im_in,size_ball,im_bg)
    varargin = args
    im_in=varargin[0]
    if len(varargin)==1:
        size_ball=30
    else:
        size_ball=varargin[1]
        
    if len(varargin)<3:
        # from https://stackoverflow.com/questions/29320954/rolling-ball-background-subtraction-algorithm-for-opencv
        # Create 3D ball structure
        s = ball(size_ball)
        # Take only the upper half of the ball
        h = int((s.shape[1] + 1) / 2)
        # Flat the 3D ball to a weighted 2D disc
        s = s[:h, :, :].sum(axis=0)
        # Rescale weights into 0-255
        s = (255 * (s - s.min())) / (s.max()- s.min())
        ss=s[25:76,25:76]
        ss = (255 * (ss - ss.min())) / (ss.max()- ss.min())
        #im_bg=scim.grey_closing(im_in,structure=ss)
        im_bg=skimage.morphology.opening(im_in,ss)
        #im_out = scim.white_tophat(im, structure=s)
    else:
        im_bg=varargin[2]
        
    im_out=im_in-im_bg #note match 3s dimension im_bg to im_in
    im_out[im_out<0]=0
    
    return im_out, im_bg
    # Use im-opening(im,ball) (i.e. white tophat transform) (see original publication)
    
