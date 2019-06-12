# -*- coding: utf-8 -*-
"""
Created on Jun 2019
@author: carlos

some features are shared within an image collection, like the file name, mapping, background, threshold, ...)
so imagecollection is a class ;)
"""
from load_file import read_one_page#_pma, read_one_page_tif
from image_adapt.rolling_ball import rollingball

from image_adapt.find_threshold import remove_background
from image_adapt.find_threshold import get_threshold
import numpy as np
from cached_property import cached_property
from Mapping import Mapping
from Image import Image
import analyze_label
import cv2
import os
from find_xy_position.Gaussian import makeGaussian

class ImageCollection(object):
    def __init__(self, tetra_fn, image_fn):
        self.image_fn = image_fn
        self.mapping = Mapping(tetra_fn)
        #self.set_background_and_transformation() # Carlos: isn't this double, you call set_background twice, Margreet: this is the bg of the image, not the tetra_image. Same formulas though
        (self.background,
         self.threshold,
         self.pts_number,
         self.dstG,
         self.ptsG,
         self.im_mean20_correct,
         self.n_images,
         self.hdim,
         self.Gauss) = self.set_background_and_transformation()

#    @cached_property
#    def read_one_page(self):
#        return read_one_page ##normally this funciton needs two inputs, why not here?
#        if '.pma' in self.image_fn:
#            return read_one_page_pma
#        elif '.tif' in self.image_fn:
#            return read_one_page_tif

    def set_background_and_transformation(self):
        """
        sum 20 image, find spots& background. Then loop over all image, do background subtract+ extract traces
        :return:
        """
        _, hdim, vdim, n_images = read_one_page(self.image_fn, pageNb=0)
        im_array = np.dstack([(read_one_page(self.image_fn, pageNb=ii)[0]).astype(float) for ii in range(20)])
        im_mean20 = np.mean(im_array, axis=2).astype(int)
        bg = rollingball(im_mean20)[1]
        im_mean20_correct = im_mean20 - bg
        im_mean20_correct[im_mean20_correct < 0] = 0       
        threshold = get_threshold(im_mean20_correct)
        im_mean20_correct=remove_background(im_mean20_correct,threshold)        

        
        #note: optionally a fixed threshold can be set, like with IDL
        # note 2: do we need a different threshold for donor and acceptor?
        
        root, name = os.path.split(self.image_fn)
        pks_fn=os.path.join(root,name[:-4]+'-P.pks') 
        if os.path.isfile(pks_fn):
             ptsG=[]
             dstG=[]
             with open(pks_fn, 'r') as infile:
                 for jj in range(0,10000):
                     A=infile.readline()
                     if A=='':
                         break
                     ptsG.append([float(A.split()[1]),float(A.split()[2])])
                     A=infile.readline()
                     dstG.append([float(A.split()[1]),float(A.split()[2])])
             ptsG=np.array(ptsG)
             dstG=np.array(dstG)
             pts_number =len(ptsG)
            # load
        else:
            pts_number, label_size, ptsG = analyze_label.analyze(im_mean20_correct[:, 0:int(vdim / 2)])
            # there should be different options:
    #        donor: im_mean20_correct[:,0:vdim//2]
    #        acceptor: im_mean20_correct[:,vdim//2:]
    #        donor+acceptor
            dstG = cv2.perspectiveTransform(ptsG.reshape(-1, 1, 2),
                                            np.linalg.inv(self.mapping._tf2_matrix))#transform_matrix))
            dstG = dstG.reshape(-1, 2)
            dstG = np.array([[ii[0] + 256, ii[1]] for ii in dstG])
        
                #saving to pks file
            with open(pks_fn, 'w') as outfile:
                 for jj in range(0,pts_number):
                     pix0=ptsG[jj][0]
                     pix1=ptsG[jj][1]
                     outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format((jj*2)+1, pix0, pix1, 0, 0, width4=4, width6=6))
                     pix0=dstG[jj][0]
                     pix1=dstG[jj][1]
                     outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format((jj*2)+2, pix0, pix1, 0, 0, width4=4, width6=6))
        
#        ALL_Gaussians_ptsG=np.zeros((11,11,pts_number))            
#        ALL_Gaussians_dstG=np.zeros((11,11,pts_number))  
        ALL_GAUSS=makeGaussian(11, fwhm=3, center=(5, 5))          
#        for jj in range(0,pts_number):
#            xpix = ptsG[jj][1]
#            ypix = ptsG[jj][0]
#
#            xpix_int = int(xpix)
#            ypix_int = int(ypix)
#            ALL_Gaussians_ptsG[:,:,jj]=makeGaussian(11, fwhm=3, center=(ypix - ypix_int + 5, xpix - xpix_int + 5))
#            
#            xf2 = dstG[jj][1]  # approach3
#            yf2 = dstG[jj][0]  # approach3
#            xf2_int = int(xf2)  # approach3
#            yf2_int = int(yf2)  # approach3
#            ALL_Gaussians_dstG[:,:,jj]= makeGaussian(11, fwhm=3, center=(yf2 - yf2_int + 5, xf2 - xf2_int + 5))  # approach3
#            
        return bg, threshold, pts_number, dstG, ptsG, im_mean20_correct, n_images, hdim, ALL_GAUSS

    def subtract_background(self, im):
        im_correct = im - self.background
        im_correct[im_correct < 0] = 0
        return remove_background(im_correct, self.threshold)

    def get_image(self, idx):
        img, hdim, vdim, n_images = read_one_page(self.image_fn, pageNb=idx)
        img = self.subtract_background(img)
        return Image(img, vdim, self.mapping._tf2_matrix, self.ptsG, self.dstG, self.pts_number, self.Gauss)
