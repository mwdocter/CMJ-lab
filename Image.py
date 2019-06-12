# -*- coding: utf-8 -*-
"""
Created on Jun 2019
@author: carlos

for each image inside an image collection you can process the image (extract trace values)
"""
from find_xy_position.Gaussian import makeGaussian
import cv2

import numpy as np

class Image(object):
    def __init__(self, im, vdim, tm, ptsG, dstG, pts_number,Gauss):
        self.tm = tm
        self.vdim = vdim
        self.image = im
        self.ptsG = ptsG
        self.dstG = dstG
        self.pts_number = pts_number
        self.gauss=Gauss
        self.process_image()
        

    def process_image(self): #extract traces
        IM_donor = self.image[:, 0:int(self.vdim / 2)]
        IM_acceptor = self.image[:, int(self.vdim / 2):]
        array_size = np.shape(IM_acceptor)
#        imA = cv2.warpPerspective(IM_acceptor.astype(float), self.tm, array_size[::-1])

        donor = np.zeros(self.pts_number)
        acceptor = np.zeros(self.pts_number)

        for jj in range(0, self.pts_number):
            xpix = self.ptsG[jj][1]
            ypix = self.ptsG[jj][0]

            xpix_int = int(xpix)
            ypix_int = int(ypix)
            # first crop around spot, then do multiplication
            impixD =  IM_donor[(xpix_int - 5): (xpix_int + 6), (ypix_int - 5): (ypix_int + 6)]
            GG = self.gauss #makeGaussian(11, fwhm=3, center=(ypix - ypix_int + 5, xpix - xpix_int + 5))
            multipD = impixD * GG
            donor[jj] = np.sum(multipD)

            xf2 = self.dstG[jj][1]  # approach3
            yf2 = self.dstG[jj][0]  # approach3
            xf2_int = int(xf2)  # approach3
            yf2_int = int(yf2)  # approach3

            impixC = self.image[(xf2_int - 5): (xf2_int + 6), (yf2_int - 5): (yf2_int + 6)]
            GGC = self.gauss#GGC = makeGaussian(11, fwhm=3, center=(yf2 - yf2_int + 5, xf2 - xf2_int + 5))  # approach3
            multipC = impixC * GGC  # approach3
            acceptor[jj] = np.sum(multipC)
        
        self.acceptor=acceptor ## added by margreet
        self.donor=donor