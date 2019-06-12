# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:29:27 2019

@author: ivoseverins
"""
import os
# Images to locations
os.chdir(r'D:\ivoseverins\SURFdrive\20199999 - Computer\Scan')

import imageio
import cv2
import numpy as np
import re

# %%
def findClusterCenters(imPath):
    img = (imageio.imread(imPath)*4).astype('uint8')
    
    #plt.imshow(img)
    #plt.show()

    pngFileName = re.sub('.bmp', '.png', imPath)
    cv2.imwrite(pngFileName, img)
    
    #cv2.imshow("Image1", img)
    ret,thresh = cv2.threshold(img,76,255,cv2.THRESH_BINARY)
    
    #cv2.imshow("Image2", thresh)
        
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x=[]
    y=[]

    colorImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
       
        # calculate x,y coordinate of center
    
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            x = np.append(x,cX)
            y = np.append(y,cY)
        else:
            cX, cY = 0, 0
        
        
        locFileName = re.sub('.bmp','.loc',imPath)
        np.savetxt(locFileName, np.transpose([x,y]), fmt='%u', delimiter='\t')


        #cv2.imshow("test",colorImg); cv2.waitKey(30)

        cv2.circle(colorImg, (cX, cY), 8, (0, 0, 255), thickness=1)
        #cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #display the image
    cv2.imshow("Image3", colorImg)

    pngFileName = re.sub('.bmp', '_locs.png', imPath)
    cv2.imwrite(pngFileName, colorImg)
    #cv2.waitKey(0)
    
files = os.listdir()
for file in files:
    if '.bmp' in file:
        findClusterCenters(file)
        
        
# %%

# m00 = [cv2.moments(c)["m00"] for c in contours]
# m10 = [cv2.moments(c)["m10"] for c in contours]
# m01 = [cv2.moments(c)["m01"] for c in contours]
# m00 = np.array(m00)
# m01 = np.array(m01)
# m10 = np.array(m10)
#
# m10=m10[~(m00==0)]
# m01=m01[~(m00==0)]
# m00=m00[~(m00==0)]
#
# cX = m10/m00
# cY = m01/m00