# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:08:00 2019

@author: mwdocter

click on 10 (!) points and then make a (manual) transformation matrix
"""
# # NOTES # #
# Install the packages opencv and opencv-contrib.
# Both versions of OpenCV have to be lower than 3.4.3.
# This is because SIFT and SURF algorithms are both patented and removed from newer versions
# Range of uint16: [0, 65535]
# code via Shirani Bisnajak (BEP 2018-2019)

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
from autopick.do_before import clear_all
clear_all()
import cv2 #computer vision?
from PIL import Image # Python Imaging Library (PIL)
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters #  Image processing in Python — scikit-image
import numpy as np
import bisect #This module provides support for maintaining a list in sorted order without having to sort the list after each insertion.
from image_adapt.find_threshold import remove_background, get_threshold

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
    
    
def imadjust(src, tol=1, vout=(0,255)): #maybe try tol=0
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    vin = [np.min(src), np.max(src)]
    vout = [0, 65535] # 65535=16 bits
    print(vin,vout)
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
    print(vin,vout)
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


def enhance_blobies(image, f, tol):
    l, r = image[:, :image.shape[1]//2], image[:, image.shape[1]//2:]
    l_adj, r_adj = imadjust(l.copy(), tol), imadjust(r.copy(), tol)
    l_bin, r_bin = im_binarize(l_adj, f).astype(np.uint8), im_binarize(r_adj,f).astype(np.uint8)
#    plt.figure(10)
#    plt.subplot(1,4,1)
#    plt.imshow(l_adj)
#    plt.subplot(1,4,2)
#    plt.imshow(l_bin)
#    plt.subplot(1,4,3)
#    plt.imshow(r_adj)
#    plt.subplot(1,4,4)
#    plt.imshow(r_bin)
    
    return l, r, l_bin, r_bin

#def mapping(file_tetra='tetraspeck.tif', show=0, f=None,bg=None): #'E:\CMJ trace analysis\\autopick\\tetraspeck.tif'
def mapping(file_tetra, show=0, f=None,bg=None, tol=1): #'E:\CMJ trace analysis\\autopick\\tetraspeck.tif'
    global points_right # globals in nested function, need to be defined here as well
    global ii
    # Open image
    if type(file_tetra)==str:
        image_tetra = tiff.imread(file_tetra)
    else: #assume you are passing an image
        image_tetra=file_tetra
    
    # default for 16 bits 50000, for 8 bits 200 (=256*50000/64000)
    if f==None:
        f=50000
    print(f)
    
    if type(bg)==type(None):
        # take two different backgrounds, one for donor, one for acceptor channel
        sh=np.shape(image_tetra)
        thr_donor=get_threshold(image_tetra[:,1:sh[0]//2])
        thr_acceptor=get_threshold(image_tetra[:,sh[0]//2:])
        bg=np.zeros(sh)
        bg[:,1:sh[0]//2]=thr_donor
        bg[:,sh[0]//2:]=thr_acceptor
    image_tetra=remove_background(image_tetra.astype(float),bg)
#    image_tetra=image_tetra.astype(float)-bg
#    image_tetra[image_tetra<0]=0
    image_tetra=image_tetra.astype(np.uint16)    
    position1=[]
    position2=[]
    # left, right, enhanced left and enhanced right image for keypoint detection, adapt f
    while np.shape(position1)[0]<50 or np.shape(position2)[0]<50 : 
        # while loop to lower f and increase the number of spots found
        l, r, l_enh, r_enh = enhance_blobies(image_tetra,f, tol)
        
        gray1 = l_enh
        gray2 = r_enh 
        
        # initialize the AKAZE descriptor, then detect keypoints and extract
        # local invariant descriptors from the image
        detector = cv2.AKAZE_create()
        (kps1, descs1) = detector.detectAndCompute(gray1, None)
        (kps2, descs2) = detector.detectAndCompute(gray2, None)
        position1=cv2.KeyPoint_convert(kps1);
        position2=cv2.KeyPoint_convert(kps2);
        f=f*0.9
    
   # now we start to move the markers in the right part, first an overall move
    from pynput.keyboard import Key, Listener
    import matplotlib.pyplot as plt
    import time
    
#    points_left=[[ii, ii+10] for ii in range(11)]
#    points_right=points_left.copy()

    fig = plt.figure(10, figsize=(18,9))
    ax1 = fig.add_subplot(1,2,1)  
    ax1.imshow(gray1)
    ax1.set_title('click on bright spots in the left image')
    #ax1=plt. subplot(1,2,1)

        

    #ax2=plt. subplot(1,2,2)
    ax2 = fig.add_subplot(1,2,2,sharex=ax1,sharey=ax1) 
    ax2.imshow(gray2)
    
    points_left=plt.ginput(4) # adapted it to be 4, I expected 3 to work, to match with IDL code
    points_right=points_left.copy()
    [ax1.plot(xx,yy,markersize=10, c='w', marker='o', fillstyle='none') for xx,yy in points_left]
        
    xp=[xx for xx,yy in points_right]
    yp=[yy for xx,yy in points_right]
    line2,=ax2.plot(xp,yp,markersize=10, c='w', marker='o', fillstyle='none', linestyle='none') 
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1)
    
    ax1.set_title('')
    ax2.set_title('move point with arrows, press esc when it matches the location in other channel')

    def on_release(key):
        global points_right
        global ii
        #print(points_right)
        if key == Key.up:
            points_right[ii]=[points_right[ii][0], points_right[ii][1]+1]
           # points_right= [ [xx,yy+1] for xx,yy in points_right]
        elif key == Key.down:
            points_right[ii]=[points_right[ii][0], points_right[ii][1]-1]
           # points_right= [ [xx,yy-1] for xx,yy in points_right]
        elif key == Key.right:
            points_right[ii]=[points_right[ii][0]+1, points_right[ii][1]]
           # points_right= [ [xx+1,yy] for xx,yy in points_right]
        elif key == Key.left:
            points_right[ii]=[points_right[ii][0]-1, points_right[ii][1]]
           # points_right= [ [xx-1,yy] for xx,yy in points_right]
               
        
        if key == Key.esc or (key==Key.up) or (key == Key.down) or (key == Key.right) or (key == Key.left):
            # Stop listener
            return False

    matches=len(points_left)   
    for ii in range(matches):
        xp_new=0*xp
        yp_new=0*yp
        ax1.set_xlim(points_left[ii][0]-50, points_left[ii][0]+50)
        ax1.set_ylim(points_left[ii][1]-50, points_left[ii][1]+50)
        line1=ax1.plot(points_left[ii][0],points_left[ii][1],markersize=10, c='y', marker='+', fillstyle='none')      
        line2b=ax2.plot(points_right[ii][0],points_right[ii][1],markersize=10, c='y', marker='+', fillstyle='none')      
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)  
        while xp!=xp_new or yp!=yp_new:
            xp=[xx for xx,yy in points_right]
            yp=[yy for xx,yy in points_right]
            line2.set_xdata(xp)
            line2.set_ydata(yp)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.1)  
            # Collect events until released
            with Listener(on_release=on_release) as listener:
                listener.join()
        
            xp_new=[xx for xx,yy in points_right]
            yp_new=[yy for xx,yy in points_right]
#        print(np.int(xp_new[ii]),np.int(yp_new[ii]))
        
           
        line1[-1].remove()
        line2b[-1].remove()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)  
    # now loop over each spot to update its position in a zoomed in window
    
    # get the points_right to match syntax pts1, pts2
    points_right=np.array(points_right)
    points_left=np.array(points_left)
    transformation_matrixC, mask = cv2.findHomography(points_right, points_left, cv2.RANSAC,20)
              
##    A=pts1[0:len(matches) : int(len(matches)/15)]
#    if len(matches)>20:
#        im3 = cv2.drawMatchesKnn(gray1, kps1, gray2, kps2,matches[1:20],None )
#    else:
#        im3 = cv2.drawMatchesKnn(gray1, kps1, gray2, kps2,matches,None )
# #    im3 = cv2.drawMatchesKnn(gray1, kps1, gray2, kps2,matches[1:20] , None, flags=2)
#    if show:
#        plt.figure(1)
#        plt.imshow(im3)
#        plt.title('mapping matching keypoints')
#        plt.show
        
    # produce an image in which the overlay between two channels is shown
    array_size=np.shape(gray2)
    imC=cv2.warpPerspective(gray2, transformation_matrixC, array_size[::-1] )
    
    #cv2.imshow("transformed ", im4)
    if show:
        plt.figure(11, figsize=(18,9))
        plt.subplot(1,1,1)
        plt.subplot(1,6,1),
        plt.imshow(gray1, extent=[0,array_size[1],0,array_size[0]], aspect=1)
        plt.title('green channel')
            
        plt.subplot(1,6,2),
        plt.imshow(gray2, extent=[0,array_size[1],0,array_size[0]], aspect=1)
        plt.title('red channel')    
        
        plt.subplot(1,6,3),
        plt.imshow(imC, extent=[0,array_size[1],0,array_size[0]], aspect=1)
        plt.title('red transformed')
        plt.show()
    
        plt.subplot(1,6,4),
        A=(gray1>0)+2*(gray2>0)
        plt.imshow(A, extent=[0,array_size[1],0,array_size[0]], aspect=1)
       # plt.colorbar()
        plt.title( 'unaligned #(yellow) \nspots overlap {:d}'.format(np.sum(A==3))   ) 
            
        plt.subplot(1,6,5),
        AA=(gray1>0)+2*(imC>0)
        plt.imshow((gray1>0)+2*(imC>0), extent=[0,array_size[1],0,array_size[0]], aspect=1)
        #plt.colorbar()
        plt.title(  'manual align \n#spots overlap y{:d}'.format(np.sum(AA==3)) )    
        plt.show()
        plt.pause(0.05)

# ask whether the user agree, if not: rerun with clicking corresponding points
# cv2.getAffineTransform(src, dst) → retval
    plt.figure(10), plt.close()            
    return  transformation_matrixC, points_left, points_right