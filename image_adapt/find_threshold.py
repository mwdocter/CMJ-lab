# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:55:54 2019

@author: margreet

this code can calculate the threshold, or subtract the background
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt

def get_threshold(image_stack, show=0):
    ydata=(np.sort(image_stack.ravel()))
    xdata=np.array(range(0,len(ydata)))
    # scale the data to make x and y evenly important
    ymaxALL=float(max(ydata))
    xmaxALL=float(max(xdata))
    ydata=ydata*xmaxALL/ymaxALL #don't forget this is scaled

    if show:
        plt.figure(1)
        plt.plot(xdata,ydata)
        plt.show()

    # fit a line through the lowest half of x
    xd=xdata[:int(np.floor(len(xdata)/2))]
    yd=ydata[:int(np.floor(len(xdata)/2))]
    p_start=np.polyfit(xd,yd,1)

    #fit a line through the upper half of y
    ymax=max(ydata)
    yhalf=ymax/2
    x2=np.argwhere( abs(ydata-yhalf) ==min(abs(ydata-yhalf)) )
    x2=int(x2[0])
    xd=xdata[x2:]
    yd=ydata[x2:]
    p_end=np.polyfit(xd,yd,1)

    # find the crossing of these lines
    #a1*x+b1=a2*x+b2
    #(a1-a2)*x=b2-b1
    #x=(b2-b1)/(a1-a2)
    x_cross=int((p_end[1]-p_start[1])/(p_start[0]-p_end[0]))
    y_cross=int(np.polyval(p_start,x_cross))

    # add polyfits to the plot
    y_fit_start=np.polyval(p_start,xdata[:x_cross])
    x_fit_end=xdata[x_cross:] # start to draw from crossing y=0
    y_fit_end=np.polyval(p_end,x_fit_end)
    # now find the closest distance from cross to actual data. x and y should be simarly scaled
    xx=xdata-x_cross
    xx=[float(ii) for ii in xx]
    yy=ydata-y_cross
    yy=[float(ii) for ii in yy]
    rr=(np.array(xx)**2 + np.array(yy)**2)**0.5 #int32 is not large enough
    x_found=np.argwhere(min(rr)==rr)
    x_found=x_found[0,0]
    if show:
        fig2=plt.subplot(1,2,2)
        fig2.plot(xdata,rr*ymaxALL/xmaxALL)
        #fig2.title("{:s}".format(x_found))

        plt.figure(1)
        fig1=plt.subplot(1,2,1)
        fig1.plot(xdata,ydata*ymaxALL/xmaxALL,'b')
        fig1.plot(xdata[:x_cross],y_fit_start[:x_cross]*ymaxALL/xmaxALL,'g')
        fig1.plot(x_fit_end,y_fit_end*ymaxALL/xmaxALL,'r')
        fig1.plot(x_cross,y_cross*ymaxALL/xmaxALL,'kx')

        fig1.plot(x_found,ydata[x_found]*ymaxALL/xmaxALL,'mo')
        plt.show()
        
    thr = ydata[x_found]*ymaxALL/xmaxALL
    im_uit=image_stack-thr.astype(type(image_stack[0,0]))
    im_uit[im_uit<0]=0
    return thr


def remove_background(image_stack, thr, show=0):
    im_uit = image_stack - thr.astype(type(image_stack[0, 0]))
    im_uit[im_uit < 0] = 0
    return im_uit
