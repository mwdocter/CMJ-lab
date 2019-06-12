# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:16:48 2019

@author: mwdocter
"""

from find_xy_position.Gaussian import makeGaussian
import matplotlib.pyplot as plt
GG=makeGaussian(11, fwhm=3, center=(5,5))  # approach3

SS=np.zeros((6,6))
for ii in range(0,6)  :
    for jj in range(0,6):
        spot=makeGaussian(11, fwhm=3, center=(5+ii*.1,5+jj*.1)) 
        SS[ii,jj]=np.sum(spot*GG)
        
#
#SS=
#
#array([[5.09890741, 5.09105942, 5.06758784, 5.02870879, 4.97477875,
#        4.90628917],
#       [5.09105942, 5.0832235 , 5.05978805, 5.02096884, 4.96712181,
#        4.89873764],
#       [5.06758784, 5.05978805, 5.03646065, 4.9978204 , 4.94422163,
#        4.87615273],
#       [5.02870879, 5.02096884, 4.9978204 , 4.95947661, 4.90628905,
#        4.83874239],
#       [4.97477875, 4.96712181, 4.94422163, 4.90628905, 4.8536719 ,
#        4.78684964],
#       [4.90628917, 4.89873764, 4.87615273, 4.83874239, 4.78684964,
#        4.72094734]])        