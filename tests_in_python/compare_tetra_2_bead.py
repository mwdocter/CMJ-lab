# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:01:57 2019

@author: mwdocter

compare mapping file from IDL and Python
"""

from autopick.do_before import clear_all
clear_all()
import cv2 #computer vision?
from PIL import Image # Python Imaging Library (PIL)
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters #  Image processing in Python — scikit-image
import numpy as np
import bisect #This module provides support for maintaining a list in sorted order without having to sort the list after each insertion.


file_tetra1="E://CMJ trace analysis/autopick/tetraspeck.tif" # most likely not related to the data
file_tetra2='H://projects//research practicum//single molecule fluorescence//Matlab//HJA-data from Ivo//hel6_ave.tif'

image_tetra1 = tiff.imread(file_tetra1)

image_tetra2 = tiff.imread(file_tetra2)

transform_matrix=mapping(file_tetra1,show=1,bg=0)[0]
transform_matrix=mapping(file_tetra2,show=1)[0]