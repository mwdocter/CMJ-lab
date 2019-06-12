# -*- coding: utf-8 -*-
"""
Created on Jun 2019
@author: carlos
meant for mapping, but now with class implementation
"""
import os
from ImageCollection import ImageCollection
from Mapping import Mapping
import numpy as np
import matplotlib.pyplot as plt

tetra_fn=os.path.normpath('N:/tnw/BN/CMJ/Shared/Margreet/20181203_HJ_training/mapping/rough_ave.tif')

#tetra_fn='/home/carlos/PycharmProjects/margreet_code_review/rough_ave.tif'
#image_fn ='/home/carlos/PycharmProjects/margreet_code_review/hel4.pma'

tmp=Mapping(tetra_fn)

if 1:
    root=os.path.normpath("N:/tnw/BN/CMJ/Shared/Margreet/20181203_HJ_training/#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies")
    name='hel4.pma'

else:
    root=os.path.normpath("N:/tnw/BN/CMJ/Shared/Margreet/181218 - First single-molecule sample (GattaQuant)/RawData") 
    name='Spooled files.sifx'
 
image_fn=os.path.join(root,name)
imc = ImageCollection(tetra_fn, image_fn) 
# better: give image directory, and search for specific name or extension
img = imc.get_image(1)

#tmp._tf2_matrix
#imc.mapping._tf2_matrix
#imc.pts_number # is too 19 for the pma
#import matplotlib.pyplot as plt; plt.imshow(imc.im_mean20_correct)

traces_fn=os.path.join(root,name[:-4]+'-P.traces') 
Ncolours=2
if os.path.isfile(traces_fn):
     with open(traces_fn, 'r') as infile:
         Nframes = np.fromfile(infile, dtype = np.int32, count = 1).item()
         Ntraces = np.fromfile(infile, dtype = np.int16, count = 1).item()
         rawData = np.fromfile(infile, dtype = np.int16, count = Ncolours*Nframes * Ntraces)
     orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, Nframes), order = 'F') 
     donor=orderedData[0,:,:]   
     acceptor=orderedData[1,:,:]
     donor=np.transpose(donor)
     acceptor=np.transpose(acceptor)
else:
    donor=np.zeros(( imc.n_images,imc.pts_number))
    acceptor=np.zeros((imc.n_images,imc.pts_number))
    import time
    t0 = time.time()  
    for ii in range(0,imc.n_images):
        img=imc.get_image(ii)
        donor[ii,:]=img.donor
        acceptor[ii,:]=img.acceptor
    t1=time.time()
    elapsed_time=t1-t0; print(elapsed_time)    
    
    #root, name = os.path.split(self.image_fn)
    
    #if os.path.isfile(trace_fn):
       
    with open(traces_fn, 'w') as outfile:
         off = np.array([imc.n_images], dtype=np.int32)
         off.tofile(outfile)
         off = np.array([2*imc.pts_number], dtype=np.int16)
         off.tofile(outfile)
         time_tr=np.zeros((imc.n_images,2*imc.pts_number))
         Ncolours=2
         for jj in range(2*imc.pts_number//Ncolours):
             time_tr[:,jj*2] = donor[:,jj]
             time_tr[:,jj*2+1]=  acceptor[:,jj]
         off = np.array((time_tr), dtype=np.int16)
         off.tofile(outfile)
         