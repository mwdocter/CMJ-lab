# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:51:28 2019

@author: mwdocter

trial to make the traces written by Python similar structure as written in IDL
"""
import numpy as np
import matplotlib.pyplot as plt

root='H:\\projects\\research practicum\\single molecule fluorescence\\Matlab\\HJA-data from Ivo'
name='hel6.pma'

## version a4 is working!

Ncolours=2      
with open(root+'\\'+name[:-4]+'.traces', 'r') as infile:
#    nImages=infile.readline(1)
#    pts_number=infile.readline(1)
#    for jj in range(pts_number):
#        donor[jj,:]=infile.read(nImages)
#        acceptor[jj,:]=infile.read(nImages)
     Nframes = np.fromfile(infile, dtype = np.int32, count = 1).item()
     Ntraces = np.fromfile(infile, dtype = np.int16, count = 1).item()
     rawData = np.fromfile(infile, dtype = np.int16, count = Ncolours*Nframes * Ntraces)

#orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, Nframes), order = 'F') 
orderedData = np.reshape(rawData, (Ncolours, Ntraces//Ncolours, Nframes ), order = 'F') 
donor=orderedData[0,:,:]   
acceptor=orderedData[1,:,:]
plt.subplot(3,1,1)
plt.imshow(donor)
#-------------------------------------------------------------------

with open(root+'\\'+name[:-4]+'a1.traces', 'w') as outfile:
    off = np.array([Nframes], dtype=np.int32)
    off.tofile(outfile)
    off = np.array([Ntraces], dtype=np.int16)
    off.tofile(outfile)
    for jj in range(Ntraces//Ncolours):
        off = np.array(0*donor[jj,:]+jj, dtype=np.int16)
        off.tofile(outfile)
        off = np.array(0*acceptor[jj,:]+jj+1000, dtype=np.int16)
        off.tofile(outfile)

Ncolours=2     
with open(root+'\\'+name[:-4]+'a1.traces', 'r') as infile:

     Nframes = np.fromfile(infile, dtype = np.int32, count = 1).item()
     Ntraces = np.fromfile(infile, dtype = np.int16, count = 1).item()
     rawData = np.fromfile(infile, dtype = np.int16, count = Ncolours*Nframes * Ntraces)
orderedData = np.reshape(rawData, (Ncolours, Ntraces//Ncolours, Nframes ), order = 'F') 
donorREAD=orderedData[0,:,:]   
acceptorREAD=orderedData[1,:,:]
plt.subplot(3,1,2)
plt.imshow(donorREAD)
plt.subplot(3,1,3)
plt.imshow(acceptorREAD)
#-------------------------------------------------------------------
with open(root+'\\'+name[:-4]+'a2.traces', 'w') as outfile:
    off = np.array([Nframes], dtype=np.int32)
    off.tofile(outfile)
    off = np.array([Ntraces], dtype=np.int16)
    off.tofile(outfile)
    for ii in range(Nframes):
        off = np.array(0*donor[:,ii]+ii, dtype=np.int16)
        off.tofile(outfile)
        off = np.array(0*acceptor[:,ii]+1000+ii, dtype=np.int16)
        off.tofile(outfile)
    
Ncolours=2     
with open(root+'\\'+name[:-4]+'a2.traces', 'r') as infile:
     Nframes = np.fromfile(infile, dtype = np.int32, count = 1).item()
     Ntraces = np.fromfile(infile, dtype = np.int16, count = 1).item()
     rawData = np.fromfile(infile, dtype = np.int16, count = Ncolours*Nframes * Ntraces)

orderedData = np.reshape(rawData, (Ncolours, Ntraces//Ncolours, Nframes ), order = 'F') 
donorREAD=orderedData[0,:,:]   
acceptorREAD=orderedData[1,:,:]
plt.subplot(3,1,2)
plt.imshow(donorREAD)
plt.subplot(3,1,3)
plt.imshow(acceptorREAD)
 #-------------------------------------------------------------------
with open(root+'\\'+name[:-4]+'a3.traces', 'w') as outfile:
    off = np.array([Nframes], dtype=np.int32)
    off.tofile(outfile)
    off = np.array([Ntraces], dtype=np.int16)
    off.tofile(outfile)
    donor_rot90=np.rot90(donor)
    acceptor_rot90=np.rot90(acceptor)
    for jj in range(Ntraces//Ncolours):
        off = np.array(0*donor_rot90[:,jj]+jj, dtype=np.int16)
        off.tofile(outfile)
        off = np.array(0*acceptor_rot90[:,jj]+jj+1000, dtype=np.int16)
        off.tofile(outfile)
    
Ncolours=2     
with open(root+'\\'+name[:-4]+'a3.traces', 'r') as infile:
     Nframes = np.fromfile(infile, dtype = np.int32, count = 1).item()
     Ntraces = np.fromfile(infile, dtype = np.int16, count = 1).item()
     rawData = np.fromfile(infile, dtype = np.int16, count = Ncolours*Nframes * Ntraces)
orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, Nframes), order = 'F') 
donorREAD=orderedData[0,:,:]   
acceptorREAD=orderedData[1,:,:]
plt.subplot(3,1,2)
plt.imshow(donorREAD)
plt.subplot(3,1,3)
plt.imshow(acceptorREAD)
#-------------------------------------------------------------------
with open(root+'\\'+name[:-4]+'a4.traces', 'w') as outfile:
    off = np.array([Nframes], dtype=np.int32)
    off.tofile(outfile)
    off = np.array([Ntraces], dtype=np.int16)
    off.tofile(outfile)
    # for testing use ramp values, constant per xy position, varying between positions
#    time_tr=np.zeros((Nframes,Ntraces))
#    for jj in range(Ntraces//Ncolours):
#        time_tr[:,jj*2] = donor[jj,:]+jj
#        time_tr[:,jj*2+1]=  acceptor[jj,:] + jj + 1000
#    off = np.array((time_tr), dtype=np.int16)
    time_tr=np.zeros((Nframes,Ntraces))
    for jj in range(Ntraces//Ncolours):
        time_tr[:,jj*2] = donor[jj,:]
        time_tr[:,jj*2+1]=  acceptor[jj,:]
    off = np.array((time_tr), dtype=np.int16)
    off.tofile(outfile)
    
Ncolours=2     
with open(root+'\\'+name[:-4]+'a4.traces', 'r') as infile:
#    nImages=infile.readline(1)
#    pts_number=infile.readline(1)
#    for jj in range(pts_number):
#        donor[jj,:]=infile.read(nImages)
#        acceptor[jj,:]=infile.read(nImages)
     Nframes = np.fromfile(infile, dtype = np.int32, count = 1).item()
     Ntraces = np.fromfile(infile, dtype = np.int16, count = 1).item()
     rawData = np.fromfile(infile, dtype = np.int16, count = Ncolours*Nframes * Ntraces)
orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, Nframes), order = 'F') 
#orderedData = np.reshape(rawData, (Ncolours, Ntraces//Ncolours, Nframes ), order = 'F') 
donorREAD=orderedData[0,:,:]   
acceptorREAD=orderedData[1,:,:]
plt.subplot(3,1,1)
plt.imshow(donor)
plt.subplot(3,1,2)
plt.imshow(donorREAD)
plt.subplot(3,1,3)
plt.imshow(acceptorREAD)