# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:54:18 2019

@author: mwdocter

used to check whether the traces and pks from Python have similar structure as generated from IDL
"""
import numpy as np
import matplotlib.pyplot as plt

root='H:\\projects\\research practicum\\single molecule fluorescence\\Matlab\\HJA-data from Ivo'
name='hel6.pma'

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

orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, Nframes), order = 'F') 
donor=orderedData[0,:,:]   
acceptor=orderedData[1,:,:]

pksData=np.zeros((Ntraces,5))
with open(root+'\\'+name[:-4]+'.pks', 'r') as infile:
    for jj in range(Ntraces):
        tmp= (infile.readline()).split()
        print(tmp)
        if tmp!=[]:
            pksData[jj,:]=tmp


for jj in range(0,2):#pts_number):
    plt.figure(jj)
    plt.clf() #clear plot
    plt.subplot(2,1,1)    
    plt.plot(donor[jj,:],'g')
    plt.plot(acceptor[jj,:],'r')
    plt.subplot(2,1,2)
    plt.plot(acceptor[jj,:]/ (donor[jj,:]+acceptor[jj,:]+0.001))
    plt.title(jj)
    plt.pause(0.1)#    plt.waitforbuttonpress()
    print(jj)

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
#orderedData = np.reshape(rawData.ravel(), (Ncolours, Ntraces//Ncolours, Nframes), order = 'F') 
orderedData = np.reshape(rawData, (Ncolours, Ntraces//Ncolours, Nframes ), order = 'F') 
donorREAD=orderedData[0,:,:]   
acceptorREAD=orderedData[1,:,:]

pksDataREAD=np.zeros((Ntraces,5))
with open(root+'\\'+name[:-4]+'a.pks', 'r') as infile:
    for jj in range(Ntraces):
        tmp= (infile.readline()).split()
        if tmp!=[]:
            pksDataREAD[jj,:]=tmp