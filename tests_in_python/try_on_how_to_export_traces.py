# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:53:51 2019

@author: mwdocter

find out best way how to save traces
"""
root='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies'
name='hel4.pma'
import numpy as np
import matplotlib.pyplot as plt



donor=np.array([jj+np.array(range(20)) for jj in np.array(range(50))])
acceptor=np.array([jj+np.array(range(20)) for jj in np.array(range(50,-1,-1))])

Nframes=np.shape(donor)[1]
Ntraces=np.shape(donor)[0]
Ncolours=2

## how to save traces file?
with open(root+'\\'+name[:-4]+'-P try1.traces', 'w') as outfile:
        off = np.array([Nframes], dtype=np.int32)
        off.tofile(outfile)
        off = np.array([2*Ntraces], dtype=np.int16)
        off.tofile(outfile)
        time_tr=np.zeros((Nframes,2*Ntraces))
        for jj in range(2*Ntraces//Ncolours):
            time_tr[:,jj*2] = donor[jj,:]
            time_tr[:,jj*2+1]=  acceptor[jj,:]
        off = np.array((time_tr), dtype=np.int16)
        off.tofile(outfile)
 
with open(root+'\\'+name[:-4]+'-P try2.traces', 'w') as outfile:
    
        off = np.array([Nframes], dtype=np.int32)
        off.tofile(outfile)
        off = np.array([2*Ntraces], dtype=np.int16)
        off.tofile(outfile)
        time_tr=np.zeros((2*Ntraces,Nframes))
        for jj in range(Ntraces//Ncolours):
            time_tr[jj*2,:] = donor[jj,:]
            time_tr[jj*2+1,:]=  acceptor[jj,:]
        off = np.array((time_tr), dtype=np.int16)
        off.tofile(outfile)
    

#open traces file
from traceAnalysisCode_MD import Experiment
mainPath=r'N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies'
exp1 = Experiment(mainPath)

for ii in range(len(exp1.files)):
    print([ii,exp1.files[ii].name])

plt.figure(1), plt.subplot(1,1,1)
plt.subplot(3,3,1), plt.plot(donor[jj,:]) # this is the input to the file
plt.subplot(3,3,2), plt.plot(acceptor[jj,:])

donor_out6=exp1.files[6].molecules[jj].intensity[0,:] 
acceptor_out6=exp1.files[6].molecules[jj].intensity[0,:]

plt.subplot(3,3,4), plt.plot(donor_out6) # this is the output to the file
plt.subplot(3,3,5), plt.plot(acceptor_out6)
#plt.subplot(2,3,6), plt.plot(acceptor_out/(acceptor_out+donor_out))
donor_out7=exp1.files[3].molecules[jj].intensity[0,:]
acceptor_out7=exp1.files[3].molecules[jj].intensity[0,:]

plt.subplot(3,3,7), plt.plot(donor_out7)
plt.subplot(3,3,8), plt.plot(acceptor_out7)
