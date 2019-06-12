# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:12:14 2019

@author: mwdocter

three parts: first part checks the difference between different extraction methods (point from image)
second part: check how many traces do have acceptor or donor signal
third part: tries to find best match of traces between IDL and Python (usefull if pks file is missing)
"""
import sys
import time
import analyze_label
import cv2

from autopick.do_before import clear_all
import autopick.pick_spots_akaze_manual 
import autopick.pick_spots_akaze_final 
from load_file import read_one_page_pma
from image_adapt.rolling_ball import rollingball
from image_adapt.find_threshold import remove_background
from find_xy_position.Gaussian import makeGaussian

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
root='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies'
name='hel4.pma'

fn_IDL='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\mapping\\rough.map';
fn_Python='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies\\hel4-P.map'
fn_Python_COEFF='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies\\hel4-P.coeff'

## copied from applyMapping_v3
_,hdim,vdim,nImages=(read_one_page_pma(root,name, pageNb=0))
im_sum=(read_one_page_pma(root,name, pageNb=0 )[0]).astype(float)
for ii in range (1,20):
    im=(read_one_page_pma(root,name, pageNb=ii)[0]).astype(float)
    im[im<0]=0
    im_sum=im_sum+im
im_mean20=(im_sum/20).astype(int)
im_bg=rollingball(im_mean20)[1]
##

transform_matrix=np.zeros((3,3))
transform_matrix[2][2]=1
## read in coeff into transform_matrix
with open(root+'\\'+name[:-4]+'-P.coeff', 'r') as infile:
    transform_matrix[0,2]=float(infile.readline())-256
    transform_matrix[0,0]=float(infile.readline())   
    transform_matrix[0,1]=float(infile.readline())
    transform_matrix[1,2]=float(infile.readline())  
    transform_matrix[1,0]=float(infile.readline())
    transform_matrix[1,1]=float(infile.readline())  
    
ii=1
jj=82


im=(read_one_page_pma(root,name, pageNb=ii))[0]
plt.close('all')
plt.figure(1), plt.imshow(im),plt.colorbar()

im_correct=im-im_bg
im_correct[im_correct<0]=0
im_correct2,threshold=remove_background(im_correct, show=1)
plt.figure(1), plt.subplot(1,1,1), plt.imshow(im_correct2),plt.colorbar()

IM_donor=im_correct2[:,0:int(vdim/2)]
IM_acceptor=im_correct2[:,int(vdim/2):]

array_size=np.shape(IM_acceptor)
imA=cv2.warpPerspective(IM_acceptor.astype(float), transform_matrix,array_size[::-1])

plt.figure(10), 
ax=plt.subplot(1,3,1); ax.imshow(IM_donor), ax.set_title('donor')
ax=plt.subplot(1,3,2); ax.imshow(IM_acceptor),ax.set_title('acceptor, not shifted')
ax=plt.subplot(1,3,3); ax.imshow(imA),ax.set_title('acceptor, shifted')

A=(IM_donor>0)+2*(IM_acceptor>0)
B=(IM_donor>0)+2*(imA>0)
plt.figure(11), plt.subplot(1,2,1), plt.imshow(A), plt.title([(A==1).sum(),(A==2).sum(),(A==3).sum()])
plt.figure(11), plt.subplot(1,2,2), plt.imshow(B), plt.title([(B==1).sum(),(B==2).sum(),(B==3).sum()])
# display the size of non overlapping channels, and overlap

#initiating plots, plus put on right position in screen
fig=plt.figure(2, figsize=(6.5,2.5)) #figsize in inches
fig.canvas.manager.window.move(0,0)

fig=plt.figure(3, figsize=(6.5,2.5)) #figsize in inches
fig.canvas.manager.window.move(0,300)

fig=plt.figure(4, figsize=(6.5,2.5)) #figsize in inches
fig.canvas.manager.window.move(0,600)

fig=plt.figure(5, figsize=(6.5,2.5)) #figsize in inches
fig.canvas.manager.window.move(1000,0)

fig=plt.figure(6, figsize=(6.5,2.5)) #figsize in inches
fig.canvas.manager.window.move(1000,300)

fig=plt.figure(7, figsize=(6.5,2.5)) #figsize in inches
fig.canvas.manager.window.move(1000,600)

## donor intensity ##

##LOAD VALUES FROM PKS pts_number,labels,ptsG=analyze_label.analyze(im_correct2[:,0:int(vdim/2)])
##also dstG
ptsG=[]
dstG2=[]
with open(root+'\\'+name[:-4]+'-PC.pks', 'r') as infile:
    cc=0
    for line in infile:
        cc=cc+1
        if cc%2 != 0: # cc%2!= 0 means odd, so ptsG, cc%2!= 1 means even, so dstG
            A=[float(ii) for ii in line.split()]
            ptsG.append(A[1:3])
        else:
            B=[float(ii) for ii in line.split()]
            dstG2.append(B[1:3])
pts_number=len(ptsG)
ptsG=np.array(ptsG)
dstG2=np.array(dstG2)

tmp = np.genfromtxt(fn_IDL) 
Pidl=tmp.reshape(8,4)[0:4,:]
Qidl=tmp.reshape(8,4)[4:,:]
dstGidl=np.zeros(np.shape(ptsG))
    
tmp = np.genfromtxt(fn_Python) 
Ppyt=tmp.reshape(8,4)[0:4,:]
Qpyt=tmp.reshape(8,4)[4:,:]
dstGpyt=np.zeros(np.shape(ptsG))

## make a loop over jj to watch each selected spot individually             
for jj in range(pts_number):
    print(jj)
    xpix=ptsG[jj][1] 
    ypix=ptsG[jj][0]
    
    xpix_int=int(xpix)
    ypix_int=int(ypix)
    DD=10
    impixD=IM_donor[ (xpix_int-DD) : (xpix_int+DD+1) , (ypix_int-DD) : (ypix_int+DD+1)]
    GG=makeGaussian(2*DD+1, fwhm=3, center=(ypix-ypix_int+DD,xpix-xpix_int+DD))
    multipD=impixD*GG
    donor=np.sum(multipD)
    
    plt.figure(2), 
    plt.subplot(1,3,1)
    plt.imshow(impixD), plt.title('donor')
    plt.subplot(1,3,2)
    plt.imshow(GG), plt.title([jj,xpix_int,ypix_int] )
    plt.subplot(1,3,3)
    plt.imshow(multipD)
    plt.title(sum(sum(multipD)))
    #plt.pause(0.05)
    
    
    ## acceptor intensity, without warping ##
    
    impixE=IM_acceptor[ (xpix_int-DD) : (xpix_int+DD+1) , (ypix_int-DD) : (ypix_int+DD+1)]
    GGE=makeGaussian(2*DD+1, fwhm=3, center=(ypix-ypix_int+DD,xpix-xpix_int+DD))
    multipE=impixE*GGE
    acceptor0=np.sum(multipE)
    
    plt.figure(3)
    plt.subplot(1,3,1)
    plt.imshow(impixE), plt.title('acceptor, original')
    plt.subplot(1,3,2)
    plt.imshow(GG), plt.title([jj,xpix_int,ypix_int] )
    plt.subplot(1,3,3)
    plt.imshow(multipE)
    plt.title(sum(sum(multipE)))
    #plt.pause(0.05)
    
    ## acceptor intensity, warping the image ##
    
    impixA=imA[ (xpix_int-DD) : (xpix_int+DD+1) , (ypix_int-DD) : (ypix_int+DD+1)]
    
    multip=impixA*GG
    acceptor=np.sum(multip)
    
    plt.figure(4)
    plt.subplot(1,3,1)
    plt.imshow(impixA), plt.title('acceptor, image warped')
    plt.subplot(1,3,2)
    plt.imshow(GG), plt.title([jj,xpix_int,ypix_int] )
    plt.subplot(1,3,3)
    plt.imshow(multip)
    plt.title(sum(sum(multip)))
    #plt.pause(0.05)
   
    # approach 3, find transformed coordinates in im_correct (full image)
    xf2=dstG2[jj][1]#approach3,  NOTE: xy are swapped here
    yf2=dstG2[jj][0]#approach3
    xf2_int=int(xf2)#approach3
    yf2_int=int(yf2)#approach3
                    
    impixC=im_correct2[ (xf2_int-DD) : (xf2_int+DD+1) , (yf2_int-DD) : (yf2_int+DD+1)]
    GGC=makeGaussian(2*DD+1, fwhm=3, center=(yf2-yf2_int+DD,xf2-xf2_int+DD))#approach3
    multipC=impixC*GGC#approach3
    acceptorC=np.sum(multipC)
    
    plt.figure(5)
    plt.subplot(1,3,1)
    plt.imshow(impixC), plt.title('warped positions')
    plt.subplot(1,3,2)
    plt.imshow(GGC), plt.title([jj,xf2_int,yf2_int] )
    plt.subplot(1,3,3)
    plt.imshow(multipC)
    plt.title(sum(sum(multipC)))
    #plt.pause(0.05)
                
    ## acceptor intensity, warping the position, with the IDL transform matrix  ##
    
    ### compare to extraction IDL

    #for LL in range( len(ptsG)):
    new0=0
    new1=0
    old0=ptsG[jj][0]
    old1=ptsG[jj][1]
    for iii in range(4):
            for jjj in range(4):
                new0=new0+Pidl[iii,jjj]* (old0**iii)*(old1**jjj)
                new1=new1+Qidl[iii,jjj]* (old0**iii)*(old1**jjj)
    dstGidl[jj][0]=new0
    dstGidl[jj][1]=new1
    
    xnew=dstGidl[jj][1]
    ynew=dstGidl[jj][0]+256
    xnew_int=int(xnew)
    ynew_int=int(ynew)
    
    fig=plt.figure(6,figsize=(6.5,2.5))
    fig.canvas.manager.window.move(1000,300)
    if (xnew_int>10 and ynew>10 and xnew<hdim/2-10 and ynew<vdim-10):
        impixG=im_correct2[ (xnew_int-DD) : (xnew_int+DD+1) , (ynew_int-DD) : (ynew_int+DD+1)]
        GGC=makeGaussian(2*DD+1, fwhm=3, center=(ynew-ynew_int+DD,xnew-xnew_int+DD))#approach3
        multipG=impixG*GGC#approach3
        acceptorG=np.sum(multipG)
        
        plt.figure(6)
        plt.subplot(1,3,1)
        plt.imshow(impixG), plt.title('warped positions with IDL P&Q')
        plt.subplot(1,3,2)
        plt.imshow(GG),plt.title([jj,xnew_int,ynew_int] )
        plt.subplot(1,3,3)
        plt.imshow(multipG)
        plt.title(sum(sum(multipG)))
    #    plt.pause(0.05)
    
    ## acceptor intensity, warping the position, with the Python transform matrix  ##
    ### compare to extraction IDL
    

    #for LL in range( len(ptsG)):
    new0=0
    new1=0
    old0=ptsG[jj][0]
    old1=ptsG[jj][1]
    for iii in range(4):
            for jjj in range(4):
                new0=new0+Ppyt[iii,jjj]* (old0**iii)*(old1**jjj)
                new1=new1+Qpyt[iii,jjj]* (old0**iii)*(old1**jjj)
    dstGpyt[jj][0]=new0
    dstGpyt[jj][1]=new1
    
    xnew=dstGpyt[jj][1]
    ynew=dstGpyt[jj][0]+256
    xnew_int=int(xnew)
    ynew_int=int(ynew)
    
    plt.figure(7)
    plt.cla()
    if (xnew_int>10 and ynew>10 and xnew<hdim/2-10 and ynew<vdim-10):
        impixF=im_correct2[ (xnew_int-DD) : (xnew_int+DD+1) , (ynew_int-DD) : (ynew_int+DD+1)]
        GGF=makeGaussian(2*DD+1, fwhm=3, center=(ynew-ynew_int+DD,xnew-xnew_int+DD))#approach3
        multipF=impixF*GGF#approach3
        acceptorF=np.sum(multipD)
        
        plt.figure(7)
        plt.subplot(1,3,1)
        plt.imshow(impixF), plt.title('warped positions with Python P&Q')
        plt.subplot(1,3,2)
        plt.imshow(GG),plt.title([jj,xnew_int,ynew_int] )
        plt.subplot(1,3,3)
        plt.imshow(multipF)
        plt.title(sum(sum(multipF)))
     #   plt.pause(0.05)
    
    plt.pause(0.1)
    print(jj)

### from the above conclude that the mapping + alignment is working pretty well. 
### from below: 
########################################################################
#
from traceAnalysisCode_MD import Experiment

mainPath=r'N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies'

exp1 = Experiment(mainPath)
#

for ii in range(len(exp1.files)):
    print([ii,exp1.files[ii].name])
    
jjM=86
jj=86

donor=exp1.files[0].molecules[jjM].intensity[0,:]
acceptor=exp1.files[0].molecules[jjM].intensity[1,:]
plt.figure(1), plt.subplot(1,1,1)
plt.subplot(4,3,1), plt.plot(donor)
plt.subplot(4,3,2), plt.plot(acceptor)
#plt.subplot(2,3,3), plt.plot(acceptor[jj,:]/(acceptor[jj,:]+donor[jj,:]))

donor_out3=exp1.files[3].molecules[jj].intensity[0,:]
acceptor_out3=exp1.files[3].molecules[jj].intensity[1,:]
plt.subplot(4,3,4), plt.plot(donor_out3)
plt.subplot(4,3,5), plt.plot(acceptor_out3)   
#plt.subplot(2,3,6), plt.plot(acceptor_out/(acceptor_out+donor_out))

donor_out4=exp1.files[4].molecules[jj].intensity[0,:]
acceptor_out4=exp1.files[4].molecules[jj].intensity[1,:]
plt.subplot(4,3,7), plt.plot(donor_out4)
plt.subplot(4,3,8), plt.plot(acceptor_out4)   

donor_out5=exp1.files[5].molecules[jj].intensity[0,:]
acceptor_out5=exp1.files[5].molecules[jj].intensity[1,:]
plt.subplot(4,3,10), plt.plot(donor_out5)
plt.subplot(4,3,11), plt.plot(acceptor_out5)
   
if 0:
        
    plt.figure(15) # from this I can tell acceptor 
    cc1=0
    FF=0
    print(exp1.files[FF].name)
    for jj in range(len(exp1.files[FF].molecules)):
            plt.cla()
            exp1.files[FF].molecules[jj].plot()
            plt.title(jj)
            plt.pause(.7)
            if np.sum(exp1.files[FF].molecules[jj].intensity)>0:
                cc1=cc1+1
                
                
      
            
    plt.figure(16)
    FF=5
    cc2=0
    for jj in range(len(exp1.files[FF].molecules)):
            plt.cla()
            exp1.files[FF].molecules[jj].plot()
            plt.title(jj)
            plt.pause(0.2)
            if np.sum(exp1.files[FF].molecules[jj].intensity)>0:
             cc2=cc2+1  
        
###############################################################
             
from traceAnalysisCode_MD import Experiment
import os
import matplotlib.pyplot as plt #Provides a MATLAB-like plotting framework
import numpy as np

mainPath=r'N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies'
os.chdir(os.path.dirname(os.path.abspath('__file__')))

plt.close("all")

exp1 = Experiment(mainPath)


for ii in range (len(exp1.files)):
    print(exp1.files[ii].name)
    print(len(exp1.files[ii].molecules))
    # manually matched points
    
nIDL=len(exp1.files[0].molecules)
nPYT=len(exp1.files[3].molecules)

## in ss (and variants) is stored how well IDL and Python traces match
ss=np.zeros((nIDL,nPYT))
ss0=np.zeros((nIDL,nPYT))
ss1=np.zeros((nIDL,nPYT))
ss2=np.zeros((nIDL,nPYT))
for ii in range (nIDL):
    s0=np.zeros(nPYT)
    s1=np.zeros(nPYT)
    s2=np.zeros(nPYT)
    for jj in range (nPYT):
       
        tIDL_0= exp1.files[0].molecules[ii].intensity[0] #  to prevent them being zero
        tPYT_0= exp1.files[3].molecules[jj].intensity[0]
        tIDL_1= exp1.files[0].molecules[ii].intensity[1]
        tPYT_1= exp1.files[3].molecules[jj].intensity[1]
        #make into two state
#        if np.min(tIDL_0)<1:             tIDL_0=tIDL_0-np.min(tIDL_0)+1
#        if np.min(tIDL_1)<1:             tIDL_1=tIDL_1-np.min(tIDL_1)+1
#        if np.min(tPYT_0)<1:             tPYT_0=tPYT_0-np.min(tPYT_0)+1
#        if np.min(tPYT_1)<1:             tPYT_1=tPYT_1-np.min(tPYT_1)+1
        tPYT_0=tPYT_0/max(tPYT_0)
        tPYT_1=tPYT_1/max(tPYT_1)
        tIDL_0=tIDL_0/max(tIDL_0)
        tIDL_1=tIDL_1/max(tIDL_1)
        
        tI=np.abs(tIDL_0[:-1]-tIDL_0[1:])/max(tIDL_0)
        tP=np.abs(tPYT_0[:-1]-tPYT_0[1:])/max(tPYT_0)
        
        tmp=np.asarray(np.where(tI>0.3))
        tIindex=np.insert(tmp,[0,np.shape(tmp)[1]],[0,len(tIDL_0)])
        tIm=0*tI
        for iii in range(len(tIindex)-1)  :
            tIm[tIindex[iii]:tIindex[iii+1]]=np.mean(tIDL_0[tIindex[iii]:tIindex[iii+1]])
        
        tmp=np.asarray(np.where(tP>0.3))
        tPindex=np.insert(tmp,[0,np.shape(tmp)[1]],[0,len(tPYT_0)])
        tPm=0*tP
        for iii in range(len(tPindex)-1)  :
            tPm[tPindex[iii]:tPindex[iii+1]]=np.mean(tPYT_0[tPindex[iii]:tPindex[iii+1]])
        
        if 0:
            plt.figure(31), plt.subplot(2,1,1), plt.plot(tIDL_0,'b'), plt.plot(tIm,'r'), plt.show()
            plt.figure(31), plt.subplot(2,1,2), plt.plot(tPYT_0,'b'), plt.plot(tPm,'r'), plt.show()
            
        AA0=np.sum(np.absolute(np.subtract(tIDL_0,tPYT_0)))
        AA1=np.sum(np.absolute(np.subtract(tIDL_1,tPYT_1)))
        AA2=np.sum(np.absolute(tIm-tPm))
        
        s0[jj]=AA0
        s1[jj]=AA1
        s2[jj]=AA2
    ss0[ii,:]=s0
    ss1[ii,:]=s1  
    ss2[ii,:]=s2      
    ss[ii,:]=s0+s1

# from all matches, find the best one
tmp=np.zeros(nIDL)
whr00=tmp.astype(int)
whr01=tmp.astype(int)
whr02=tmp.astype(int)
for ii in range(nIDL): #er wordt nu een eenzijdige match gemaakt, cross correlatie werkt nog niet twee kanten op 
    result=np.amin(ss0[ii,:])     
    whr=np.where(ss0[ii,:]==result) 
    whr00[ii]=int(whr[0][0])
    
    result=np.amin(ss[ii,:])     
    whr=np.where(ss[ii,:]==result) 
    whr01[ii]=int(whr[0][0])
    
    result=np.amin(ss2[ii,:])     
    whr=np.where(ss2[ii,:]==result) 
    whr02[ii]=int(whr[0][0])
    
    resultjj=np.amin(ss0[:,jj])
    whrjj=np.where(ss0[:,jj]==resultjj) 
    whrjj=int(whrjj[0][0])
    
    print([ii,whr00[ii],whr01[ii],whr02[ii]])

# compare donor + acceptor is better than only donor, now visualise and manually sayy whether it is a good match
# while optimizing on ss, 75 out of 186 traces were good. 
goodbad2=    tmp.astype(int)
for ii in range(nIDL):   
    tIDL_0= exp1.files[0].molecules[ii].intensity[0] # +1 to prevent them being zero
    tPYT_0= exp1.files[3].molecules[whr01[ii]].intensity[0]
    tPYT_00= exp1.files[3].molecules[whr00[ii]].intensity[0]
    tPYT_02= exp1.files[3].molecules[whr02[ii]].intensity[0]
    tIDL_1= exp1.files[0].molecules[ii].intensity[1]
    tPYT_1= exp1.files[3].molecules[whr01[ii]].intensity[1]
    tPYT_10= exp1.files[3].molecules[whr00[ii]].intensity[1]
    tPYT_12= exp1.files[3].molecules[whr02[ii]].intensity[1]
    
    plt.figure(5)

    if 0: # from these images found that ss is better than ss0  
        plt.subplot(2,4,1), plt.cla(), plt.plot(tIDL_0), plt.title('donor IDL')
        plt.subplot(2,4,2), plt.cla(), plt.plot(tPYT_0), plt.title('donor PYTHON')
        plt.subplot(2,4,3), plt.cla(), plt.plot(tPYT_00,'g'), plt.title('donor PYTHON')
        plt.subplot(2,4,4), plt.cla(), plt.plot(np.absolute(np.subtract(tIDL_0/max(tIDL_0),tPYT_00/max(tPYT_00))),'r'), plt.title([ii,jj])
        plt.subplot(2,4,5), plt.cla(), plt.plot(tIDL_1), plt.title('acceptor IDL')
        plt.subplot(2,4,6), plt.cla(), plt.plot(tPYT_1), plt.title('acceptor PYTHON')
        plt.subplot(2,4,7), plt.cla(), plt.plot(tPYT_10,'g'), plt.title('acceptor PYTHON')
        plt.subplot(2,4,8), plt.cla(), plt.plot(np.absolute(np.subtract(tIDL_1/max(tIDL_1),tPYT_10/max(tPYT_10))),'r')
    elif 0:
        plt.subplot(2,2,1), plt.cla(), plt.plot(tIDL_0,'r'), plt.title('donor IDL')
        plt.subplot(2,2,1),            plt.plot(tPYT_00,'g'), plt.title('donor PYTHON')
        plt.show()
        plt.subplot(2,2,2), plt.cla(), plt.plot(np.absolute(np.subtract(tIDL_0/max(tIDL_0),tPYT_00/max(tPYT_00))),'r'), plt.title([ii,jj])
        plt.subplot(2,2,3), plt.cla(), plt.plot(tIDL_1,'r'), plt.title('acceptor IDL')
        plt.subplot(2,2,3),            plt.plot(tPYT_10,'g'), plt.title('acceptor PYTHON')
        plt.show()
        plt.subplot(2,2,4), plt.cla(), plt.plot(np.absolute(np.subtract(tIDL_1/max(tIDL_1),tPYT_10/max(tPYT_10))),'r')
    else:
        tI=(tIDL_0[:-1]-tIDL_0[1:])/max(tIDL_0)
        tP=(tPYT_02[:-1]-tPYT_02[1:])/max(tPYT_02)
        plt.subplot(2,2,1), plt.cla(), plt.plot(tIDL_0,'r'), plt.title('donor IDL')
        plt.subplot(2,2,1),            plt.plot(tPYT_00,'g'), plt.title('donor PYTHON')
        plt.show()
        plt.subplot(2,2,2), plt.cla(), plt.plot(tI,'r'), plt.title('donor IDL')
        plt.subplot(2,2,2),            plt.plot(tP+1,'g'), plt.title('donor PYTHON')
        
        plt.subplot(2,2,3), plt.cla(), plt.plot(tIDL_1,'r'), plt.title('acceptor IDL')
        plt.subplot(2,2,3),            plt.plot(tPYT_10,'g'), plt.title('acceptor PYTHON')
        plt.show()
        plt.subplot(2,2,4), plt.cla(), plt.plot(np.absolute((tI-tP)),'r'), plt.title([ii,jj])
            
    
    plt.pause(0.05)
    goodbad2[ii]=input('is this a good trace (1) or bad (0)    ')


