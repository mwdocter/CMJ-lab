# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:17:13 2019

@author: mwdocter

compare the use of P&Q with use of transform_matrix -> same?
"""
import numpy as np 
import cv2


fn_IDL='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\mapping\\rough.map';
fn_Python='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies\\hel4-P.map'
fn_Python_COEFF='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies\\hel4-P.coeff'

root='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\20181203_HJ_training\\#3.10_0.1mg-ml_streptavidin_50pM_HJC_G_movies'
name='hel4.pma'

# load saved transform_matrix
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

# load saved P,Q    
P=np.zeros((4,4))   
tm=np.linalg.inv(transform_matrix)      
P[0,0]=tm[0,2]
P[0,1]=tm[0,1]
P[1,0]=tm[0,0]
Q=np.zeros((4,4))         
Q[0,0]=tm[1,2]
Q[0,1]=tm[1,1]
Q[1,0]=tm[1,0]

# generate new P,Q from saved transform matrix
tmp = np.genfromtxt(fn_Python) 
Ppyt=tmp.reshape(8,4)[0:4,:]
Qpyt=tmp.reshape(8,4)[4:,:]

# load saved points
ptsG=[]
dstG=[]
with open(root+'\\'+name[:-4]+'-PC.pks', 'r') as infile:
    cc=0
    for line in infile:
        cc=cc+1
        if cc%2 != 0: #odd
            A=[float(ii) for ii in line.split()]
            ptsG.append(A[1:3])
        else:
            B=[float(ii) for ii in line.split()]
            dstG.append(B[1:3])
pts_number=len(ptsG)
ptsG=np.array(ptsG)
dstG=np.array(dstG)

# test 1: generate new xy positions from perspective transform

ptsG=np.array(ptsG)
dstG1= cv2.perspectiveTransform(ptsG.reshape(-1,1,2),np.linalg.inv(transform_matrix))
dstG1= dstG.reshape(-1,2)

# test 2: generate new xy positions from saved P,Q
dstG2=np.zeros(np.shape(dstG))
for LL in range( len(dstG)):
    new0=0
    new1=0
    old0=ptsG[LL][0]
    old1=ptsG[LL][1]
    for iii in range(4):
        for jjj in range(4):
            new0=new0+P[iii,jjj]* (old0**iii)*(old1**jjj)
            new1=new1+Q[iii,jjj]* (old0**iii)*(old1**jjj)
    dstG2[LL][0]=new0
    dstG2[LL][1]=new1

#test 3: generate new xy positions from generated P,Q
dstG3=np.zeros(np.shape(dstG))
for LL in range( len(dstG)):
    new0=0
    new1=0
    old0=ptsG[LL][0]
    old1=ptsG[LL][1]
    for iii in range(4):
        for jjj in range(4):
            new0=new0+P[iii,jjj]* (old0**iii)*(old1**jjj)
            new1=new1+Q[iii,jjj]* (old0**iii)*(old1**jjj)
    dstG3[LL][0]=new0
    dstG3[LL][1]=new1
    