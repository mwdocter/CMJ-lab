# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:45:41 2019

@author: mwdocter

main function = read_one_page, which then refers to reading a tif/pma/sifx file
"""
import os
import re
import tifffile
import numpy as np 
import matplotlib.pyplot as plt
import time
#from sifreader.sifreader import SIFFile
from sifreaderA import SIFFile

from autopick.do_before import clear_all
clear_all()

def read_one_page(image_fn,pageNb): # distributes to specific readers (sif/tif/pma)
    root, name = os.path.split(image_fn)
    for string2compare in ['.pma$','.tif$','.sifx$','.sif$']:
        A=re.search(string2compare,name)
        im0,hdim,vdim,nImages=-1,0,0,0
        if A!=None: 
            #print(name, string2compare)
            if string2compare=='.pma$':
                im0,hdim,vdim,nImages=read_one_page_pma(root,name,pageNb)
            elif A!=None and string2compare=='.tif$':
                im0,hdim,vdim,nImages=read_one_page_tif(root,name,pageNb)
            elif A!=None and string2compare=='.sifx$':
                im0,hdim,vdim,nImages=read_one_page_sifx(root,name,pageNb)
#            elif A!=None and string2compare=='.sif$':
#                im0,hdim,vdim,nImages=read_one_page_sif(root,name,pageNb)
            break
    return im0,hdim,vdim,nImages    
    
def read_one_page_pma(root,name, pageNb=0):
    
    with open(root+'\\'+name, 'rb') as fid:
        hdim = np.fromfile(fid, np.int16,count=1)
        vdim=  np.fromfile(fid, np.int16,count=1)
        hdim=int(hdim[0])
        vdim=int(vdim[0])
    statinfo = os.stat(root+'\\'+name)
    nImages=int((statinfo.st_size-4)/(hdim*vdim))
    
    #determine 8 bits or 16 bits
    A=re.search('_16.pma$',name)
    if A==None: #8 bits
        with open(root+'\\'+name, 'rb') as fid: #did the offset reset?    
            # for image pageNb, 4 for skipping header, plus certain amount of images to read image pageNb
            fid.seek(4+ (pageNb*(hdim*vdim)), os.SEEK_SET)
            im=np.reshape(np.fromfile(fid,np.int8,count=hdim*vdim),(hdim,vdim))
    else:
        with open(root+'\\'+name, 'rb') as fid: #did the offset reset?  
            fid.seek(4+ 2*pageNb*(hdim*vdim), os.SEEK_SET)
            msb=np.reshape(np.fromfile(fid,np.int8,count=(hdim*vdim)),(hdim,vdim))
            lsb=np.reshape(np.fromfile(fid,np.int8,count=(hdim*vdim)),(hdim,vdim))
#            msb = np.core.records.fromfile(fid, 'int8', offset=4+ 2*pageNb*(hdim*vdim), shape=(hdim,vdim)) # for first image
#            lsb = np.core.records.fromfile(fid, 'int8', offset=4+ (1+2*pageNb)*(hdim*vdim), shape=(hdim,vdim)) # for first image
            im=256*msb+lsb;
    return im, hdim,vdim,nImages # still need to convert im
    
def read_one_page_tif(root,name, pageNb=0):
    t = time.time()

    with tifffile.TiffFile(root+'\\'+name) as tif:
#         tif_tags = {}
#         for tag in tif.pages[0].tags.values():
#            name, value = tag.name, tag.value
#            tif_tags[name] = value
#         hdim=tif_tags['ImageWidth']
#         vdim=tif_tags['ImageLength']   
        ### hdim,vdim=tif.pages[0].shape
        
         tifpage=tif.pages
         nImages=(len(tifpage))
  
         if nImages==1:
             #return -1,0,0,0
             0
         elif (nImages-1)>=pageNb:
              im = tifpage[pageNb].asarray()
         else:
              im = tifpage[nImages-1].asarray()
              print('pageNb out of range, printed image {0} instead'.format(nImages))
         hdim,vdim=np.shape(im)
    elapsed = time.time() - t
    print(elapsed),
    return im, hdim,vdim, nImages#, elapsed

    
def read_one_page_sif(root,name,pageNb=0):
    file = SIFFile(root+'\\'+name)
    im = file.read_block(0) 
    hdim,vdim=np.shape(im)
    
    print('sif not working yet')
    return im, 
    
def read_one_page_sifx(root,name,pageNb=0):
 #   filelist = [x for x in os.listdir(root) if x.endswith("spool.dat")]
    A=SIFFile(root+'\\Spooled files.sifx')
    hh=A.height
    ww=A.width
    nImages=A.stacksize
    im=read_one_page_sifx2(root,name,pageNb,A)
#   
    return im, hh,ww, nImages
   
def read_one_page_sifx2(root,name, pageNb,A):
     count=A.height*A.width*3//2
     #name should follow from A.filelist
     with open(root+'\\'+A.filelist[pageNb], 'rb') as fid:
          raw=np.uint16(np.fromfile(fid,np.uint8,count))
     
     #print([A.height,A.width,A.stacksize,np.shape(raw)])        
     ii=np.array(range(1024*1024))
     AA=raw[ii*6+0]*16 + (raw[ii*6+1]%16)
     BB=raw[ii*6+2]*16 + raw[ii*6+1]//16
     CC=raw[ii*6+3]*16 + raw[ii*6+4]%16
     DD=raw[ii*6+5]*16 + raw[ii*6+4]//16 
          
     ALL=np.uint16(np.zeros(A.height*A.width))
     ALL[0::4] = AA
     ALL[1::4] = BB
     ALL[2::4] = CC
     ALL[3::4] = DD
          
     im=np.reshape(ALL,(A.height, A.width))
     im=np.rot90(im)     
     if 0: # for testing match real data
         plt.imshow(im)
         tifffile.imwrite(root+"Python image.tif" , im ,  photometric='minisblack')
        
     return im   
# MATLAB code    
#    filename='0000000000spool.dat'
#    % filedir='D:\data\personal data\20190110 testing matlab image conversion\spool\';
#    
#    file = fopen([filedir,filename],'rb')
#    [data] = fread(file,'uint8=>double',0);
#    fclose(file);
#    %
#    dd=0; %images for dd 0:3:39 look reasonable, but not identical
#    ii=1:(1024*1024); %floor(numel(data)/6)=1048582, 1024*1024=1048576
#    AA=data(dd+(ii-1)*6+1)*(2^4) +rem(dd+data((ii-1)*6+2),2^4);
#    BB=data(dd+(ii-1)*6+3)*(2^4) +floor(dd+data((ii-1)*6+2)/(2^4));
#    CC=data(dd+(ii-1)*6+4)*(2^4) +rem(dd+data((ii-1)*6+5),2^4);
#    DD=data(dd+(ii-1)*6+6)*(2^4) +floor(dd+data((ii-1)*6+5)/(2^4));
#    
#    ALL=zeros(2048);
#    ALL(1:4:end)=AA(1:end);
#    ALL(2:4:end)=BB(1:end);
#    ALL(3:4:end)=CC(1:end);
#    ALL(4:4:end)=DD(1:end);
#        
#    ALLm=ALL(end:-1:1,:);    
    
# BELOW A TEST IS DESCRIBED
#if 0: #test pma
#    mainPath='H:\\projects\\research practicum\\single molecule fluorescence\\Matlab\\HJA-data from Ivo'
#elif 0: #test multipage tif
#    mainPath='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\180920 super resolution folder\\190326 data superres C1 D1-210'
#elif 1:
#    #mainPath='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\older\\170111 cmos camera\\data\\170228 cy37'
#    mainPath='E:\\CMJ trace analysis\\test data'
#elif 0: 
#    mainPath='N:\\tnw\\BN\\CMJ\\Shared\\Margreet\\181218 - First single-molecule sample (GattaQuant)\\RawData'
##read in pma data
#for root, dirs, fileNames in os.walk(mainPath, topdown=False):
#    for name in fileNames:
#        # detect whether you would like to read a pma file, sif(x), or multipage tif
#        im0,hdim,vdim,nImages=read_one_page(root,name,0)
#        im1=read_one_page(root,name,1)[0]
#        im100=read_one_page(root,name,100)[0]
#        if type(im0)!=int:
#            plt.figure(1)
#            plt.subplot(1,3,1)
#            plt.imshow(im0)
#            plt.subplot(1,3,2)
#            plt.imshow(im1)
#            plt.subplot(1,3,3)
#            plt.imshow(im100)
#            plt.show()
#        
#            break #for testing do only the first one in the file directory
#        