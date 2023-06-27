#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__date__ = 'MAY 2023'
__version__ = '0.1'
__status__ = "initial release"




import numpy as np
import pandas as pd
import cupy as cp
cp._default_memory_pool.free_all_blocks()
from cupyx.scipy.ndimage import label as cplabel
pd.options.mode.chained_assignment = None
import os
import time
#import skimage.io as io
# import rasterio
# from rasterio.windows import from_bounds
# from rasterio.enums import Resampling
from osgeo import gdal
from scipy.ndimage import generate_binary_structure

'''user inputs'''
ClassPath=''#path to the class rasters
ResultsFile='.csv' #this file needs to exist
DistribFolder=''#path for the bar size distribution results, a series of small numpy arrays
Month='M'#2 digit month with M prefix as a string


'''functions'''
def tic():
    #Homemade version of matlab tic and toc functions

    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():

    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str((time.time() - startTime_for_tictoc)) + " seconds.")
    else:
        print ("Toc: start time not set")
        



def GetCRS(letter, zone):
  
  North=['N','P', 'Q', 'R','S','T','U','V','W', 'X']
  
  if letter in North:
    thisCRS='EPSG:326'+str(zone).zfill(2)
  else:
    thisCRS='EPSG:327'+str(zone).zfill(2)
  
  return thisCRS


'''Main'''
s=generate_binary_structure(2, 2)

    
#ClassPath='/media/patrice/DataDrive/GORiM/GORiM_gen2/'+Month[m]+'/'


ClassSheet=pd.read_csv(ResultsFile)


for gzd in range(len(ClassSheet.Zone)):
    tic()
    #outmask='/home/patrice/Documents/FastData/GolbalGrid/TempGridImage'+str(gzd)+'.tif'
    ULUTMx=ClassSheet.UTMllx[gzd]
    ULUTMy=ClassSheet.UTMury[gzd]
    LRUTMx=ClassSheet.UTMurx[gzd] 
    LRUTMy=ClassSheet.UTMlly[gzd]
    UL4326x=ClassSheet.LLx[gzd]
    UL4326y=ClassSheet.URy[gzd]
    LR4326x=ClassSheet.URx[gzd] 
    LR4326y=ClassSheet.LLy[gzd]
    # ULmapx=ClassSheet.UTMllx[gzd]
    # ULmapy=ClassSheet.UTMury[gzd]
    # LRmapx=ClassSheet.UTMurx[gzd] 
    # LRmapy=ClassSheet.UTMlly[gzd]
    thiscrs=GetCRS(ClassSheet.GZD[gzd], ClassSheet.Zone[gzd])
    MaskName=ClassPath+'Class_S2_'+Month+'_Z'+str(int(ClassSheet.Zone[gzd])).zfill(2)+str(ClassSheet.GZD[gzd])+'.tif'
    if os.path.isfile(MaskName):
        #the the class raster subimage
    # try:
    #     with rasterio.open(MaskName) as src:
    #         ClassRaster = src.read(1, window=from_bounds(ULUTMx, LRUTMy, LRUTMx, ULUTMy, src.transform))
    #         #ClassRaster1 = src.read(1, window=from_bounds(ULmapx, LRmapy, LRmapx, ULmapy, src.transform))
    # except:
        ds = gdal.Open(MaskName)
        ds = gdal.Translate('', ds, projWin = [ULUTMx, ULUTMy, LRUTMx, LRUTMy,], format='MEM')
        ClassRaster=ds.ReadAsArray()
        ds = None

        
        #start pixel counts
        ClassSheet.Done[gzd]=1   
        ClassSheet.BackPix[gzd]=np.count_nonzero(ClassRaster==0)
        ClassSheet.RiverPix[gzd]=np.count_nonzero(ClassRaster==1)
        ClassSheet.LakePix[gzd]=np.count_nonzero(ClassRaster==2)
        ClassSheet.BarPix[gzd]=np.count_nonzero(ClassRaster==3)
        ClassSheet.OceanPix[gzd]=np.count_nonzero(ClassRaster==4)
        ClassSheet.GlacialPix[gzd]=np.count_nonzero(ClassRaster==5)
        ClassSheet.SnowPix[gzd]=np.count_nonzero(ClassRaster==6)
        ClassSheet.CloudPix[gzd]=np.count_nonzero(ClassRaster==7)
        ClassSheet.NoPix[gzd]=np.count_nonzero(ClassRaster==8)

        if ClassSheet.BackPix[gzd]+ClassSheet.RiverPix[gzd]+ClassSheet.LakePix[gzd]+ClassSheet.SnowPix[gzd] != 0:
            #NoGroundFrac=(ClassSheet.CloudPix[gzd]+ClassSheet.NoPix[gzd])/(ClassSheet.BackPix[gzd]+ClassSheet.RiverPix[gzd]+ClassSheet.LakePix[gzd]+ClassSheet.BarPix[gzd]+ClassSheet.GlacialPix[gzd]+ClassSheet.SnowPix[gzd]+ClassSheet.CloudPix[gzd]+ClassSheet.NoPix[gzd])
            #ClassSheet.CloudFactor[gzd]=1-NoGroundFrac
            RD=np.float32(1000000*ClassSheet.RiverPix[gzd])/float(ClassSheet.BackPix[gzd]+ClassSheet.BarPix[gzd]+ClassSheet.RiverPix[gzd]+ClassSheet.LakePix[gzd]+ClassSheet.SnowPix[gzd]+ClassSheet.GlacialPix[gzd])
            LD=np.float32(1000000*ClassSheet.LakePix[gzd])/float(ClassSheet.BackPix[gzd]+ClassSheet.BarPix[gzd]+ClassSheet.RiverPix[gzd]+ClassSheet.LakePix[gzd]+ClassSheet.SnowPix[gzd]+ClassSheet.GlacialPix[gzd])
            BD=np.float32(1000000*ClassSheet.BarPix[gzd])/float(ClassSheet.BackPix[gzd]+ClassSheet.BarPix[gzd]+ClassSheet.RiverPix[gzd]+ClassSheet.LakePix[gzd]+ClassSheet.SnowPix[gzd]+ClassSheet.GlacialPix[gzd])
            ClassSheet.River_Density[gzd]=RD#np.float32(10000*ClassSheet.RiverPix[gzd])/np.float32(ClassSheet.BackPix[gzd]+ClassSheet.RiverPix[gzd]+ClassSheet.LakePix[gzd]+ClassSheet.BarPix[gzd]+ClassSheet.SnowPix[gzd])
            ClassSheet.Lake_Density[gzd]=LD#np.float32(10000*ClassSheet.LakePix[f])/np.float32(ClassSheet.BackPix[f]+ClassSheet.RiverPix[f]+ClassSheet.LakePix[f]+ClassSheet.BarPix[f]+ClassSheet.SnowPix[f])
            ClassSheet.Bar_Density[gzd]=BD
        else:
            ClassSheet.CloudFactor[gzd]=-1
            
        #bar object data

        bars=cp.array(ClassRaster==3, dtype='uint8')
        cpBarLabels=cplabel(bars, structure=s, output='int32')[0]
        del bars

        x, count=cp.unique(cpBarLabels, return_counts=True)
        del cpBarLabels
        x=x[1:]
        count=count[1:]
        #save the counts
        outname=os.path.join(DistribFolder, 'BarDist_'+str(gzd)+'.npy')
        np.save(outname, cp.asnumpy(count))


        cp._default_memory_pool.free_all_blocks()
        ClassSheet.Nbar[gzd]=len(x)
        if len(x)>0:
            ClassSheet.MedianBarArea[gzd]=np.median(count)

        
        cp._default_memory_pool.free_all_blocks()
            
        
        toc()
        print('processing '+os.path.basename(MaskName))
        
        
#save the csv
ClassSheet.to_csv(ResultsFile, index=False)
    #ClassSheet=None