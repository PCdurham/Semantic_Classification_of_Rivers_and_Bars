#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__date__ = 'Jan 2023'
__version__ = '0.1'
__status__ = "initial release"


'''
Name:           PostProcess.py
Compatibility:  Python 3.9
Description:    Takes the outputs of GEEinference and produces final class rasters

Requires:       numba, numpy, scipy, skimage, sklearn, gdal, gdal_array, osgeo
                

this script will apply vector filters from global datasets.  
Does not use the GPU and only uses 1 CPU core so can run in parallel to the main classification script
Resilting class scheme is:
    1- river water
    2- lake water
    3- exposed sediment bars
    4- seas and oceans
    5- glaciated terrain
    6- snow
    7- cloud
    8- no data
    
IMPORTANT: Classes 4 to 7 have not been classified by a model, they are the result of a filter op.


'''

import os
import numpy as np
import numba as nb
import pandas as pd
from skimage import io
import glob
try:
    from osgeo import gdal, gdal_array, osr
except:
    import gdal_array, gdal
    import osr
import time

from skimage.measure import regionprops
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import label as sclabel
from scipy.ndimage import generate_binary_structure
import fill_voids
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 10000000000





PredictPath = ''#location of puotputs from GEEinference.py
ResultPath=''#write location for final class rasters
MetaPath=''#location of metadata for filters
ParameterFile=''#Location of numpy array with results of parameter optimisation
skipLAST=True #set to true if GEEinference is running in parallel
NorthList=['V','W', 'X']
'''Get optimised parameters'''
ErrMatrix=np.load(ParameterFile)
p1=np.float32(ErrMatrix[np.argmin(ErrMatrix[:,0]),3])
p2=np.float32(ErrMatrix[np.argmin(ErrMatrix[:,0]),4])
p4=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),5])
p5=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),6])
p6=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),7])
p7=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),8])
p8=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),9])
p9=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),10])
p10=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),11])
p11=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,2]),12])
p12=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),13])
p13=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,0]),14])
p14=np.uint8(ErrMatrix[np.argmin(ErrMatrix[:,2]),15])


'''function defs'''



def GDALpix2map(ds, x, y):
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    posX = px_w * x + rot1 * y + xoffset
    posY = rot2 * x + px_h * y + yoffset
    # shift to the center of the pixel
    posX += px_w / 2.0
    posY += px_h / 2.0
    return posX, posY 


def tic():
    #Homemade version of matlab tic and toc functions

    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():

    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str((time.time() - startTime_for_tictoc)) + " seconds.")
    else:
        print ("Toc: start time not set")
        
        
def Im_to_EPSG4326(Imfile):
    tempfile1='/home/patrice/Documents/FastData/Tempfiles/imtemp4326.tif'
    command1='gdalwarp -q -overwrite -co NUM_THREADS=12 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 -t_srs EPSG:4326 '+Imfile+' '+tempfile1
    os.system(command1)
    Im4326=io.imread(tempfile1)
    Imshape=Im4326.shape
    Im4326=None
    return Imshape
    
 


#numba functions
@nb.jit(parallel=True, nopython=True)  
def nbputmask(mask, condition, value): #20% faster than nunpy on very large arrays. 
    #print('jit')
    maskshape=mask.shape
    mask=mask.ravel()
    condition=condition.ravel()
    n=len(mask)
    for i in nb.prange(n):
        if condition[i]:
            mask[i]=value
    return mask.reshape(maskshape)

@nb.jit(parallel=True, nopython=True)  
def nblogical_and(A, B): #faster than nunpy on very large arrays. 
    #print('jit')
    Ashape=A.shape
    A=A.ravel()
    B=B.ravel()
    n=len(A)
    C=np.zeros(n, dtype='uint8')
    for i in nb.prange(n):
        C[i]=A[i]*B[i]
    C=C.reshape(Ashape)

    return C==1

def nbRemoveSmall(ClassRaster1, Labels):
    x, counts=np.unique(Labels, return_counts=True)
    smallBods=[]
    for o in range(len(x)):
        if counts[o]<11:
            smallBods.append(x[o])
    cpsmall=np.int32(smallBods)
    #del Labels
    Labels1D=np.int32(Labels.ravel())
    Mask1D=is_in_set_pnb(Labels1D, cpsmall)
    Mask=Mask1D.reshape(ClassRaster1.shape)
    del Mask1D
    ClassRaster1=nbputmask(ClassRaster1, Mask, 0)

    del Mask
    return ClassRaster1

@nb.jit(parallel=True, nopython=True)
def is_in_set_pnb(a, b):
    #print('jit')
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)

@nb.jit(parallel=True, nopython=True)
def isnot_in_set_pnb(a, b):
    #print('jit')
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] not in set_b:
            result[i] = True
    return result.reshape(shape)

@nb.jit(parallel=True, nopython=True) 
def nbtimes(A,B):
    n = len(A)
    C=np.zeros(A.shape, dtype='int32')
    for i in nb.prange(n):
        C[i]=A[i]*B[i]
    return C



def RiverSHEDS(ClassRaster1, MetaMask, ProbImage50, ProbImage10):
    if isinstance(ClassRaster1, tuple):
        ClassRaster1=ClassRaster1[0]

    Lakes=ClassRaster1==2
    if np.count_nonzero(MetaMask==1)>0:
        
        MetaRivers=(MetaMask!=1)
        del MetaMask
    #weal lakes in the metamask
        

        LikelyRivers=np.logical_and(ProbImage50[:,:,0]<200, np.logical_or(ProbImage50[:,:,1]>p4, ProbImage50[:,:,2]>p5))
        LikelyRivers=np.logical_and(LikelyRivers, np.logical_not(ClassRaster1==3))#exclude the bars
        #LikelyRivers=np.logical_and(ProbImage50[:,:,1]<200,ProbImage10[:,:,2]>30)
        #LikelyRivers=np.logical_and(ProbImage50[:,:,1]<200,ClassRaster1==2)
        #remove small connected channels
        ThinLikelyRivers=binary_erosion(LikelyRivers, iterations=5)
        FatLikelyRivers=binary_dilation(ThinLikelyRivers, iterations=6)
        LikelyRivers=np.logical_and(FatLikelyRivers, LikelyRivers)
        del ThinLikelyRivers, FatLikelyRivers
        #LikelyRivers=nblogical_and(Lakes, WeakWater)
        #del WeakWater
        LikelyRiverObjects=sclabel(LikelyRivers, output='int32')[0]
        print(str(np.amax(LikelyRiverObjects))+' total river objects')
        if np.amax(LikelyRiverObjects)<32766:
            LikelyRiverObjects=np.int16(LikelyRiverObjects)
        x, counts=np.unique(LikelyRiverObjects, return_counts=True)
        counts[0]=0 #get rid of background
        np.save('/home/patrice/Documents/FastData/Tempfiles/templabels.npy',LikelyRiverObjects)
        LikelyRiverObjects=nbputmask(LikelyRiverObjects,MetaRivers, 0)#MetaRivers*LikelyRiverObjects
        #LikelyRiverObjects=LikelyRiverObjects.ravel()

        xmasked=pd.unique(LikelyRiverObjects.ravel())
        del LikelyRiverObjects
        #del MaskedLikelyRivers
        riverbods=[]

        for o in range(len(xmasked)):
            #y=np.where(x==o)
            if counts[xmasked[o]]>224*224:
                riverbods.append(xmasked[o])
                #ClassRaster1=nbputmask(ClassRaster1, LikelyRiverObjects==xmasked[o],1)

        Labels1D=np.load('/home/patrice/Documents/FastData/Tempfiles/templabels.npy').ravel()
        Labels1D=np.int32(Labels1D)
        cpsmall=np.int32(riverbods)
        #del Labels
        Mask1D=is_in_set_pnb(Labels1D, cpsmall)
        del Labels1D
        Mask=Mask1D.reshape(ClassRaster1.shape)
        ClassRaster1=nbputmask(ClassRaster1, Mask, 1)
        del Mask1D, Mask
        #weak rivers in the metamask
        WeakRivers=nblogical_and(ProbImage50[:,:,0]<p6, ProbImage50[:,:,1]>p7)
        WeakRivers=nblogical_and(np.logical_not(ClassRaster1==1), WeakRivers)
        LikelyRiverObjects=sclabel(WeakRivers, output='int32')[0]
        if np.amax(LikelyRiverObjects)<32766:
            LikelyRiverObjects=np.int16(LikelyRiverObjects)
        del WeakRivers
        
        x, counts=np.unique(LikelyRiverObjects, return_counts=True)
        counts[0]=0 #get rid of background
        np.save('/home/patrice/Documents/FastData/Tempfiles/templabels.npy',LikelyRiverObjects)
        LikelyRiverObjects=nbputmask(LikelyRiverObjects,MetaRivers, 0)#MetaRivers*LikelyRiverObjects

        #MaskedLikelyRivers=MetaRivers*WeakRiverObjects
        #WeakRiverObjects=WeakRiverObjects.ravel()
        #del WeakRiverObjects
        xmasked=pd.unique(LikelyRiverObjects.ravel())
        del LikelyRiverObjects
        #del MaskedLikelyRivers
        riverbods=[]
        for o in range(len(xmasked)):
            #y=np.where(x==o)
            if counts[xmasked[o]]>224*224*0.5:
                riverbods.append(xmasked[o])
                #ClassRaster1=nbputmask(ClassRaster1, WeakRiverObjects==xmasked[o],1)
        Labels1D=np.load('/home/patrice/Documents/FastData/Tempfiles/templabels.npy').ravel()
        Labels1D=np.int32(Labels1D)
        cpsmall=np.int32(riverbods)
        #del Labels
        Mask1D=is_in_set_pnb(Labels1D, cpsmall)
        del Labels1D
        Mask=Mask1D.reshape(ClassRaster1.shape)
        ClassRaster1=nbputmask(ClassRaster1, Mask, 1)
        del Mask1D, Mask     
        #del WeakRiverObjects
    #lake slivers in river and in the metamask
    Rivers=binary_dilation(ClassRaster1==1, iterations=5)
    slivers=nblogical_and(Lakes, Rivers)
    ClassRaster1=nbputmask(ClassRaster1, slivers, 1)
    
        
    return ClassRaster1



def LakeSHEDS2(ClassRaster1, MetaMask, ProbImage50, ProbImage10):
    Lake=MetaMask==2
    #from 5 to 125 km2
    #catch false background in the middle
    weakwater=np.logical_or(ProbImage50[:,:,0]>p8, ProbImage50[:,:,1]>p9)
    likelylakes=nblogical_and(Lake, weakwater)
    ClassRaster1=nbputmask(ClassRaster1, likelylakes, 2)
    del weakwater
    del likelylakes
    
    
    MetaLakes1=binary_dilation(Lake, iterations=5)
    MetaLakes=binary_dilation(MetaLakes1, iterations=5)
    BadRiver=nblogical_and(MetaLakes, ClassRaster1==1)
    BadRiver=nblogical_and(BadRiver, ProbImage10<p10)
    ClassRaster1=nbputmask(ClassRaster1, BadRiver, 2)
    ProbImage50[:,:,0]=nbputmask(ProbImage50[:,:,0], MetaLakes1, 200)#ensure no rivers within 50m of a HydroSHEDS lake
    
    #125+ just set the interior of megalakes to lake
    MegaLakes=MetaMask==3
    if np.count_nonzero(MegaLakes)>0:
        MegaLakes=binary_dilation(MegaLakes, iterations=50)
        # MegaLakes=nblogical_and(MetaMask==3, ClassRaster1==2)
        filledlakes=fill_voids.fill(MegaLakes, in_place=False)
        # del MegaLakes
        truelakes=nblogical_and(ClassRaster1==1, filledlakes)
        
        ClassRaster1=nbputmask(ClassRaster1, truelakes, 2)
        ProbImage50[:,:,0]=nbputmask(ProbImage50[:,:,0], truelakes, 200)
    
           
        
        
    return ClassRaster1, ProbImage50

#Image Processing and binary morphology filters

def FixSmallWaterBody(ClassRaster1, clear10s):

    WaterBodies=ClassRaster1==1#np.logical_or(ClassRaster1==1, ClassRaster1==2)
    s = generate_binary_structure(2,2) #8-connectedness
    Labels=sclabel(WaterBodies, output='int32', structure=s)[0]


    Props=regionprops(Labels)
    smallBods=[]
    for o in range(len(Props)):
        majAX=Props[o].major_axis_length
        if majAX>0:
            if (Props[o].minor_axis_length/majAX)>p1: 
                if Props[o].solidity>p2:
                    Ar=Props[o].area
                    if Ar<100000 and Ar>=10:
                        smallBods.append(Props[o].label)
    print('Found '+str(len(smallBods))+ ' small compact water bodies to set to lakes')
    Labels1D=np.int32(Labels.ravel())
    cpsmall=np.int32(smallBods)
    #del Labels
    Mask1D=is_in_set_pnb(Labels1D, cpsmall)
    Mask=Mask1D.reshape(ClassRaster1.shape)
    if clear10s:
        ClassRaster1 =nbRemoveSmall(ClassRaster1, Labels)
        WaterBodies=ClassRaster1==2
        Labels=sclabel(WaterBodies, output='int32', structure=s)[0]
        ClassRaster1 =nbRemoveSmall(ClassRaster1, Labels)
    del Mask1D
    ClassRaster1=nbputmask(ClassRaster1, Mask, 2)
    del Mask

    
    return ClassRaster1



def CleanBars(ClassRaster):
    bars=ClassRaster==3
    if np.count_nonzero(bars)>0:
        river=binary_dilation(ClassRaster==1, iterations=2)
        s = generate_binary_structure(2,2) #8-connectedness
        barlabel=sclabel(bars, output='int32', structure=s)[0]
        if np.amax(barlabel)<32766:
            print('int16 bar labels')
            barlabel=np.int16(barlabel)
        del bars
        contactbarlabel=nbtimes(barlabel.ravel(), river.ravel())
        del river
        barlabel=barlabel.ravel()
        barobjects=pd.unique(contactbarlabel)
        #Mask=np.ones(ClassRaster.shape, dtype='bool').ravel()==0
        Mask=isnot_in_set_pnb(barlabel, barobjects).reshape(ClassRaster.shape)
        newbars=nbputmask(barlabel.reshape(ClassRaster.shape), Mask, 0)>0
        del Mask
        ClassRaster=nbputmask(ClassRaster, ClassRaster==3, 0)
        ClassRaster=nbputmask(ClassRaster, newbars, 3)
    return ClassRaster


def Rivers_in_Lakes(ClassRaster):
    # if river water is fully in lake water, make it lake water.     Rivers=ClassRaster==1
    Lake_filled=fill_voids.fill(ClassRaster==2, in_place=False)#binary_fill_holes(ClassRaster==2)
    RiversinLakes=nblogical_and(Lake_filled, ClassRaster==1)
    ClassRaster=nbputmask(ClassRaster, RiversinLakes, 2)
    del Lake_filled
    del RiversinLakes
    #do the reverse
    # R_filled=fill_voids.fill(ClassRaster==1, in_place=False)#binary_fill_holes(ClassRaster==2)
    # LakesinRivers=nblogical_and(R_filled, ClassRaster==2)
    # ClassRaster=nbputmask(ClassRaster, LakesinRivers, 1)
    return ClassRaster



def Lake2Ocean(ClassRaster1):
    #any lake that touches the ocean is set to ocean


    FatOcean=binary_dilation(ClassRaster1==4)
    strip=nblogical_and(ClassRaster1==2, FatOcean)
    xstrip, ystrip = np.where(strip)
    FatOcean=None
    s = generate_binary_structure(2,2) #8-connectedness
    LakeObjects=sclabel(ClassRaster1==2, structure=s, output='int32')[0]
    if np.amax(LakeObjects)<32766:
        LakeObjects=np.int16(LakeObjects)
    striploc=np.zeros((len(xstrip),1))
    for i in range(len(xstrip)):
        striploc[i]=LakeObjects[xstrip[i], ystrip[i]]
    numloc, countloc=np.unique(striploc, return_counts=True)
    print('Found '+str(len(numloc))+' lakes in contact with ocean')
    Raster1D=LakeObjects.reshape(-1,1)
    del LakeObjects
    Mask1D=is_in_set_pnb(Raster1D, numloc)
    Mask=Mask1D.reshape(ClassRaster1.shape)
    #ClassRaster[Mask]=4
    ClassRaster1=nbputmask(ClassRaster1, Mask, 4)

    return ClassRaster1

def River2Ocean(ClassRaster):
    FatOcean=binary_dilation(ClassRaster==4, iterations=15)
    badrivers=nblogical_and(FatOcean,ClassRaster==1)
    ClassRaster=nbputmask(ClassRaster, badrivers, 4)
    return ClassRaster


'''MAIN CODE'''
'''prime numba functions'''

A=np.random.randint(low=0, high=10, size=(1000,1000))==5
B=np.random.randint(low=0, high=10, size=(1000,1000))==5
C=np.int32(np.random.randint(low=0, high=10, size=(1000,1000)))
static=np.int32([5,1])

D=nblogical_and(A,B)

E=nbputmask(A,B, 2)

E=is_in_set_pnb(C, static)

Labels=sclabel(A, output='int32')[0]

F=nbRemoveSmall(A, Labels)
del A, B, C, static, D, E, F, Labels

'''get the folder list and loop through'''
folderlist=glob.glob(PredictPath+'*/')
folderlist.sort()


for f in range(1, len(folderlist)-int(skipLAST)):
    tic()
    '''Merge the split class files to a temp file'''
    command='gdalwarp -q -overwrite -tr 10.0 10.0 -r near -co BIGTIFF=YES -co NUM_THREADS=20 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 -ot Byte -of GTiff '
    folder=folderlist[f]
    imagelist=glob.glob(folder+'Class*.tif')
    if len(imagelist)!=0:
        IminProgress=os.path.basename(imagelist[0])[0:17]

        finalClass=os.path.join(ResultPath, IminProgress+'.tif')
        if not os.path.isfile(finalClass):#skip existing images

            print('Merging tiles for '+IminProgress)
            mergename='/home/patrice/Documents/FastData/Tempfiles/'+os.path.basename(imagelist[0])[0:17]+'.tif'
            filelist=[None]*len(imagelist)
            for i,im in enumerate(imagelist):
                command=command+im+' '
            command=command+mergename
            os.system(command)
            command=None
            C=io.imread(mergename)
            

            '''apply global vector masks'''
            
            if isinstance(C, tuple):
                ClassRaster=C[0]
            else:
                ClassRaster=C
            del C
            ClassRaster1=ClassRaster[:,:,0]
            ProbImage50=ClassRaster[:,:,1:4]
            ProbImage10=ClassRaster[:,:,4]
            nodata=ClassRaster1==8
            
            del ClassRaster

        
            print('Applying global HydroSHEDS masks...')
            MetaMaskFile=os.path.join(MetaPath, 'Meta_'+IminProgress[13:]+'.tif')
            MetaMask=gdal_array.LoadFile(MetaMaskFile)
            print('HydroSHED large lakes')
            ClassRaster1, ProbImage50 =LakeSHEDS2(ClassRaster1, MetaMask, ProbImage50, ProbImage10)

            print('HydroSHED large rivers')
            ClassRaster1=RiverSHEDS(ClassRaster1, MetaMask, ProbImage50, ProbImage10)
            del ProbImage10
            del ProbImage50
            print('small water bodies')
            ClassRaster1=FixSmallWaterBody(ClassRaster1, clear10s=True)
            ClassRaster1=Rivers_in_Lakes(ClassRaster1)

                

            #re-impose no data areas
            ClassRaster1=nbputmask(ClassRaster1,nodata ,8)
            #ocean goes last so that sonw, no data and cloud areas in the ocean are ignored
            if np.count_nonzero(MetaMask==4)>0:
                print('cleaning ocean edges')
                cond=MetaMask==4
                ClassRaster1=nbputmask(ClassRaster1, cond,4)
                ClassRaster1 =Lake2Ocean(ClassRaster1)
                ClassRaster1 =River2Ocean(ClassRaster1)
                del cond
                
            print('re-imposing bar criteria') #bars must touch a river
            ClassRaster1=CleanBars(ClassRaster1)

            
            del MetaMask
            print('Saving geotif to disk')
            name=os.path.basename(im)[0:17]+'.tif'
            #name='Class_S2_M08_Z01W.tif'
            SavedClass=os.path.join(ResultPath, name)
            ImageSize=ClassRaster1.shape
            ImageFile = gdal.Open(MetaMaskFile)
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(SavedClass, ImageSize[1], ImageSize[0], gdal.GDT_Byte, options=['NUM_THREADS=20','COMPRESS=DEFLATE','PREDICTOR=2','ZLEVEL=9'])
            outdata.SetGeoTransform(ImageFile.GetGeoTransform())
            outdata.SetProjection(ImageFile.GetProjection())
            outdata.GetRasterBand(1).WriteArray(ClassRaster1)
            ct = gdal.ColorTable()
            # Some examples
            ct.SetColorEntry( 1, (45, 150, 245) )
            ct.SetColorEntry( 2, (5, 255, 125) )
            ct.SetColorEntry( 3, (255, 0, 0) )
            ct.SetColorEntry( 4, (0, 0, 255) )
            ct.SetColorEntry( 5, (150, 245, 240) )
            ct.SetColorEntry( 6, (220, 220, 220) )
            ct.SetColorEntry( 7, (125, 125, 125) )
            ct.SetColorEntry( 8, (0, 0, 0) )
            outdata.GetRasterBand(1).SetRasterColorTable( ct )
            outdata.GetRasterBand(1).SetNoDataValue(0)
            outdata.FlushCache() ##saves to disk!!
            outdata = None
            ImageFile = None
            ct = None
            

            os.remove(mergename)
            print('Finished post-processing '+IminProgress)
            toc()
            print(' ')
            #np.save(flag, [1])#tiny  numpy file to mark completion
            
    else:
        print('no classified images in '+folder)
        
        
