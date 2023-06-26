#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__date__ = 'Jan 2023'
__version__ = '0.1'
__status__ = "initial release"


"""
Name:           GEEinference.py
Compatibility:  Python 3.9
Description:    Applies a pre-trained CNN to to tiles extracted from Google Earth
                Engine.  Follows up with a set of filters running on the GPU.

Requires:       Tensorflow 2.5 or above (GPU version, 8Gb minimum), numpy, numba, skimage, sklearn
                gdal, fill_voids. 

"""

###############################################################################

""" Libraries"""
'''Control tensorflow verbosity'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
#from tensorflow.keras.models import load_model
import tensorflow as tf
from tiramisu_tf2 import tiramisu
from tensorflow.keras import optimizers
import numpy as np
import numba as nb
from skimage import io
#from skimage.measure import label, regionprops
from skimage.transform import downscale_local_mean
from skimage.morphology import remove_small_objects
from skimage.morphology import skeletonize
from skimage.measure import regionprops
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as sclabel
from scipy.ndimage import generate_binary_structure
import fill_voids
import os
#import shutil
import glob
try:
    from osgeo import gdal
except:
    import gdal
      
import gc
import time
import copy
 
#time.sleep(60*45)                                
###############################################################################


"""User data input. Fill in the info below before running"""
#drop the serialised model in this path
modelpath='.hdf5'#model weights path in hdf5
#drop the outputs from GEE in this path. Needs to have subfolders with the data in the form GEE_MXX_ZYYY where XX is the month from 01 to 12 and YYY is the GZD, eg 32T or 01W
PredictPath = ''#S2 images to predict
CLSpath=''#output locations
MetaMaskPath=''#metamasks for filters
ParameterFile='.npy'#numpy array with optimised filter params
Month='M07'#month and year as strings
year='2021'




'''BASIC PARAMETER CHOICES'''
img_size=[224,224] #imagenet img_size[0]s
Ndims=3 #assumption is 3 Ndims
Nclasses=4
batch=32
###############################################################################

'''Get optimised filter parameters'''
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


""" HELPER FUNCTIONS SECTION """
def tic():
    #Homemade version of matlab tic and toc functions

    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():

    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(int(time.time() - startTime_for_tictoc)) + " seconds.")
    else:
        print ("Toc: start time not set")
        
def GDALpix2map(ds, x, y):
    xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
    posX = px_w * x + rot1 * y + xoffset
    posY = rot2 * x + px_h * y + yoffset
    # shift to the center of the pixel
    posX += px_w / 2.0
    posY += px_h / 2.0
    return posX, posY 

# Helper function to crop images to have an integer number of tiles. No padding is used.
def CropToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]

def PadToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    pad_dim0 = size * (1+(Im.shape[0]//size))-Im.shape[0]
    pad_dim1 = size * (1+(Im.shape[1]//size))-Im.shape[1]
    return np.pad(Im, ((0, pad_dim0), (0, pad_dim1), (0,0)), mode='symmetric')
    
# =============================================================================
#Helper functions to move images in and out of tensor format
def split_image(image3, tile_size):
    image_shape = np.shape(image3)
    tile_rows = np.reshape(image3, [image_shape[0], -1, tile_size, image_shape[2]])
    serial_tiles = np.transpose(tile_rows, [1, 0, 2, 3])
    return np.reshape(serial_tiles, [-1, tile_size, tile_size, image_shape[2]])

def unsplit_image(tiles4, image_shape):
    tile_width = np.shape(tiles4)[1]
    serialized_tiles = np.reshape(tiles4, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = np.transpose(serialized_tiles, [1, 0, 2, 3])
    return np.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])

def GPUdilate(array, scale):
    #fast binary dilation on the GPU with tensorflow
    array=array*1 #make sure it's not a bool
    kernel=tf.expand_dims(tf.convert_to_tensor(np.ones((scale,scale,1)), dtype=tf.float32), axis=2)
    ArrayTensor=tf.expand_dims(tf.convert_to_tensor(PadToTile(array, scale), dtype=tf.float32), axis=0) 
    Array2Dconv=tf.nn.convolution(tf.constant(ArrayTensor, dtype=tf.float32), tf.constant(kernel, dtype=tf.float32),padding='SAME').numpy()
    ArrayTensor=None#cleanup GPU memory
    kernel=None
    return np.squeeze(Array2Dconv[:,0:array.shape[0], 0:array.shape[1],:])>0


def TFdilate(Array, scale):
    #fast binary dilation on the GPU with tensorflow. This version returns a tf tensor
    #array=array*1 #make sure it's not a bool
    Array=tf.where(Array, 1, 0)
    Array=tf.cast(Array, dtype=tf.float16)
    kernel=tf.expand_dims(tf.convert_to_tensor(np.ones((scale,scale,1)), dtype=tf.float16), axis=2)
    ArrayTensor=tf.expand_dims(Array, axis=0)
    ArrayTensor=tf.expand_dims(ArrayTensor, axis=-1)
    Array2Dconv=tf.nn.convolution(tf.constant(ArrayTensor, dtype=tf.float16), tf.constant(kernel, dtype=tf.float16),padding='SAME')
    ArrayTensor=None#cleanup GPU memory
    kernel=None
    return tf.squeeze(Array2Dconv)>0
     

def GPUsplit_image(image3, tile_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0], image_shape[2]])

def GPUdownscale(array, scale):
    #pad the array before to make it fit into scale an integer number of times
    kernelvalue=1/(scale*scale)
    kernel=tf.expand_dims(tf.convert_to_tensor(kernelvalue*np.ones((scale,scale,1)), dtype=tf.float32), axis=2)
    ArrayOut=np.zeros((array.shape[0]//scale, array.shape[1]//scale, array.shape[2]))
    for b in range(array.shape[2]):
        arrayin=np.expand_dims(array[:,:,b], axis=2)
        ArrayTensor=tf.expand_dims(tf.convert_to_tensor(arrayin, dtype=tf.float32), axis=0)
        Array2Dconv=tf.nn.convolution(tf.constant(ArrayTensor, dtype=tf.float32), tf.constant(kernel, dtype=tf.float32),padding='SAME', strides=scale).numpy()

        ArrayOut[:,:,b] = np.squeeze(Array2Dconv[:,0:ArrayOut.shape[0], 0:ArrayOut.shape[1],:])
        ArrayTensor=None#cleanup GPU memory
        Array2Dconv=None
    return ArrayOut[0:array.shape[0]//scale, 0:array.shape[1]//scale,:]  
    
    


def FixBigRiverTF(ClassRaster1, bigwater, RiverProb50, LakeProb50, thresh):
    #for large rivers falsely classieifed s lakes
    D=nblogical_and(bigwater, ClassRaster1==1)
    D=tf.convert_to_tensor(D, dtype=tf.bool)
    tfbool=tf.convert_to_tensor(ClassRaster1==2, dtype=tf.bool)
    Pbool=RiverProb50<thresh
    for s in range(100):
        oldsum=tf.math.count_nonzero(D)
        D=TFdilate(D, 3)
        
        D=tf.math.logical_and(D, tfbool)
        #shell=nblogical_and(D, np.logical_not(Lake))
        D=D.numpy()
        D[Pbool]=False
        D=tf.convert_to_tensor(D, dtype=tf.bool)
        newsum=tf.math.count_nonzero(D)
        if newsum==oldsum:
            #print('broke big river filter at '+str(s))
            break
    cond=D.numpy()==1
    ClassRaster1=nbputmask(ClassRaster1, cond, 1)
    LakeProb50=nbputmask(LakeProb50, cond, 0)
    return ClassRaster1, LakeProb50

def FixSmallRiverTF(ClassRaster1, bigwater, RiverProb50, LakeProb50, thresh):
    #for small river slivers and isles in contact with lake water
    D=nblogical_and(bigwater, ClassRaster1==2)
    D=tf.convert_to_tensor(D, dtype=tf.bool)
    tfbool=tf.convert_to_tensor(ClassRaster1==1, dtype=tf.bool)
    Pbool=LakeProb50<thresh
    for s in range(10):
        oldsum=tf.math.count_nonzero(D)
        D=TFdilate(D, 3)
        
        D=tf.math.logical_and(D, tfbool)
        #shell=nblogical_and(D, np.logical_not(Lake))
        D=D.numpy()
        D[Pbool]=False
        D=tf.convert_to_tensor(D, dtype=tf.bool)
        newsum=tf.math.count_nonzero(D)
        if newsum==oldsum:
            #print('broke big river filter at '+str(s))
            break
    cond=D.numpy()==1
    ClassRaster1=nbputmask(ClassRaster1, cond, 2)
    #np.putmask(LakeProb50, cond, 0)
    return ClassRaster1#, LakeProb50

def FixBarsTF(ClassRaster, fuzzybars, thresh):
    #interative dilation filter that captures weak bars near classified bars
    D=ClassRaster==3
    D=tf.convert_to_tensor(D, dtype=tf.bool)
    #choose the class bars can grow into
    area=ClassRaster==0#np.logical_or(ClassRaster==1, ClassRaster==3)
    tfbool=tf.convert_to_tensor(area, dtype=tf.bool)
    Pbool=fuzzybars<thresh
    for s in range(50):
        oldsum=tf.math.count_nonzero(D)
        D=TFdilate(D, 2)
        
        D=tf.math.logical_and(D, tfbool)
        #shell=nblogical_and(D, np.logical_not(Lake))
        D=D.numpy()
        D[Pbool]=False
        D=tf.convert_to_tensor(D, dtype=tf.bool)
        newsum=tf.math.count_nonzero(D)
        if newsum==oldsum:
            print('broke bar filter at iteration '+str(s))
            break
    cond=D.numpy()==1
    np.putmask(ClassRaster, cond, 3)

    return ClassRaster


# 

@nb.jit(parallel=True, nopython=True)  
def nblogical_and(A, B): #faster than numpy on very large arrays. 
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

  
@nb.jit(parallel=True, nopython=True)  
def nbputmask(mask, condition, value): #20% faster than nunpy on very large arrays. 
    maskshape=mask.shape
    mask=mask.ravel()
    condition=condition.ravel()
    n=len(mask)
    for i in nb.prange(n):
        if condition[i]:
            mask[i]=value
    return mask.reshape(maskshape)

@nb.jit(parallel=True, nopython=True)
def is_in_set_pnb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)
     


def decompress(tiles, valid):#decompress to add tiles removed for no data
    fulltiles=np.zeros((len(valid), img_size[0], img_size[0],4), dtype='float32')
    count=0
    for t in range(len(valid)):
        if valid[t]:
            fulltiles[t,:,:,:]=tiles[count, :,:,:]
            count+=1
    return fulltiles

def SetBigLakes(ClassRaster10m, ClassRaster50m):
    LakeLabel=sclabel(ClassRaster50m==2, output='int32')[0]
    num, counts=np.unique(LakeLabel, return_counts=True)
    
    #big=np.where(counts>=224*224)
    for o in range(1, len(num)):
        if counts[num[o]]>=224*224*0.33:
            ClassRaster10m=nbputmask(ClassRaster10m, LakeLabel==o, 2)
    return ClassRaster10m

def GPURiver2Ocean(OWater, ClassRaster):
    FatOcean=GPUdilate(OWater, 50)#going to set any river pixels within 500m of the ocean to ocean
    RiversinOcean=nblogical_and(FatOcean, ClassRaster==1)
    OWater=nbputmask(OWater, RiversinOcean, True)
    
    return OWater

    

 
'''prime numba functions'''

A=np.random.randint(low=0, high=10, size=(1000,1000))==5
B=np.random.randint(low=0, high=10, size=(1000,1000))==5
C=np.int32(np.random.randint(low=0, high=10, size=(1000,1000)))
static=np.int32([5,1])

D=nblogical_and(A,B)

E=nbputmask(A,B, 2)

F=is_in_set_pnb(C, static)


del A, B, C, static, D, E, F
   
""" CLASSIFY THE GEE IMAGES  """ 
# Getting Names from the files and list unclassified tiles

Imagelist=list(set(glob.glob(PredictPath+'GEES2_'+year+Month+'**/S2*.tif', recursive=True))-set(glob.glob(PredictPath+'GEE_'+Month+'**/MODIS_*.tif', recursive=True))-set(glob.glob(PredictPath+'GEE_'+Month+'**/NoData_*.tif', recursive=True)))
Existinglist=list(glob.glob(CLSpath+'**/*.tif'))
#Existinglist=[]
for i in range(len(Imagelist)):
    for j in range(len(Existinglist)):
        if os.path.basename(Imagelist[i]) in os.path.basename(Existinglist[j]):
            Imagelist[i]='blank'
            
while 'blank' in Imagelist: Imagelist.remove('blank')
Imagelist.sort()

print('Found '+str(len(Imagelist))+' GEE tiles to process')

""" Load the FCN"""
model=tiramisu(tile_size=img_size[0],bands=Ndims,Nclasses=Nclasses)
Optim = optimizers.RMSprop(learning_rate=0.001)
#compile the model with cross entropy loss.  Has no role in inference and avoids the need for TF 2.9
model.compile(optimizer=Optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights(modelpath)

#start the classification
for i,im in enumerate(Imagelist):
    tic()
    print('Loading '+os.path.basename(im))
    #Im3D = io.imread(im)#convert to float to be sure the ndwi works
    Im3D=np.float32(io.imread(im))
    #ndwi=NDWI(Im3D[:,:,0], Im3D[:,:,2])
    nodata=nblogical_and(nblogical_and(Im3D[:,:,0]==0, Im3D[:,:,1]==0), nblogical_and(Im3D[:,:,2]==0, Im3D[:,:,3]==0)) 
    clouds=nblogical_and(nblogical_and(Im3D[:,:,0]!=0, Im3D[:,:,1]!=0), nblogical_and(Im3D[:,:,2]!=0, Im3D[:,:,3]==0))  
    MetaMaskFile=MetaMaskPath+'Meta'+os.path.basename(im)[6:11]+'.tif'

    

    print('Starting FCN inference for '+os.path.basename(im))
    '''inference'''
    Im3D=Im3D[:,:,0:3]
    Im3Dshape=Im3D.shape
    ImCrop = PadToTile (Im3D, img_size[0])
    del Im3D
    I_tiles_main = split_image(ImCrop, img_size[0])
    #try to compress and remove tiles with no data or ocean
    #no data
    nodat_tiles_main = np.squeeze(split_image(PadToTile(np.expand_dims(nodata, axis=-1), img_size[0]), img_size[0]))
    val1=np.sum(1*nodat_tiles_main, axis=-1)
    val2=np.sum(1*val1, axis=-1)
    valid1=val2<224*224
    del nodat_tiles_main
    del val1
    del val2
    #ocean, start by extracting the ocean from the metamask
    ds=gdal.Open(im)
    pixelSizeX = ds.RasterXSize
    pixelSizeY = ds.RasterYSize
    ULmapx,ULmapy= GDALpix2map(ds, 0, 0)
    LRmapx,LRmapy= GDALpix2map(ds, pixelSizeX, pixelSizeY)
    ds=None

    ds = gdal.Open(MetaMaskFile)
    ds = gdal.Translate('', ds, projWin = [ULmapx, ULmapy, LRmapx, LRmapy], format='MEM')
    Ocean=ds.ReadAsArray()==4
    ds = None
    
    ocean_tiles_main = np.squeeze(split_image(PadToTile(np.expand_dims(Ocean, axis=-1), img_size[0]), img_size[0]))
    if np.count_nonzero(ocean_tiles_main)>224*224:
        val1=np.sum(1*ocean_tiles_main, axis=-1)
        val2=np.sum(1*val1, axis=-1)
        valid2=val2<224*224

        del ocean_tiles_main
        del val1
        del val2
        valid=nblogical_and(valid1, valid2)
        del valid1
        del valid2
    else:
        valid=valid1
        del valid1
    if np.count_nonzero(valid)<batch:
        #make sure there is at least 1 batch after commpression to avoid bugs in the inference loop
        for s in range(batch):
            valid[s]=True
    #valid=np.logical_not(valid)   
    print('starting with '+str(I_tiles_main.shape[0])+' tiles and compressing ocean and no-data tiles')
    I_tiles_main=np.compress(valid, I_tiles_main, axis=0)
    print('using '+str(I_tiles_main.shape[0])+' tiles after compression')
    #carry on processing
    I_tiles_main=(I_tiles_main/127.5)-1
    MainPredictedTensor =tf.convert_to_tensor(I_tiles_main, dtype=tf.float32)
    MainPaddedShape=ImCrop.shape
    MainTilesNum=tf.shape(MainPredictedTensor)[0]
    ImCrop=None
    I_tiles_main=None
    
    


    MainPredictedTiles = np.empty((MainTilesNum,img_size[0],img_size[0],Nclasses), dtype=np.float32)     
    
    BATCH_INDICES = np.arange(start=0, stop=MainTilesNum, step=batch)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, MainTilesNum)  # add final batch_end row
    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        MainPredictedTiles[batch_start:batch_end] =model(MainPredictedTensor[batch_start:batch_end],training=False).numpy()
    MainPredictedTensor=None
    
    #do the 5x downsampled version
    Im3D=np.float32(io.imread(im))
    Im5X=PadToTile (Im3D[:,:,0:Ndims], 5)
    del Im3D
    #ImRivers = np.uint8(downscale_local_mean(Im2X, (2,2,1)))
    ImRivers=GPUdownscale(Im5X, 5)
    Im5X=None
    ImCrop = PadToTile (ImRivers[:,:,0:Ndims], img_size[0])
    #I_tiles_half=split_image(ImCrop, size)
    #HalfPredictedTensor =tf.convert_to_tensor(I_tiles_half, dtype=tf.float16)
    HalfPredictedTensor = GPUsplit_image(ImCrop, [img_size[0],img_size[0]])
    HalfPredictedTensor = tf.cast(HalfPredictedTensor, dtype=tf.float32)
    HalfPredictedTensor = tf.math.divide(HalfPredictedTensor, 127.5)
    HalfPredictedTensor = tf.math.subtract(HalfPredictedTensor, 1.0)
    HalfCropShape=ImCrop.shape
    HalfTilesNum=tf.shape(HalfPredictedTensor)[0]
    ImCrop=None
    ImRivers=None
    I_tiles_half=None
    #downscaled tles
    HalfPredictedTiles = np.empty((HalfTilesNum,img_size[0],img_size[0],Nclasses), dtype=np.float32)     
    #data = tf.data.Dataset.from_tensor_slices(MainPredictedTensor).batch(8)
    BATCH_INDICES = np.arange(start=0, stop=HalfTilesNum, step=batch)  # row indices of batches
    BATCH_INDICES = np.append(BATCH_INDICES, HalfTilesNum)  # add final batch_end row
    for index in np.arange(len(BATCH_INDICES) - 1):
        batch_start = BATCH_INDICES[index]  # first row of the batch
        batch_end = BATCH_INDICES[index + 1]  # last row of the batch
        HalfPredictedTiles[batch_start:batch_end] =model(HalfPredictedTensor[batch_start:batch_end],training=False).numpy()
    MainTilesNum=None
    HalfTilesNum=None
    HalfPredictedTensor=None

    #cleanup the gpu for cupy

    tf.keras.backend.clear_session()
    gc.collect()

    print('FCN inference done, untiling')
    '''untile the tensors as predicted classes'''
    if not valid.all():
        MainPredictedTiles=decompress(MainPredictedTiles, valid)
    
    MainPredictions=np.argmax(MainPredictedTiles, axis=3)
    MainClasses=np.uint8(unsplit_image(MainPredictions, [MainPaddedShape[0], MainPaddedShape[1],1]))
    Rtiles=MainPredictedTiles[:,:,:,3]
    Gtiles=MainPredictedTiles[:,:,:,2]
    Btiles=MainPredictedTiles[:,:,:,1]
    Rimage=np.uint8(100*unsplit_image(Rtiles, [MainPaddedShape[0], MainPaddedShape[1],1]))
    Gimage=np.uint8(200*unsplit_image(Gtiles, [MainPaddedShape[0], MainPaddedShape[1],1]))
    Bimage=np.uint8(200*unsplit_image(Btiles, [MainPaddedShape[0], MainPaddedShape[1],1]))
    ProbImage10=np.zeros((MainPaddedShape[0], MainPaddedShape[1],3), dtype='uint8')
    BarProb10=np.squeeze(Rimage)
    ProbImage10[:,:,0]=np.squeeze(Gimage)
    ProbImage10[:,:,1]=np.squeeze(Bimage)
    ProbImage10[:,:,2]=np.squeeze(Rimage)
       
    MainPredictions=None
    MainPredictedTiles=None
    
    
    HalfPredictions=np.argmax(HalfPredictedTiles, axis=3)
    HalfClasses=np.uint8(unsplit_image(HalfPredictions, [HalfCropShape[0], HalfCropShape[1],1]))
    Rtiles=HalfPredictedTiles[:,:,:,3]
    Gtiles=HalfPredictedTiles[:,:,:,2]
    Btiles=HalfPredictedTiles[:,:,:,1]
    
    
    Rimage=np.uint8(200*unsplit_image(Rtiles, [HalfCropShape[0], HalfCropShape[1],1]))
    Gimage=np.uint8(200*unsplit_image(Gtiles, [HalfCropShape[0], HalfCropShape[1],1]))
    Bimage=np.uint8(200*unsplit_image(Btiles, [HalfCropShape[0], HalfCropShape[1],1]))
    RimageBig=Rimage.repeat(5, axis=0).repeat(5, axis=1)
    GimageBig=Gimage.repeat(5, axis=0).repeat(5, axis=1)
    BimageBig=Bimage.repeat(5, axis=0).repeat(5, axis=1)
    ProbImage50=np.zeros((5*HalfCropShape[0], 5*HalfCropShape[1],3), dtype='uint8')
    ProbImage50[:,:,0]=np.squeeze(RimageBig)
    ProbImage50[:,:,1]=np.squeeze(GimageBig)
    ProbImage50[:,:,2]=np.squeeze(BimageBig)
    RiverProb50=np.squeeze(BimageBig)/2
    LakeProb50=np.squeeze(GimageBig)/2
    HalfPredictedTiles=None
    HalfPredictions=None
    HalfClassBig = HalfClasses.repeat(5, axis=0).repeat(5, axis=1)
    #HalfNotClassBig = HalfNotClasses.repeat(5, axis=0).repeat(5, axis=1)
    del Rtiles
    del Gtiles
    del Btiles
    del Rimage
    del Gimage
    del Bimage
    del RimageBig
    del GimageBig
    del BimageBig
    
    ClassRaster1=MainClasses[0:Im3Dshape[0], 0:Im3Dshape[1]].reshape((Im3Dshape[0], Im3Dshape[1]))
    ClassRaster2=HalfClassBig[0:Im3Dshape[0], 0:Im3Dshape[1]].reshape((Im3Dshape[0], Im3Dshape[1]))
    ClassRaster3=MainClasses[0:Im3Dshape[0], 0:Im3Dshape[1]].reshape((Im3Dshape[0], Im3Dshape[1]))
    BarProb10=BarProb10[0:Im3Dshape[0], 0:Im3Dshape[1]]
    ProbImage10=ProbImage10[0:Im3Dshape[0], 0:Im3Dshape[1],:]
    RiverProb50=np.uint8(RiverProb50[0:Im3Dshape[0], 0:Im3Dshape[1]])
    LakeProb50=np.uint8(LakeProb50[0:Im3Dshape[0], 0:Im3Dshape[1]])
    ProbImage50=ProbImage50[0:Im3Dshape[0], 0:Im3Dshape[1], :]
    MainClasses=None
    RiversClassBig=None
    HalfClassBig=None
    #cut the edges of ProbImage50 according to edges of the 10m Probs
    WeakWater=np.logical_or(ProbImage10[:,:,0]>=40, ProbImage10[:,:,1]>=40)
    
    for b in range(1,3):
        ProbImage50[:,:,b]=WeakWater*ProbImage50[:,:,b]
    del WeakWater
    '''Combine multiscale inputs'''
    ClassRaster1=SetBigLakes(ClassRaster1, ClassRaster2)
    #keep the 50m bars but don't let them cut into water
    bar50=nblogical_and(ClassRaster2==3, np.logical_not(np.logical_or(ClassRaster1==1, ClassRaster1==2)))
    ClassRaster1=nbputmask(ClassRaster1, bar50,3)
    #make the fuzzy bar raster
    if np.count_nonzero(ClassRaster1==3)>0:
        corridor=np.logical_not(GPUdilate(ClassRaster1==1, p11))
        badbars=nblogical_and(ClassRaster1==3, corridor)
        ClassRaster1=nbputmask(ClassRaster1, badbars, 0)

    
    '''filters'''
    print('GPU filtering ...')
    bigwater=np.logical_or(ClassRaster1==1, ClassRaster1==2)
    bigwater=binary_erosion(bigwater, iterations=10)
    ClassRaster1, LakeProb50 = FixBigRiverTF(ClassRaster1, bigwater, RiverProb50, LakeProb50, p12)#
    ClassRaster1 = FixSmallRiverTF(ClassRaster1, bigwater, RiverProb50, LakeProb50, p13)#
    ClassRaster1 = FixBarsTF(ClassRaster1, ProbImage10[:,:,2], p14)
    del bigwater
    if np.count_nonzero(Ocean)>0:
        Ocean=GPURiver2Ocean(Ocean, ClassRaster1)

    

     
    '''Make nodata class 8, clouds class 7, classes 4 to 6 will be filled later'''

    ClassRaster1=nbputmask(ClassRaster1, nodata, 8)
    ClassRaster1=nbputmask(ClassRaster1, clouds, 7)
    ClassRaster1=nbputmask(ClassRaster1, Ocean, 4)
    
    '''final image with the class and the fuzzy bars'''
    FinalRaster=np.zeros((ClassRaster1.shape[0], ClassRaster1.shape[1], 5), dtype='uint8')
    FinalRaster[:,:,0]=ClassRaster1
    FinalRaster[:,:,1]=ProbImage50[:,:,0]
    FinalRaster[:,:,2]=ProbImage50[:,:,1]
    FinalRaster[:,:,3]=ProbImage50[:,:,2]
    FinalRaster[:,:,4]=ProbImage10[:,:,1]#export 10m river probs




            
        
# # =============================================================================
        
#         """ SAVE AND OUTPUT  """


    Zone=os.path.basename(im)[8:11]
    CLSzonepath=CLSpath+Month+'_Z'+Zone+'/'
    try:
        os.mkdir(CLSzonepath)
    except:
        print('GZD folder exists')
        
    print('Saving geotif to disk')
    ImageRoot=os.path.basename(im)[:-4]
    SavedClass=CLSzonepath+'Class_'+ImageRoot+'.tif'
    ImageSize=FinalRaster.shape
    ImageFile = gdal.Open(im)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(SavedClass, ImageSize[1], ImageSize[0], 5, gdal.GDT_Byte, options=['NUM_THREADS=10','COMPRESS=DEFLATE','PREDICTOR=2','ZLEVEL=9'])
    outdata.SetGeoTransform(ImageFile.GetGeoTransform())
    outdata.SetProjection(ImageFile.GetProjection())
    outdata.GetRasterBand(1).WriteArray(FinalRaster[:,:,0])
    outdata.GetRasterBand(2).WriteArray(FinalRaster[:,:,1])
    outdata.GetRasterBand(3).WriteArray(FinalRaster[:,:,2])
    outdata.GetRasterBand(4).WriteArray(FinalRaster[:,:,3])
    outdata.GetRasterBand(5).WriteArray(FinalRaster[:,:,4])
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    ImageFile = None
    ct = None



    print('Image classified')
    del FinalRaster

    toc()
    print(' ')
  
            

                
    


