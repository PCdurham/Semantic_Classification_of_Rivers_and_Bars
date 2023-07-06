# Semantic Classification of Rivers and Sediment
  


## A global semantic classification method for rivers and sediment deposits based on deep learning and image processing of Sentinel 2 data 



## Local Dependencies
- Tested with python 3.9
- Tensorflow v2.9 with CUDA support
- osgeo with gdal, gdal_array and osr modules
- numba
- scikit-learn
- scikit-image
- fill_voids
- basics: PIL, pandas, numpy, scipy, os, shutil, glob, etc..

## Hardware requirements
- CUDA compatible GPU with minimum 8Gb memory
- 128 Gb system RAM (for the post-processing phase)
- Local storage.  Global coverage requires ~5 Tb which must also be available on Google Drive. 

## Data download with Google Earth Engine

The first step is to download Sentinel-2 imagery.  We use the Python API for Google Earth Engine (GEE) via Google Colaboratory. The data arrives in 4 bands: Band 8 (NIR), Band 4 (R) and Band 3 (G) have the main image data.  We add an additional band with a cloud mask as determined by the S2cloudless database from GEE.  We preserve the full 10 m spatial resolution but the radiometry is downsampled to a a range of 0-255 (8-bit) in order to reduce volume and facilitate global scale usage (on the order of 6Tb for global coverage).  Spatially, the data are organised as per the Military Grid Reference System (MGRS) which further divides each UTM zone of 6 degrees in longitude into squares of 8 (10 at the poles) degrees of latitude. Each image is converted to a UTM projected coordinate system which will preserve the size of each pixel (1 pixel is 10 meters wide at all latitudes). Image files are written to Google Drive in a seperate folder for each MGRS grid zone, GEE limits the size of any given image to ~2Gb.  Images for an entire MGRS range from 3-10 GB and therefore the GEE outputs is split into 3-12 image sub-tiles. 



The full Jupyter notebook is include in the code folder. This notebook uses a csv file to read map bounds for the desired MGRS zones. The resulting images are saved to Google Drive and must then be synced to a local directory.  

## Semantic class selection
The method is designed with 4 modelled classes and 4 inherited classes.  The modelled classes are: background (0), rivers (1), lakes (2) and exposed sediment bars (3).  These are the 4 classes used in the FCN.  In addition, we use 4 classes that are derived from metadata: oceans (4), glaciated terrain (5), snow (6), cloud (7) and data gaps (8).  

## External data pre-processing
Inherited classes 4 (ocean) and 5 (glaciated terrain) rely on external datasets.  Furthermore, many of the filters described below leverage existing information on lakes, rivers and roads. This information is extracted from existing global databases:
+ Rivers with strahler orders > 6 are derived from the [HydroRIVERS](https://www.hydrosheds.org/products/hydrorivers) database.
+ Lakes with a surface in excess of 5km<sup>2</sup> are derived from the [HydroLAKES](https://www.hydrosheds.org/products/hydrolakes) database
+ Continel margins are derived from the [HydroSHEDS](https://www.hydrosheds.org/) database.
+ The location of glaciers is taken from the [Randolf Glacier Inventory](https://www.glims.org/RGI/).


Prior to all processing, these metadata sets are processed to create rasters that will encode their information for each MGRS. Each of these metadata rasters has the same XY dimensions and pixel size as the output classification rasters thus allowing for rapid and efficient filter operations.  We first used QGIS in order to select and extract rivers with a strahler order greater than 6 from the HydroRIVERS data.  We then applied a buffer of 0.01 degrees (~1km) in order to get an initial estimate of the location of large rivers.  This is then burned into the matadata raster for each given MGRS as a value of 1.  We then burn the lake vectors from the HydroLAKES data.  Lakes from 5km<sup>2</sup> to 125km<sup>2</sup> are burned as a value of 2 and lakes in excess of 125km<sup>2</sup> are burned as a value of 3. Oceans, derived from the HydroSHEDS database are burned as a value of 4.  We then open the Randolf Glacier Inventory and apply a buffer of 0.05 degrees (5 km) and define this area as 'Glaciated terrain'.  This is burned into the metadata rasters as a value of 5. The burn-in values for the ocean (4) and glaciated terrain (5) will directly translate into classes 4 and 5 during processing.  The other values will be used to inform filter algorithms.
  
  

## FCN model architecture and training
We use the Tiramisu fully convolutional densenet architecture of [Jegou et al (2017)](https://arxiv.org/abs/1611.09326)adapted to use Tensorflow v2.5 or greater. We provide our training script (Train_FCN_datagen.py).  Once the model is created, we use the sigmoid-focal loss function ([Lin et al (2018)](https://arxiv.org/abs/1708.02002)) to mitigate against class imbalance which we expect to encounter given that rivers occupy ~0.5% of the global surface.  We use a customisable learning rate scheduler. We also use a model checkpoint callback in order to save model weights at each training epoch. The training script uses a data generator that reads images as jpg files with corresponding label masks stored as png files in the same folder.  
## FCN inference
Inference is delivered by the GEEinference.py script.  This takes the raw sub-tile images produced by GEE and runs the FCN on tiles of 224x224.  Tiles that are 100% ocean or that do not have data, usually due to cloud cover, are skipped for efficiency. After the first pass inference is complete, we downsample the initial imagery by a factor of 5 and run the inference again in order to detect water features larger than 1 tile (ie 2.24 km) and up to a maximum characteristic length of ~11 km.

Outputs for each GEE image are an initial semantic classification raster and a copy of raw softmax probabilities for the classes of rivers, lakes and bars for the native 10m resolution inference along with the softmax probabilities.  The gdal library is used to write the output 5 channel image with the same XY geocoding as the initial GEE image.

## GPU filters
We find that post-processing significantly improves the accuracy of predictions, in particular for the river class.  We apply a number of filters that combine spatial criteria via traditional binary morphology operations, the raw softmax probabilities and external datasets on the global distribution of lakes and rivers.

We broadly divide these filters in so-called 'GPU' and 'CPU' filters. GPU filters are applied by the main inference script and are accelerated by using Tensorflow to compute binary dilations as thresholded convolutions on the GPU.  We have 3 GPU filters: FixBigRiverTF; FixRiverSliverTF and FixBarsTF.

**FixBigRiverTF** is defined on line 204 of the inference script.  It's function is to disambiguate large rivers falsely classified as lakes.  The fundamental assumption is that if a lake pixel is next to a river pixel, it is potentially misclassified and should also be a river pixel.  The raw softmax probabilities saved after the inference script provide the criteria.  It starts by approximating that 'large' water bodies, either rivers or lakes, are those with a core of pixels remaining after 10 iterations of binary erosions.  After isolating which of these water bodies are rivers, the algorithm uses progressive binary dilations (on the GPU) to grow the core large river pixels.  After each iteration, the algorithm checks the softmax probabilities of new *lake* pixels after dilation. If these new pixels have a softmax probability in excess of a threshold, they are re-classified as river.  

 **FixRiverSliverTF** is defined on line 228 of the inference script.  It's function is to disambiguate small slivers of river in contact with large lakes.  Similarly to **FixBigRiverTF**, the fundamental assumption is that if a river pixel is in contact with a lake pixel, it may be misclassified.  Again it is the raw softmax output that will provide the decision criteria.  This algorithm starts with large lakes and then uses progressive dilations on the GPU.  After each dilation, the algorithm checks each new *river* pixel.  If the softmax probabilities are above a threshold, new river pixels are re-classified as lakes. 
 
 **FixBarTF** is defined on line 252 of the inference script.  It's function is to disambiguate background areas adjacent to bars that might in fact be bars.  Similar to the other 2 filters above, the assumption is that if a background pixel is adjacent to a bar pixel, it might be misclassified.  The softmax probabilities will again give the decision criteria.  This algorithm starts with the initially classified bar areas and again uses binary dilation on the GPU, after each dilation, the algorithm checks background pixels against a threshold and conditionally reclassifies them as bars. 

## Post-processing and CPU filters
Once inference and GPU filters have been executed for a given MGRS, the final post-processing script can be executed.  If system RAM is > 128 Gb, this can be run simultaneously to the inference script.
The post-processing script begins by merging the multiple sub-images into a single large image mosaic of 5 channels (the initial class raster and 4 channels of softmax probabilities). The script then applies CPU filters. These CPU filters use the numba library to execute basic operations such as set-based masking and pixel-wise boolean operations on parallel CPU threads. 

**LakeSHEDS2** is defined on line 304 of the post-procesing script (PostProcess.py).  It's primary purpose is to correct river class pixels that are within a known large lake. This can happen when water bodies are larger that the 224x224 image tile size used by the model. For lakes larger than 5km<sup>2</sup> but smaller than 125<sup>2</sup>, the algorithm examines the softmax probability outputs for the lake areas, extended by a binary dilation of 5 iterations. Any pixel in this area with softmax probabilities above a threshold get re-classed as lakes.  In the cases of very large lakes (>125km<sup>2</sup>), the filter dilates the lake edges by 50 pixels, fills the voids in the lakes, and then reclassifies any river pixel within this expanded area to lake. 

**RiverSHEDS** is defined on line 206 of the post-processing script.  It's primary purpose is to correct lake class pixels that are in a known large river.  The difficulty in this case is that the metadata from the HydroRIVERS database does not give a precise outline for rivers and comes in the form of a vector line.  Our solution is to buffer this river line and then analyse the connected components of lake pixels that pass a threshold for softmax probability and that touch this buffer.    

**FixSmallWaterBody** is defined on line 341 of the post-processing script. The purpose of this filter is to correct small compact lakes that have mistakenly been classified as rivers.  We use binary labelling of connected components and region properties.  For each small river object of less than 10000 pixels, we combine the ratio of the major axis to the minor axis with the solidity property.  If these both pass a threshold value, the compact river object is classed as a lake. 
 
**Rivers_in_Lakes** is define on line 401 of the post-processing script.  This script uses a void fill algorithm to reclass any river object completely enclosed in a lake as lake.

Before applying the final filters, all main road areas in the metadata raster are set to background (class 0) on line 524 of the post processing script.

**Lake2Ocean** is defined on line 416 of the post-processing script. It's purpose is to correct lake slivers in contact with the ocean caused by the model often classifying coastal waters as lakes. This filter labels connected lake objects and re-classifies any lake object that intersects with a binary dilation of the ocean as ocean.

**River2Ocean** is defined on line 442 of the post processing script.  Similar to the Lake2OCean filter, this filter seeks to eliminate river slivers that are in contact with the ocean.  Given that connected river objects extend inland, this filter uses a hard ocean buffer of 150 meters.  Within this buffer, any river pixel is reclassified to ocean. 

**CleanBars** is defined on line 378 of the post-processing script. This filter forces the criteria than any sediment bar object (class 3) must be adjacent to a river pixel (class 1).   

## Filter Optimisation
The CPU and GPU filters have a total of 13 threshold parameters with ranges of 0-100  that need to be optimised.  We used a Monte-Carlo type approach.  Rather that a brute force approach across the full 13 dimension parameter space, we started by manually estimating a parameter set that gave reasonable performance.  Then we randomly created 10 000 parameter combinations within a range of +/- 10% for each parameter.  The resulting final parameters are stored in a numpy array that can be read by the filters.

## Final Outputs
The final class raster for each MGRS zone will be saved as Class_S2_Month_Zone.tif.  The data are highly compressed and are typically 20 to 50 Mb for each MGRS gridzone comprising in excess of 4 Gigapixels.  Global coverage can be achieved in less than 10 Gb.  For each class raster, a color table is saved with the background set to transparent.  Below is an example from MGRS zone 53V with a Google Satellite base image.    

![Screenshot](Example.png)

## Data Examples. 
We provide a sample of 33 full MGRS zones as examples (samples folder).  These are from the Americas, Africa, Europe, East Asia and North-West Russia.  Once downloaded, the tiles can be directly read in any GIS software. Classes are: 0: background (transparent) 1: Rivers (light blue); 2: Lakes (green); 3: Sediment Bars (red); 4: Oceans (royal blue); 5: Glaciated Terrain (cyan); 6: Snow (light grey); 7: Clouds (dark grey); 8: Data Gaps (black)