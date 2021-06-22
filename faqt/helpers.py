#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:41 2019

@author: aimachine
"""

from __future__ import print_function, unicode_literals, absolute_import, division
#import matplotlib.pyplot as plt
import numpy as np
import collections

import warnings
from skimage.filters import gaussian
from six.moves import reduce
from matplotlib import cm
from skimage.filters import threshold_local, threshold_otsu
from skimage.morphology import remove_small_objects, thin
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_fill_holes

from skimage.segmentation import watershed
import os
import difflib
import pandas as pd
import glob
from tifffile import imread, imwrite
from scipy import ndimage as ndi
from pathlib import Path
from tqdm import tqdm
from skimage.segmentation import  relabel_sequential
from skimage import morphology
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import  binary_dilation, binary_erosion
from skimage.util import invert 
from skimage import measure
from skimage.filters import sobel
from skimage.measure import label
from scipy import spatial

from csbdeep.data import  create_patches,create_patches_reduced_target, RawData
from skimage import transform

def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled
def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_dilation(mask,iterations = iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")



    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out


def UNETPrediction(filesRaw, model,Savedir, min_size, n_tiles, axis,show_after = 1):

    count = 0
    for fname in filesRaw:
            count = count + 1
            print('Applying UNET prediction')
            Name = os.path.basename(os.path.splitext(fname)[0])
            image = imread(fname)
            
            Segmented = model.predict(image, axis, n_tiles = n_tiles)
            thresh = threshold_otsu(Segmented)
            Binary = Segmented > thresh
            
            #Postprocessing steps
            Filled = binary_fill_holes(Binary)
            Finalimage = label(Filled)
            Finalimage = fill_label_holes(Finalimage)
           
                    
            Finalimage = relabel_sequential(Finalimage)[0]
            
            if count%show_after == 0:
                    doubleplot(image, Finalimage, "Original", "Segmentation")
            imwrite(Savedir + Name + '.tif', Finalimage.astype('uint16'))

    return Finalimage

 
def WingArea(LeftImage, RightImage):

   Leftcount =  np.sum(LeftImage > 0)
   Rightcount = np.sum(RightImage > 0)
   
   RightMinusLeft = Rightcount - Leftcount
   RightPlusLeft = Rightcount + Leftcount
   Assymetery = 2 * RightMinusLeft / RightPlusLeft
   
   return Rightcount, Leftcount, RightMinusLeft, RightPlusLeft, Assymetery



def AsymmetryComputer(MaskResults,AsymmetryResults,AsymmetryResultsName, extra_title = "" , computeAsymmetry = True):
    
    
    
            Raw_pathRight = os.path.join(MaskResults, '*tif')
            Raw_pathLeft = os.path.join(MaskResults, '*tif')
            
            filesRawRight = glob.glob(Raw_pathRight)
            filesRawLeft = glob.glob(Raw_pathLeft)
            filesRawRight.sort()
            filesRawLeft.sort()
            
            AllRightArea = []
            AllLeftArea = []
            AllRightMinusLeftArea = []
            AllRightPlusLeftArea = []
            AllAssymetery = []
            AllName = []
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 150)
            if computeAsymmetry:
                        for fnameRight in filesRawRight:
                            
                           NameRight = os.path.basename(os.path.splitext(fnameRight)[0]) 
                           imageRight = imread(fnameRight)
                           for fnameLeft in filesRawLeft:
                               NameLeft = os.path.basename(os.path.splitext(fnameLeft)[0]) 
                               imageLeft = imread(fnameLeft) 
                               
                               ChangeName = difflib.ndiff(NameLeft, NameRight)
                               delta = ''.join(x[0:] for x in ChangeName if x.startswith('- '))
                               
                               #ChangeName = NameRight.replace(RightName, LeftName) 
                               if delta == '- L':
                                   print(NameLeft, NameRight)
                                   RightArea, LeftArea, RightMinusLeft, RightPlusLeft, Assymetery = WingArea(imageLeft, imageRight)
                                 
                                   AllName.append(NameLeft)
                                   AllRightArea.append(RightArea)
                                   AllLeftArea.append(LeftArea)
                                   AllRightMinusLeftArea.append(RightMinusLeft)
                                   AllRightPlusLeftArea.append(RightPlusLeft)
                                   AllAssymetery.append(Assymetery)
                                    
                        df = pd.DataFrame(list(zip(AllRightArea,AllLeftArea,AllRightMinusLeftArea,AllRightPlusLeftArea,AllAssymetery)), index =AllName, 
                                                                      columns =['RightArea', 'LeftArea', 'Right-Left', 'Right+Left', 'Assymmetery'])
                        
                        df.to_csv(AsymmetryResults + '/' + AsymmetryResultsName + extra_title +  '.csv')  
                        df
                    
                        positivecount = np.sum(df['Assymmetery']>0)
                        negativecount = np.sum(df['Assymmetery']<0)   
                        print('Positive Count' , positivecount)
                        print('Negative Count' , negativecount)
                        plt.plot(AllAssymetery)
                        plt.title("Asymmetry" + extra_title)
                        plt.ylabel("Asymmetry")
                        plt.xlabel("Filenumber")
                        plt.show()

            else:
                
                
                  for fnameRight in filesRawRight:
                      
                         NameRight = os.path.basename(os.path.splitext(fnameRight)[0]) 
                         imageRight = imread(fnameRight)
                         Area, _, _, _, _ = WingArea(imageRight, imageRight) 
                         AllName.append(NameRight)
                         AllRightArea.append(Area)
                   
                  df = pd.DataFrame(list(zip(AllRightArea)), index =AllName, 
                                                                      columns =['Area'])
                
                  df.to_csv(AsymmetryResults + '/' + AsymmetryResultsName + extra_title +  '.csv')  
                  df
            
                  plt.plot(AllAssymetery)
                  plt.title("Area" + extra_title)
                  plt.ylabel("Area")
                  plt.xlabel("Filenumber")
                  plt.show()
    
def generate_2D_patch_training_data(BaseDirectory, SaveNpzDirectory, SaveName, patch_size = (512,512), n_patches_per_image = 64, transforms = None):

    
    raw_data = RawData.from_folder (
    basepath    = BaseDirectory,
    source_dirs = ['Original'],
    target_dir  = 'BinaryMask',
    axes        = 'YX',
    )
    
    X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = patch_size,
    n_patches_per_image = n_patches_per_image,
    transforms = transforms,
    save_file           = SaveNpzDirectory + SaveName,
    )

def generate_2D_patch_training_dataRGB(BaseDirectory, SaveNpzDirectory, SaveName, patch_size = (512,512), n_patches_per_image = 64, transforms = None):

    
    raw_data = RawData.from_folder (
    basepath    = BaseDirectory,
    source_dirs = ['Original'],
    target_dir  = 'BinaryMask',
    axes        = 'YXC',
    )
    
    X, Y, XY_axes = create_patches_reduced_target (
    raw_data            = raw_data,
    patch_size          = (patch_size[0],patch_size[1], None),
    n_patches_per_image = n_patches_per_image,
    transforms = transforms,
    target_axes         = 'YX',
    reduction_axes      = 'C',
    save_file           = SaveNpzDirectory + SaveName,
    )    
     
     
    
    
def OrientationArea(filesRaw, UnetModel, Savedir, show_after = 1, min_size = 20, flip = True, UnetCompartmentModel = None,UnetVeinModel = None, computeAsymmetry = True):
    
        count = 0
        axes = 'YXC'
        Path(Savedir).mkdir(exist_ok = True)
        MaskResults = Savedir + '/MaskResults/'
        if UnetCompartmentModel is not None:
            MaskCompartmentResults = Savedir + '/MaskCompartmentResults/'
            MaskCompartmentLabelResults = Savedir + '/MaskLabelCompartmentResults/'
            AsymmetryCompartmentResults = Savedir + '/AsymmetryCompartmentResults/'
            AsymmetryCompartmentResultsName = 'AsymmetryCompartment'
            
            Path(MaskCompartmentResults).mkdir(exist_ok = True)
            Path(MaskCompartmentLabelResults).mkdir(exist_ok = True)
            Path(AsymmetryCompartmentResults).mkdir(exist_ok = True)
        
        
        if UnetVeinModel is not None:
            MaskVeinResults = Savedir + '/VeinResults/'
            Path(MaskVeinResults).mkdir(exist_ok = True)
            
        AsymmetryResults = Savedir + '/AsymmetryResults/'
        AsymmetryResultsName = 'Asymmetry'
        Path(MaskResults).mkdir(exist_ok = True)
        Path(AsymmetryResults).mkdir(exist_ok = True)
        for fname in filesRaw:
          
                    #Read image  
                    Name = os.path.basename(os.path.splitext(fname)[0])
            
                    image = imread(fname)


                    image = image[:,:,0:3]
                    #DO the segmentation
                    Segmented = UnetModel.predict(image,axes)
                    thresh = threshold_otsu(Segmented) 
                    Binary = Segmented > thresh
                    Filled = binary_fill_holes(Binary)
                    Finalimage = remove_small_objects(Filled, min_size)
                    Finalimage = Finalimage[:,:,0]
                    y, x = np.nonzero(Finalimage)
                    x = x - np.mean(x)
                    y = y - np.mean(y)
                    coords = np.vstack([x, y])

                    cov = np.cov(coords)
                    evals, evecs = np.linalg.eig(cov) 


                    sort_indices = np.argsort(evals)[::-1]
                    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
                    x_v2, y_v2 = evecs[:, sort_indices[1]]

                    #Uncomment lines below to see the eigenvectors
                    #scale = 20
                    #plt.plot([x_v1*-scale*2, x_v1*scale*2],
                            #[y_v1*-scale*2, y_v1*scale*2], color='red')
                    #plt.plot([x_v2*-scale, x_v2*scale],
                            #[y_v2*-scale, y_v2*scale], color='blue')

                    #plt.axis('equal')
                    #plt.gca().invert_yaxis()  # Match the image system with origin at top left
                    #plt.show()
                    theta1 = np.arctan((x_v1)/(y_v1)) 
                    theta2 = np.arctan((x_v2)/(y_v2)) 
                    theta2deg = theta2 * 180 / 3.14
                    theta1deg = theta1 * 180/  3.14


                    rotation_mat = np.matrix([[np.cos(theta2), -np.sin(theta2)],
                              [np.sin(theta2), np.cos(theta2)]])
                    rotatedimage = transform.rotate(image,-theta2deg, resize = False, mode = "edge" )


                    #Trial 2
                    #DO bad segmentation

                    testimage = rotatedimage[:,:,0] 
                    thresh = threshold_otsu(testimage) 
                    testimage = testimage > thresh
                    testimage = invert(testimage)
                    testimage = label(testimage)

                    testimage = remove_small_objects(testimage, min_size)
                    testimage = testimage > 0



                    ySec, xSec = np.nonzero(testimage)
                    xSec = xSec - np.mean(xSec)
                    ySec = ySec - np.mean(ySec)
                    coordsSec = np.vstack([xSec, ySec])

                    try:
                        covSec = np.cov(coordsSec)
                        evalsSec, evecsSec = np.linalg.eig(covSec) 


                        sort_indicesSec = np.argsort(evalsSec)[::-1]
                        x_v1Sec, y_v1Sec = evecsSec[:, sort_indicesSec[0]]  # Eigenvector with largest eigenvalue
                        x_v2Sec, y_v2Sec = evecsSec[:, sort_indicesSec[1]]


                        theta1Sec = np.arctan((x_v1Sec)/(y_v1Sec)) 
                        theta2Sec = np.arctan((x_v2Sec)/(y_v2Sec)) 
                        theta2degSec = theta2Sec * 180 / 3.14
                        theta1degSec = theta1Sec * 180/3.14
                    except: 
                            theta2degSec = 0
                            theta1degSec = 0

                    count = count + 1
                    
                    if flip and Name[-1] == 'R'and theta1degSec < 0:                                          #uncomment to apply vertical flipping
                        flippedimage = np.flip(rotatedimage, axis = 0)
                        imwrite(Savedir + Name + '.tif', flippedimage)
                        if count%show_after == 0:
                           doubleplot(image, flippedimage, "Original", "Rotated-and-Flipped")

                    if Name[-1] == 'R'and theta1degSec >= 0:   
                        imwrite(Savedir + Name + '.tif', rotatedimage)
                        if count%show_after == 0:
                            doubleplot(image, rotatedimage, "Original", "Rotated")    

                    if flip and Name[-1] == 'L'and theta1degSec >= 0:                                         #uncomment to apply vertical flipping
                        flippedimage = np.flip(rotatedimage, axis = 0)
                        imwrite(Savedir + Name + '.tif', flippedimage)
                        if count%show_after == 0:
                           doubleplot(image, flippedimage, "Original", "Rotated-and-Flipped")    

                    if Name[-1] == 'L'and theta1degSec < 0:   
                        imwrite(Savedir + Name + '.tif', rotatedimage)
                        if count%show_after == 0:
                            doubleplot(image, rotatedimage, "Original", "Rotated")
                      
                        
        Raw_path = os.path.join(Savedir, '*tif')

        axes = 'YXC'
        filesRaw = glob.glob(Raw_path)
        filesRaw.sort
        count = 0
        
        for fname in filesRaw:
                                
                                #Read image        
                                image = imread(fname)
                                Name = os.path.basename(os.path.splitext(fname)[0])
                          
                              
                                if Name[-1] == 'R':
                                    image = transform.rotate(image,  180,  resize=False)
                                    image = np.flip(image, axis = 0)
                                    imwrite(Savedir + Name + '.tif', image)
                                x = image[:,:,0:3]
                                
                    
                                #Make sure image is 2D
                    
                                Segmented = UnetModel.predict(x,axes)
                                thresh = threshold_otsu(Segmented) 
                                Binary = Segmented > thresh
                                Filled = binary_fill_holes(Binary[:,:,0])
                                Finalimage = remove_small_objects(Filled, min_size)
                         
                                #Compartment model
                                if UnetCompartmentModel is not None:
                                        SegmentedCompartment = UnetCompartmentModel.predict(x,axes)
                                        threshComp = threshold_otsu(SegmentedCompartment) 
                                        BinaryCompartment = SegmentedCompartment > threshComp
                                        FilledCompartment =label(BinaryCompartment[:,:,0])
                                        FilledCompartment = fill_label_holes(FilledCompartment)
                                        FilledCompartment = FilledCompartment > 0
                               
                                if UnetVeinModel is not None:
                                    
                                        SegmentedVeins = UnetVeinModel.predict(x, axes)
                                        threshVein = threshold_otsu(SegmentedVeins) 
                                        BinaryVein = SegmentedVein > threshVein
                                        FilledVein =label(BinaryVein[:,:,0])
                                        FilledVein = fill_label_holes(FilledVein)
                                        FilledVein = FilledVein > 0
                                indices = [np.where(FilledVein > 0)]    
                                FilledCompartment = np.multiply(FilledCompartment,Finalimage)
                                FilledCompartment[indeices] = 0
                                if count%show_after == 0:
                                  doubleplot(image,Finalimage, 'Original', 'UNET', plotTitle = 'Segmentation Result' )
                                  if UnetCompartmentModel is not None:
                                         doubleplot(image,FilledCompartment, 'Original', 'UNET', plotTitle = 'Compartment Segmentation Result' )
                                  if UnetVeinModel is not None:
                                        doubleplot(image,FilledVein, 'Original', 'UNET', plotTitle = 'Vein Segmentation Result' )
                                count = count + 1 
                                imwrite((MaskResults + 'Mask' + Name + '.tif' ) , Finalimage.astype('uint8'))
                                if UnetCompartmentModel is not None:
                                    
                                    imwrite((MaskCompartmentResults + 'MaskCompartment' + Name + '.tif' ) , FilledCompartment.astype('uint8'))
                                    imwrite((MaskCompartmentLabelResults + 'MaskCompartment' + Name + '.tif' ) , label(FilledCompartment).astype('uint16'))
                                    
                    
        AsymmetryComputer(MaskResults,AsymmetryResults,AsymmetryResultsName, extra_title = "", computeAsymmetry = computeAsymmetry )
        if UnetCompartmentModel is not None:
                  AsymmetryComputer(MaskCompartmentResults,AsymmetryCompartmentResults,AsymmetryCompartmentResultsName, extra_title = "Compartment", computeAsymmetry = computeAsymmetry )
        

def Label_counter(filesRaw, ProbabilityThreshold, Resultdir, min_size = 10 ):


     AllCount = []
     AllName = []
     for fname in filesRaw:
        Name = os.path.basename(os.path.splitext(fname)[0])
        TwoChannel = imread(fname)
        SpotChannel = TwoChannel[:,0,:,:]
        Binary = SpotChannel > ProbabilityThreshold
        Binary = remove_small_objects(Binary, min_size = min_size)
        Integer = label(Binary)
        waterproperties = measure.regionprops(Integer, Integer)
        labels = []
        for prop in waterproperties:
            if prop.label > 0:
                     
                      labels.append(prop.label)
        count = len(labels)
        imwrite(Resultdir + Name + '.tif', Integer.astype('uint16')) 
        AllName.append(Name)
        AllCount.append(count)
        
     df = pd.DataFrame(list(zip(AllCount)), index =AllName, 
                                                  columns =['Count'])
    
     df.to_csv(Resultdir + '/' + 'CountMasks' +  '.csv')  
     df     
    
    



def multiplot(imageA, imageB, imageC, titleA, titleB, titleC, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()
    ax[2].imshow(imageC, cmap=plt.cm.nipy_spectral)
    ax[2].set_title(titleC)
    ax[2].set_axis_off()
    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off()
      
def doubleplot(imageA, imageB, titleA, titleB, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off() 

def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)




def load_full_training_data(directory, filename,axes=None, verbose= True):
    """ Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'     
    """
    
    if directory is not None:
      npzdata=np.load(directory + filename)
    else:
      npzdata=np.load(filename)  
    
    
    X = npzdata['data']
    Y = npzdata['label']
    
    
        
    
    if axes is None:
        axes = npzdata['axes']
    axes = axes_check_and_normalize(axes)
    assert 'C' in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
  
    
    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']
    

       

    X = move_channel_for_backend(X,channel=channel)
    
    axes = axes.replace('C','') # remove channel
    if backend_channels_last():
        axes = axes+'C'
    else:
        axes = axes[:1]+'C'+axes[1:]

   

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple( X.shape[ax[a]] for a in 'TZYX' if a in axes )
        n_dim = len(image_size)
        n_channel_in = X.shape[ax['C']]

        print('number of  images:\t', n_train)
       
        print('image size (%dD):\t\t'%n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in)

    return (X,Y), axes


def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first','channels_last')
    return K.image_data_format() == 'channels_last'


def move_channel_for_backend(X,channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)
        

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes
def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedt     
    
    
def _raise(e):
    raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)    
