#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:21:54 2023

@author: aforsey
"""
import os 
import numpy as np
import idx2numpy 


def load_data(images_path, labels_path):
    """
    Takes in paths to images file and path to lables file. 
    Assumes files are unzipped idx files avilable at: 
    http://yann.lecun.com/exdb/mnist/
                                           
    Returns flattened, normalized image vectors (length 728)  and one-hot 
    vector encodings of corresponding image labels.                                  
    """

    image_file_loc = os.join(os.getcwd(),images_path)
    label_file_loc = os.join(os.getcwd(),labels_path)

    #import data 
    images = idx2numpy.convert_from_file(image_file_loc)
    labels = idx2numpy.convert_from_file(label_file_loc)

    #flatten images
    images = images.reshape(np.shape(images)[0], np.shape(images)[1]*np.shape(images)[2])

    #make labels one-hot vectors 
    one_hot_labels = np.zeros((labels.size,labels.max()+1)) #have to add one because includes 0 as possible number 
    labels[np.arange(labels.size),labels] = 1
    
    #normalize images by dividing by 255
    return images/255, one_hot_labels



