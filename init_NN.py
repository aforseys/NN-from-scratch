#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:38:03 2023

@author: aforsey
"""
import numpy as np
#consider this changing to be a class, could be easier

def build_NN(lsize):
    """
    Parameters
    ----------
    lsize : list
        List length is equal to the number of hidden layers. Each value 
        represents the size of that hidden layer. First entry gives the number 
        of inputs and last entry gives the number of outputs.

    Returns
    -------
    Dictionary representation of NN with input structure. Keys are integers
    that correspond to each layer from 1 to k, with 1 being the input layer and 
    k being the output layer. The value for each layer is a tuple containing a 
    randomized matrix A and a randomized column vector b for each layer. A is 
    sized N_{k+1} x N_k, and b is sized N_{k+1} x 1, where N_k represents the 
    size of layer k specified in the input parameters. 
    """
    
    #what is best way to initialize NN params?
    NN ={}
    for layer,size in enumerate(lsize[:-1]):  #don't want to select parameters for output, no F there
        A = np.random.randn(lsize[layer+1], layer)
        b = np.random.randn(lsize[layer+1], 1)
        NN[layer] = (A,b)
        
    return NN
    