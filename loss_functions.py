#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:01:40 2023

@author: aforsey
"""
import numpy as np

def cross_entropy(NN_output, true_val):
    """
    Loss function used for softmax output
    
    Parameters
    ----------
    NN_output : output vector from NN output layer softmax function 
    true_val : 1-hot vector encoding of training sample's true category 

    Returns
    -------
    Integer value for loss that penalizes output probabilities for
    each category based on how far the probability is from the true value. 
    
    Notes 
    -------
    np.log uses base e, although cross entropy often calculated with log 
    base 2. No real effect but potentially something to change. 
    """
    
    c = np.dot(true_val, np.log(NN_output)) 
    return -c


#derivative of cross_entropy loss (including softmax) wrt output vector z is
#dL/dz = softmax(z)-y
#(derived in notebook from online sources)