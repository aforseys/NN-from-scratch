#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:01:40 2023

@author: aforsey
"""
import numpy as np
from activation_functions import softmax

def cross_entropy(y, s):
    """
    Loss function used for softmax output
    
    Parameters
    ----------
    y : 1-hot vector encoding of training sample's true category 
    s: output of softmax function on NN output z

    Returns
    -------
    Integer value for loss that penalizes output probabilities for
    each category based on how far the probability is from the true value. 
    
    Notes 
    -------
    np.log uses base e, although cross entropy often calculated with log 
    base 2. 
    
    """

    c = np.dot(np.transpose(y), np.log(s)) 
    
    return -c


def dcross_entropy(y,z):
    """
    Parameters
    ----------
    y : 1-hot vector encoding of training sample's true category 
    z : NN output (INCLUDES DERIV OF SOFTMAX)

    Returns
    -------
    derivative of cross entropy loss function with respect to NN output z

    """

    return softmax(z)-y


def cross_entropy_w_softmax(y,z):
    """
    Loss function that combines softmax and cross entropy. Use for more complex
    NN where need to use tricks to prevent computational errors when computing
    softmax and cross entropy of large output values. Difference between this
    and other cross entropy is this takes in the output vector z BEFORE softmax
    is applied, and calculates same output value but without running into 
    computational errors.
    
    Parameters
    ----------
    y : 1-hot vector encoding of training sample's true category 
    z: output of NN feedforward pass
    
    """
    
    z_max = np.max(z)
    
    #trick used to calculate log of softmax without errors
    log_softmax = z-z_max - np.log(np.sum(np.exp(z-z_max)))
    
    return -np.dot(np.transpose(y),log_softmax)