#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:01:40 2023

@author: aforsey
"""
import numpy as np
from activation_functions import softmax

def cross_entropy(y, s, print_flag=False):
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
    base 2. No real effect but potentially something to change. 
    """
  #  print('y', y)
  #  print('s', s)
  #  print('log s and some', np.log(s+1e-15))
  #  print('test problem', np.log(s+1e-15)) #add small constant so don't take log(0)
  

    c = np.dot(np.transpose(y), np.log(s)) 
    
    if print_flag:
        print("s", s)
        print('y', y)
        print("cross entropy loss", -c)
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

#derivative of cross_entropy loss (including softmax) wrt output vector z is
#dL/dz = softmax(z)-y
#(derived in notebook from online sources)