#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:04:55 2023

@author: aforsey
"""
import numpy as np

def ReLU(z):
    
    """
    Non linear activation function. 
    Sets all negative values to zero. 
    
    Input is VECTOR z, output is VECTOR with ReLU applied element wise to each 
    term.
    
    """
    
    return np.maximum(0,z)

def dReLU(z):
    """
    Derivative of ReLU wrt input vector z. 
    Diagonal square matrix sized z x z. Index i,i = 1 if z[i]>0, 0 otherwise.
    
    """
    
    return np.diag((1*(z>0)).reshape(np.shape(z)[0],)) #flatten array before diagonalization
    

def softmax(z_vec):
    """
    Output layer activation function.
    
    Input is VECTOR z, output is a VECTOR representing the probability a
    distribution over categorical data.
    """

    z_softmax = z_vec - np.max(z_vec)
    

    return np.exp(z_softmax) / np.sum(np.exp(z_softmax))
    