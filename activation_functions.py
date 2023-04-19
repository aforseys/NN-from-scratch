#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:04:55 2023

@author: aforsey
"""
import numpy as np

def ReLU(z):
    return np.maximum(0,z)

def dReLU(z):
    return 1*(z>0)
    
def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    """
    Note: tanh is very similar to the sigmoid function, but pushes values to 
    -1 and 1. Tanh also has a much greater gradient in the center region near
    zero, therefore it gives higher values of gradient during training resulting
    in larger updates in weights during training. 
    """
    return (np.exp(2*z)-1)/(np.exp(2*z)+1)
    

def softmax(z_vec):
    """
    Output layer activation function.
    
    Input is VECTOR z, output is a VECTOR representing the probability a
    distribution over categorical data.
    """
    z_softmax = [np.exp(z_i) for z_i in z_vec]
    return z_softmax/sum(z_softmax)
