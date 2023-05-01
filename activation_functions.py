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
    # print(z)
    # print(type(z))
    # print('test', (z>0))
    # print('test2', np.shape(1*(z>0)))
    # print(np.diag((1*(z>0))))
    
    # print((1*(z>0)).reshape(np.shape(z)[0],))
    
    # print(np.diag((1*(z>0)).reshape(np.shape(z)[0],)))
    
    return np.diag((1*(z>0)).reshape(np.shape(z)[0],)) #flatten array before diagonalization
    
   # return np.diag(1*(z>0)) #HOW DOES THIS OUTPUT A MATRIX WHEN APPLIED TO A VECTOR?????
    
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
    #print(z_vec)
    #z_softmax = [np.exp(z_i) for z_i in z_vec]
    z_softmax = z_vec #- np.max(z_vec) #NORMALIZATION TRICK TO HELP W NUMERICAL STABILITY (doesn't change output)
  #  print("softmax output", np.exp(z_softmax) / np.sum(np.exp(z_softmax)))
    return np.exp(z_softmax) / np.sum(np.exp(z_softmax))
    
   # print('softmax', z_softmax)
    
   # print(z_softmax)
   # return z_softmax/sum(z_softmax)
