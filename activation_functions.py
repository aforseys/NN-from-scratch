#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:04:55 2023

@author: aforsey
"""
from math import e

def ReLU(z):
    if z>0:
        return z
    elif z<=0:
        return 0
    
def sigmoid(z):
    return 1/(1+e**(-z))

#Tanh similar to sigmoid but pushes values to -1 and 1 (instead of 0 and 1)
#also has a much greater gradient in the center region near zero 
#--> gives higher values of gradient during training, and larger updates
#in weights in NN during training --> gives strong gradients and big learnign steps
#also symmetric about zero so faster convergence(?)..
def tanh(z):
    return (e**(2*z)-1)/(e**(2*z)+1)
    
#output latyer activation function (takes in VECTOR z and outputs a vector)
def softmax(z):
    z_softmax = [e**z_i for z_i in z]
    return z_softmax/sum(z_softmax)
