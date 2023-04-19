#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:38:03 2023

@author: aforsey
"""
import numpy as np
from activation_functions import softmax

class NeuralNetwork:
    def __init__(self,lsize,activation,dactivation, loss):
        """
        Parameters
        ----------
        lsize : List with length equal to number of layers. Each entry gives 
        the size of the layer. First entry gives number of inputs and last 
        entry gives number of outputs.
        activation_fcn : Activation function used throughout. 
        loss_fcn : Loss function used for training.
        """
        self.layers=lsize
        self.inputs = lsize[0]
        self.outputs = lsize[-1]
        self.act = activation
        self.dact = dactivation
        self.loss = loss
        self.init_params()

    def init_params(self):
        """
        Creates dictionary representation of NN params. Keys are integers
        that correspond to which layer params are applied to from 0 to N-1, 
        with 0 being the input layer and N-1 being the last layer before output.
        The value of each key is a tuple containing a randomized matrix A and a 
        randomized column vector b. A is sized N_{k+1} x N_k, and b is sized 
        N_{k+1} x 1, where N_k represents the size of layer k 
        specified in the input parameters. 
        """
        #what is best way to initialize NN params?
        self.params ={}
        for l,lsize in enumerate(self.layers[:-1]):  #don't select parameters for output
            A = np.random.randn(self.layers[l+1], lsize)
            b = np.random.randn(self.layers[l+1], 1)
            self.params[l] = (A,b)
            
    def feedforward(self, x,y):
        v = x
        for k in range(len(self.layers[:-1])): #double check appropriate indexing for architecture 
            A,b = self.parms[k]
            v = np.dot(A,v)+b 
            v = self.act(v)
        
       # A,b = self.params[k+1]
       # v = np.dot(A,v)+b
        v=softmax(v)
            
        return v
            
    #def backprop(self,batch):
     #   for x,y in batch:
      #      v = self.feedforward(x,y)
            #calculate gradients
        #take average of gradients
        #take step 

    #def train_minibatch(self,num):
        #break data up by input number 
        #repeat backprop for all groups
        
    #could implement training different ways (like GC or SGC)
    
        

