#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:38:03 2023

@author: aforsey
"""
import numpy as np
from activation_functions import softmax

class NeuralNetwork:
    def __init__(self,lsize,activation,dactivation,loss, dloss):
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
        self.dloss = dloss
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
            A = np.random.randn(self.layers[l+1], lsize) #A is sized mxn where n is height of current layer and m = height of next layer 
            b = np.random.randn(self.layers[l+1], 1) #b is sized nx1 
            self.params[l] = (A,b)
            
    def feedforward(self, x,y, print_flag=False):
        vecs = []
        v = x.reshape(np.shape(x)[0],1)
        vecs.append(v)
        for k in range(len(self.layers[:-1])): #double check appropriate indexing for architecture 
            A,b = self.params[k]
            
            v = np.dot(A,v)+b 
            v = self.act(v)
        #    print('after act', np.shape(v))
            vecs.append(v)
       # A,b = self.params[k+1]
       # v = np.dot(A,v)+b
      #  vecs[-1]=softmax(vecs[-1]) #change output to softmax of output #don't do this here, bcus need just z later
        
      #  print('output z', v)
        error = self.loss(y, softmax(v), print_flag)
        
      #  print('error',error)
        
        return vecs, error
            
    def backprop(self,X,Y, alpha=0.001):
        grad_bs = {l:[] for l in range(len(self.layers)-1)}
        grad_As = {l:[] for l in range(len(self.layers)-1)} 
        
        for i in range(X.shape[0]):
        #    x = np.transpose(X.reshape(1, np.shape(X)[0]))
        #    y = np.transpose(Y.reshape(1, np.shape(Y)[0]))
            x = np.transpose(X[i].reshape(1,np.size(X[i])))
            y = np.transpose(Y[i].reshape(1,np.size(Y[i])))
            vecs, error = self.feedforward(x,y)
            z = vecs[-1]
     #       print("output z from end of vecs", z)
            W_T = np.transpose(self.dloss(y,z))    #initial W that is used to calculate all other W's is just dL/dz
            
            #print(W_T)
            
            for l, lsize in reversed(list(enumerate(self.layers[:-1]))): #go from last layer to first layer 
              #  print("W_T", W_T)
              #  print("l")
                A,b = self.params[l]
                a = A.dot(vecs[l])+b
                
               # print("dReLU/da shape", np.shape(self.dact(a)))
                
              #  print("W^T shape", np.shape(W_T))
        
                #calculate gradients (don't actually have to transpose ReLU deriv)
                grad_A = np.outer(np.transpose(self.dact(a)).dot(np.transpose(W_T)), np.transpose(vecs[l])) #FIX THIS SO THAT IT'S ELEMENTWISE PRODUCT (nevermind, only when mult da)
                grad_b = W_T.dot(self.dact(a)) #output is row vector => what is multiplied by delta b (a col vector) to get scalar df
                
                #store
                grad_As[l].append(grad_A)
                grad_bs[l].append(np.transpose(grad_b))
                
                #FIX THIS SO THAT TAKE MEAN OF THESE AS GO ALONG, DON'T STORE! BIG MATRICES AND A LOT OF THEM! 
                
                W_T = W_T.dot(self.dact(a)).dot(A) #calc W using old W (make sure that W^T correct..) 
                
       # return grad_bs, grad_As
        #return grad_As, grad_bs
        #TO DO: CHECK/DEBUG BY COMPARING BACK PROP W FINITE DIFFERENCE 
    
        #Find average of gradients by taking average across each list in each layer of both dictionaries 
        A_grad_descent_step = {l: np.mean(grad_As[l],axis=0)for l in range(len(self.layers)-1) }
        b_grad_descent_step = {l: np.mean(grad_bs[l],axis=0)for l in range(len(self.layers)-1) }
        
        #Gradient descent: Update self.params by taking those steps!!
        for l in range(len(self.layers)-1):
            A,b = self.params[l] 
       #     print('delta A', -alpha*grad_As[l][0])
        #    print('delat B', -grad_bs[l][0])
            self.params[l] = (A-alpha*A_grad_descent_step[l][0], b - alpha*b_grad_descent_step[l][0]) #right now takes every step it finds 
            
    def train_minibatch(self, X, Y, batch_size):
        #break up data into batches 
        X_batched = [X[i:i + batch_size,:] for i in range(0, len(X), batch_size)]
        Y_batched = [Y[i:i + batch_size,:] for i in range(0, len(Y), batch_size)]
        
        for i in range(len(X_batched)):
            self.backprop(X_batched[i],Y_batched[i])
            
    def train_epoch(self, X,Y, epochs,batch_size):
        for i in range(epochs):
            self.train_minibatch(X,Y,batch_size)
        
    #def train_GD(self, num):
    
    #def train_SGD(self, num):
        
    def test(self,X,Y):
        errors = []
        for i in range(np.shape(X)[0]):
            vecs, error = self.feedforward(np.transpose(X[i].reshape(1,np.size(X[i]))), np.transpose(Y[i].reshape(1,np.size(Y[i]))))
            errors.append(error)
        print(errors)
        print(np.mean(errors))
        return errors
            
        
