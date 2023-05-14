#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:38:03 2023

@author: aforsey
"""
import numpy as np
from tqdm import tqdm
from activation_functions import softmax

class NeuralNetwork:
    def __init__(self,lsize,activation,dactivation,loss, dloss, learning_rate=0.01):
        """
        Initializes neural net. Assumed neural net architecture has a linear 
        function and an activation function applied at each layer, EXCEPT no 
        activation function is added bewteen the last layer and the output, only
        a linear function. 
        
        Parameters
        ----------
        lsize : List with length equal to number of layers. Each entry gives 
        the size of the layer. First entry gives number of inputs and last 
        entry gives number of outputs.
        activation : Activation function used throughout. 
        dactivation: Derivative of activation function used throughout.
        loss : Loss function used for training.
        dloss: Derivative of loss function used throughout. 
    
        """
        self.layers= lsize
        self.inputs = lsize[0]
        self.outputs = lsize[-1]
        self.act = activation
        self.dact = dactivation
        self.loss = loss
        self.dloss = dloss
        self.init_params()
        self.learning_rate = learning_rate

    def init_params(self):
        """
        Creates dictionary representation of NN params. NN params are the values of
        A and b in the linear functions applied at each layer. 
        
        Dictionary keys are integers corresponding to each layer of the NN. The
        value for each key is a tuple (A,b) of the params for the linear function
        applied at that layer. The key 0 gives the values of (A,b) applied to the
        input vector. 
        
        At a given layer k, A_k is sized as l_{k+1} x l_k, where l_k is the size
        of the input vector to layer k, and l_{k+1} is the size of the desired 
        output vector at layer k+1. The parameter b_k is sized l_{k+1}. 
        
        The parameters A and b for each layer are randomly initialized from a 
        normal distribution centered at 0 with and std of 1. Believe this is the
        default initialization for PyTorch.
        """
        #what is best way to initialize NN params?
        self.params ={}
        for l,lsize in enumerate(self.layers[:-1]):  #don't select parameters for output
           
            A = np.random.normal(0, 1, (self.layers[l+1], lsize))
            b = np.random.normal(0, 1, (self.layers[l+1], 1))

            self.params[l] = (A,b)
            
         #   print(f'initializaed values for {l}:', (np.min(A), np.max(A), np.min(b), np.max(b)))
            
    def feedforward(self, x,y):
        """
        Parameters
        ----------
        x: flattened image vector 
        y: corresponding one-hot vector encoding of image label
        
        Returns
        -------
        vecs: List of vector representation at each layer, beginning with input
        layer, including hidden layers, and ending with output layer. Helpful to
        store for calculation of gradient. 
        error: scalar value of cross entropoy loss
        
        Takes in a sample and passes through the network. Applies linear function
        then activation function to each layer, except between last hidden layer 
        and output layer only applies linear function. The error is calculated 
        using the model's specified loss function. Currently this takes the output
        vector z, applies the softmax function, then finds the cross entropy loss
        of this output with the one-hot vector encoding of the sample label, y. 
        """
        
        vecs = []
        v = x.reshape(np.shape(x)[0],1)
        vecs.append(v)
        for k in range(len(self.layers[:-1])): 
        
            A,b = self.params[k]
            v = np.dot(A,v)+b 
            
            #apply ReLU excpet if it's the last layer
            if k != len(self.layers[:-1])-1: 
           
                v = self.act(v)
                
            vecs.append(v)
        
        error = self.loss(y, v)
        
        return vecs, error
            
    def backprop(self,X,Y):
        """
        Parameters
        ----------
        X : Matrix of flattened image vectors (each row represents one image)
        that make up the batch that should be used to inform one parameter update.
        
        Y : Corresponding matrix of one-hot encodings of image labels (each
        corresponds to one image). 

        Returns
        -------
        losses of each batch.
        
        For each sample in the batch: (1)calculates error and vectors at each layer 
        through feed forward pass and (2)calculates gradient for each parameter. 
        
        Then takes average of gradients across all samples in the batch, and updates
        parameters using this average. Parameter update = -learning_rate*gradient.
        
        """
        #Keep track of parameter gradients at each layer 
        grad_b = {l:0 for l in range(len(self.layers)-1)}
        grad_A = {l:0 for l in range(len(self.layers)-1)} 
        batch_size = X.shape[0]

        losses = []
        
       #compute gradient for each sample in batch 
        for i in range(batch_size):
            x = np.transpose(X[i].reshape(1,np.size(X[i])))
            y = np.transpose(Y[i].reshape(1,np.size(Y[i])))
            
            #Find intermediate vectors and error from feed forward pass 
            vecs, error = self.feedforward(x,y)
            z = vecs[-1]
            losses.append(error)

            #Find initial W = dL/dz, used to calculate all other W's 
            W_T = np.transpose(self.dloss(y,z)) 
            
            #backpropagate from last layer to first layer 
            for l, lsize in reversed(list(enumerate(self.layers[:-1]))): 
                A,b = self.params[l]
                a = A.dot(vecs[l])+b
                
                if l == len(self.layers[:-1])-1: #If last linear activation function
                    #calculate gradient differently bcus no ReLU function applied after 
                    grad_b_new = W_T
                    grad_A_new = np.outer(np.transpose(W_T), np.transpose(vecs[l]))
                    W_T = W_T.dot(A) 
          
                else: #Otherwise account for ReLU fcn after linear fcn 
                    grad_A_new = np.outer(np.transpose(self.dact(a)).dot(np.transpose(W_T)), np.transpose(vecs[l])) 
                    grad_b_new = W_T.dot(self.dact(a)) 
                    W_T = W_T.dot(self.dact(a)).dot(A) 
                 
                #add new gradient to sum of gradients for batch (will later divide by batch size)
                grad_A[l]+= grad_A_new
                grad_b[l]+= np.transpose(grad_b_new)
        
        #Update parameters 
        for l in range(len(self.layers)-1):
            
             A,b = self.params[l] 
             
             #take step in the direction of the average gradient over all samples in batch
             self.params[l] = (A-self.learning_rate*grad_A[l]/batch_size, b - self.learning_rate*grad_b[l]/batch_size) 
        
        return losses
  
          
    def train(self, X,Y, epochs,batch_size):
        """
        Parameters
        ----------
        X : Matrix of all training samples. Row for each training sample. 
        Y : Matrix of corresponding sample lables. Row for each sample. 
        epochs : Number of iterations through full training set. 
        batch_size : Number of batches to split training set into. 
        
        Returns
        -------
        None.
        
        For each epoch, makes an update to the parameters after each batch 
        has been processed. Therefore the number of updates to the params is 
        # updates =  # epochs * (#training samples/batch_size  +1) 
        (plus one accounts for # training samples not exactly divisible by batch size)
        """
        
        for i in tqdm(range(epochs)):
            print('epoch number',i)
            
            #shuffle data at start of each epoch
            p = np.random.permutation(len(X))
            X_shuffled = X[p]
            Y_shuffled = Y[p]
            
            #break data up into batches 
            X_batched = [X_shuffled[i:i + batch_size,:] for i in range(0, len(X_shuffled), batch_size)]
            Y_batched = [Y_shuffled[i:i + batch_size,:] for i in range(0, len(Y_shuffled), batch_size)]
            print('number of batches',len(X_batched))
            
            #Perform parameter update with batch 
            for j in tqdm(range(len(X_batched))):
                losses = self.backprop(X_batched[j],Y_batched[j])
                if (j+1) % 100 == 0:
                    print(f'Epoch [{i+1}/{epochs}], Step [{j+1}/{len(X_batched)}], Loss: {losses[-1]}')
            
        
    def test(self,X,Y):
        
        """
        Parameters
        ----------
        X : Matrix of all test samples. Row for each training sample. 
        Y : Matrix of corresponding sample lables. Row for each sample. 
        
        Returns
        -------
        cross_ent_errors : list of cross entropy error for each sample
        classification_matches: list of binary True or False value for each  sample.
        True if the model's most likely classification (argmax of softmax of model output z) 
        matches the one hot encoding of sample label. 
        
        Prints average cross entropy loss and percent of classification accuracy. 

        """
        
        cross_ent_errors = []
        classification_matches = []
        
        for i in range(np.shape(X)[0]):
            vecs, error = self.feedforward(np.transpose(X[i].reshape(1,np.size(X[i]))), np.transpose(Y[i].reshape(1,np.size(Y[i]))))
            cross_ent_errors.append(error[0][0])
            classification = np.argmax(softmax(vecs[-1])) 
            print(classification)
            print(Y[i][classification])
            classification_matches.append((Y[i][classification] == 1))
            
        print('Average cross entropy loss:', np.mean(cross_ent_errors))
        print(f"Classification accuracy: {(sum(classification_matches)/np.shape(X)[0])*100}%")

        return cross_ent_errors, classification_matches
    
        
