#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Mon May  1 11:34:36 2023
=======
Created on Mon May  1 14:38:19 2023
>>>>>>> backprop-test

@author: aforsey
"""

import os 
import numpy as np
import idx2numpy 
import activation_functions
import loss_functions

from init_NN import NeuralNetwork


#import training data 
training_images = idx2numpy.convert_from_file(os.getcwd()+'/training_data/train-images.idx3-ubyte')
training_labels = idx2numpy.convert_from_file(os.getcwd()+'/training_data/train-labels.idx1-ubyte')

#flatten training images
training_images = training_images.reshape(np.shape(training_images)[0], np.shape(training_images)[1]*np.shape(training_images)[2])

#make training lables one-hot vectors 
training_one_hot_labels = np.zeros((training_labels.size, training_labels.max()+1)) #have to add one because includes 0 as possible number 
training_one_hot_labels[np.arange(training_labels.size),training_labels] = 1

#import test data 
test_images = idx2numpy.convert_from_file(os.getcwd()+'/test_data/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file(os.getcwd()+'/test_data/t10k-labels.idx1-ubyte')

#flatten test images
test_images = test_images.reshape(np.shape(test_images)[0], np.shape(test_images)[1]*np.shape(test_images)[2])

#make test labels one-hot vectors 
test_one_hot_labels = np.zeros((test_labels.size, test_labels.max()+1)) #have to add one because includes 0 as possible number 
test_one_hot_labels[np.arange(test_labels.size),test_labels] = 1


#shuffle data 
p = np.random.permutation(len(training_images))
training_images_shuffled = training_images[p]
training_one_hot_labels_shuffled = training_one_hot_labels[p]


##initialize NN 
lsize = [784, 128, 10] #(input layer must be 784 and output layer must be 10), hidden layers can be anything 
nn = NeuralNetwork(lsize, activation_functions.ReLU, activation_functions.dReLU, loss_functions.cross_entropy, loss_functions.dcross_entropy)

nn.train_epoch(training_images_shuffled[0:60000,:]/255, training_one_hot_labels_shuffled[0:60000,:], epochs = 1, batch_size=1)

#nn.train_minibatch(training_images_shuffled[0:60000,:]/255, training_one_hot_labels_shuffled[0:60000,:], batch_size=100)

nn.test(test_images[0:10000,:]/255, test_one_hot_labels[0:10000,:])

##initialize NN 
# lsize = [784, 140, 10] #(input layer must be 784 and output layer must be 10), hidden layers can be anything 
# nn = NeuralNetwork(lsize, activation_functions.ReLU, activation_functions.dReLU, loss_functions.cross_entropy, loss_functions.dcross_entropy)

# nn.train_epoch(training_images_shuffled/255, test_images_shuffled, epochs=10, batch_size=10)

# nn.test(test_images/255, test_one_hot_labels)

