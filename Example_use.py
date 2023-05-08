#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:29:31 2023

@author: aforsey
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:34:36 2023

@author: aforsey
"""
import os
import activation_functions
import loss_functions
import process_MNIST
from init_NN import NeuralNetwork

#local path to files 
training_images_path = os.getcwd() + '/training_data/train-images.idx3-ubyte'
training_labels_path = os.getcwd() + '/training_data/train-labels.idx1-ubyte'
test_images_path = os.getcwd() + '/test_data/t10k-images.idx3-ubyte'
test_labels_path = os.getcwd()+ '/test_data/t10k-labels.idx1-ubyte'


#Loads MNIST data files. Assumes unzipped idx files avilable at: http://yann.lecun.com/exdb/mnist/
training_images, training_labels = process_MNIST.load_data(training_images_path, training_labels_path)
test_images, test_labels = process_MNIST.load_data(test_images_path, test_labels_path)


#Define NN architecture. Input size (first layer) and output size (last layer) can't change.
#Will have linear activation functions between each layer, and ReLU activation functions after 
#each linear activation function except between last hidden layer and output layer. 
lsize = [784, 128, 64, 10] #e.g. this structure will have 3 linear activation functions and 2 ReLU fcns

##initialize NN 
nn = NeuralNetwork(lsize, activation_functions.ReLU, activation_functions.dReLU, loss_functions.cross_entropy_w_softmax, loss_functions.dcross_entropy)

##train NN (example uses mini-batch)
nn.train(training_images, training_labels, epochs = 10, batch_size=1)
#gradient descent means batch size = # training samples (# steps you take = # epochs)
#stochastic gradient descent means batch size = 1

##test NN
cross_ent_error, classification_error = nn.test(test_images, test_labels)

