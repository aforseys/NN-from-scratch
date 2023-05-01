#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 14:21:54 2023

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

##initialize NN 
lsize = [784, 140, 10] #(input layer must be 784 and output layer must be 10), hidden layers can be anything 
nn = NeuralNetwork(lsize, activation_functions.ReLU, activation_functions.dReLU, loss_functions.cross_entropy, loss_functions.dcross_entropy)

x = training_images[0,:]/255 #normalize training data by dividing by 255
y = training_one_hot_labels[0,:]

#Finite difference checking
#Solve with initial values 
vecs, error = nn.feedforward(np.transpose(x), np.transpose(y))

A1,b1 = nn.params[0]
A2,b2 = nn.params[1]
z = vecs[-1]
a1 = A1.dot(vecs[0])+b1
a2 = A2.dot(vecs[1])+b2

W2_T = np.transpose(loss_functions.dcross_entropy(y.reshape(np.shape(y)[0],1), z))
W1_T = W2_T.dot(activation_functions.dReLU(a2)).dot(A2)

#Introduce small delta 
dA1 =np.random.normal(scale = 0.001, size= (np.shape(A1)[0], np.shape(A1)[1]))
db1 = np.random.normal(scale = 0.001, size= (np.shape(b1)[0], np.shape(b1)[1]))
dA2 = np.random.normal(scale = 0.001, size= (np.shape(A2)[0], np.shape(A2)[1]))
db2 = np.random.normal(scale = 0.001, size= (np.shape(b2)[0], np.shape(b2)[1]))

#Delta b checks (WRT Z, not L)
dz_db1 = activation_functions.dReLU(a2).dot(A2).dot(activation_functions.dReLU(a1))
dz_db2 = activation_functions.dReLU(a2)
dz_from_db1 = dz_db1.dot(db1)
dz_from_db2 = dz_db2.dot(db2)

#Delta A checks (WRT Z, not L)
dz_from_dA2 = activation_functions.dReLU(a2).dot(dA2).dot(vecs[1])
dz_from_dA1 = activation_functions.dReLU(a2).dot(A2).dot(activation_functions.dReLU(a1)).dot(dA1).dot(vecs[0])

#Deltas wrt L
grad_A1 = np.outer(np.transpose(activation_functions.dReLU(a1)).dot(np.transpose(W1_T)), np.transpose(vecs[0]))
grad_b1 = W1_T.dot(activation_functions.dReLU(a1))
grad_A2 = np.outer(np.transpose(activation_functions.dReLU(a2)).dot(np.transpose(W2_T)), np.transpose(vecs[1]))
grad_b2 = W2_T.dot(activation_functions.dReLU(a2))

#grad_bs, grad_As = nn.backprop(x,y)

# print(np.mean(grad_bs[0][0]-grad_b1))
# print(np.mean(grad_bs[1][0]-grad_b2))
# print(np.mean(grad_As[0][0]-grad_A1))
# print(np.mean(grad_As[1][0]-grad_A2))

#Test adding deltas 
#nn.params[0] = (np.add(A1, dA1), b1)
#nn.params[0] = (A1, np.add(b1,db1))
#nn.params[1] = (A2+dA2, b2)
#nn.params[1] = (A2,b2+db2)
nn.params[0] = (A1+dA1, b1+db1)
nn.params[1] = (A2+dA2, b2+db2)

#Solve for values with deltas 
vecs2, error2 = nn.feedforward(np.transpose(x), np.transpose(y))

#Compare delta of dz with new params against delta * gradient 
print("delta z", vecs2[2]-vecs[2])
print("delta error", error2 - error)

print("difference between delta z's", dz_from_dA1+ dz_from_db1 + dz_from_dA2+dz_from_db2-(vecs2[2]-vecs[2]))

print("calculated error", np.multiply(grad_A1, dA1).sum() + np.multiply(grad_A2, dA2).sum() + np.dot(grad_b1, db1) + np.dot(grad_b2, db2))





