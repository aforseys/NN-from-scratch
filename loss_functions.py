#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:01:40 2023

@author: aforsey
"""
import numpy as np

#np.log uses base e, entropy often calculated with log base of 2 (something to note)
def cross_entropy(output, true_val):
    c = np.dot(true_val, np.log(output)) + np.dot((1-true_val), np.log(1-output))
    return -c

