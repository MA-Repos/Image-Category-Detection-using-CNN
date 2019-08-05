#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:04:20 2019
"""
import keras
from keras.datasets import cifar10
from keras.utils import np_utils

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

# Change data type to float
xTrain = xTrain.astype(float)
xTest = xTest.astype(float)

print('xTrain shape:', xTrain.shape)
print(xTrain.shape[0], 'training dataset sample')
print(xTest.shape[0], 'testing dataset sample')

#Normalization
xTrain /= 255
xTest /= 255

# Encode
xTrain = np_utils.to_categorical(yTrain, 10)
yTest = np_utils.to_categorical(yTest, 10)

