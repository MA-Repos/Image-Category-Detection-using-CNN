#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:04:20 2019
"""
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D

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

##################
# Model Creation
##################
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=xTrain.shape[1:]))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256), activation='relu')
model.add(Dropout(0.5))
model.add(Dense(10), activation='softmax')

model.compile(loss = 'categorical_crossentropy',
              optimizer = SGD(0.1),
              metrics = ['accuracy'])

print(model.summary)
