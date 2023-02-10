#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:37:52 2021

@author: arifcse21
"""
import tensorflow as tf
import numpy as np
import os
import pandas as pd

import keras
from keras import backend as K
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import layer_normalization
from keras.layers.convolutional import *
from keras.preprocessing import image
import itertools
import matplotlib.pyplot as plt
%matplotlib inline
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm
from glob import glob 
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Imageimport tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization


np.random.seed(42)

DESIRED_WIDTH = 224 #Resnet50 model required image size
DESIRED_HEIGHT = 224

train_path = 'tomato/train/'
valid_path = 'tomato/val/'
test_path =  'tomato/test/'


######Generating Train batch#################
train_batches = ImageDataGenerator(  preprocessing_function=preprocess_input,
    rotation_range=40,     # randomly rotate pictures
    width_shift_range=0.1, # randomly translate pictures
    height_shift_range=0.1, 
    shear_range=0.2,       # randomly apply shearing
    zoom_range=0.2,        # random zoom range
    horizontal_flip=True)  # randomly flip image

#################Generating validation batch########
valid_batches = ImageDataGenerator( preprocessing_function=preprocess_input)


########## dataset for training#######################
ds_for_plot = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,   # using train dataset only
    shuffle = True,
    image_size = (DESIRED_WIDTH, DESIRED_HEIGHT),
    batch_size = 32   
    
)
class_names = ds_for_plot.class_names         # extracting classes name from dataset
class_names

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(DESIRED_WIDTH,DESIRED_HEIGHT), classes=class_names, batch_size=32)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(DESIRED_WIDTH,DESIRED_HEIGHT), classes=class_names, batch_size=32)

############### Model Definition #################
model = Sequential()
"""

Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer.

We use the ‘add()’ function to add layers to our model.

Our first 2 layers are Flatten() and Dense() layers. These are convolution layers that will deal with our input images, which are seen as 2-dimensional matrices.

1280 in the first layer, 1280 in the 2nd layer and 640 in the Third layer are the number of nodes in each layer. This number can be adjusted to be higher or lower, depending on the size of the dataset


"""
model.add(ResNet50(include_top=False, weights='imagenet', input_shape=(DESIRED_WIDTH,DESIRED_HEIGHT,3)))

################ Multiple CNN layer initialization #######################
model.add(Flatten())                    # Flatten() -> make a 1d array
model.add(Dense(1280))                  # Dense layer --> Neurons (matrix-vector multiplications) those are connected to the next layer neorons
model.add(BatchNormalization())         # BatchNormalization to avoid overfitting and standardize input and output
model.add(Activation('relu'))
model.add(Dropout(0.5))                 # Dropout also prevent overfitting

model.add(Dense(1280))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(640))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))

model.add(Dense(10,activation='softmax'))                 # Number of classes is 10

adam = tf.keras.optimizers.Adam(learning_rate=0.00001)

############### Compiling model ##########
"""
we need to compile our model. Compiling the model takes three parameters: optimizer, loss and metrics.

The optimizer controls the learning rate. We will be using ‘adam’ as our optmizer. Adam is generally a good optimizer to use for many cases. The adam optimizer adjusts the learning rate throughout training.

The learning rate determines how fast the optimal weights for the model are calculated. A smaller learning rate may lead to more accurate weights (up to a certain point), but the time it takes to compute the weights will be longer.

We will use ‘categorical_crossentropy’ for our loss function. This is the most common choice for classification. A lower score indicates that the model is performing better.

To make things even easier to interpret, we will use the ‘accuracy’ metric to see the accuracy score on the validation set when we train the model.
"""


model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


################# Generating checkpoints for graph ############
from keras.callbacks import ModelCheckpoint
filepath = "Resnet50.h5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_acc', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')
callbacks_list = [checkpoint]

###################  Training the Model#######

history = model.fit(
        train_batches,
        batch_size=32,
        steps_per_epoch=8,
        epochs=200,
        callbacks=[callbacks_list,tf.keras.callbacks.CSVLogger('history_resnet50.csv')],
        verbose=2)



model.save(filepath)


history= pd.read_csv('/mnt/Documents/Project⁄Thesis/ResNet50/history_resnet50.csv') 
history = max(history['accuracy'])           # maximum accuracy valuee from the csv file
history


# Plotting accuracy and loss points
plt.plot(history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

