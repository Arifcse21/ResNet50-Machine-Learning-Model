#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:31:15 2021

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
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import layer_normalization
from keras.layers.convolutional import *
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm
from glob import glob 

from PIL import Image


np.random.seed(42)

DESIRED_WIDTH = 224 #Change back to 224
DESIRED_HEIGHT = 224

train_path = 'Splitted_Dataset/train/'
valid_path = 'Splitted_Dataset/val/'
test_path =  'Splitted_Dataset/test/'

test_batches = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_batches = ImageDataGenerator(  preprocessing_function=preprocess_input,
    rotation_range=40,     # randomly rotate pictures
    width_shift_range=0.1, # randomly translate pictures
    height_shift_range=0.1, 
    shear_range=0.2,       # randomly apply shearing
    zoom_range=0.2,        # random zoom range
    horizontal_flip=True)  # randomly flip image

valid_batches = ImageDataGenerator( preprocessing_function=preprocess_input)


ds_for_plot = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,   # using train dataset only
    shuffle = True,
    image_size = (DESIRED_WIDTH, DESIRED_HEIGHT),
    batch_size = 32   
    
)


class_names = ds_for_plot.class_names
class_names

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(DESIRED_WIDTH,DESIRED_HEIGHT), classes=class_names, batch_size=32)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(DESIRED_WIDTH,DESIRED_HEIGHT), classes=class_names, batch_size=32)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(DESIRED_WIDTH,DESIRED_HEIGHT),classes=class_names, batch_size=32)




plt.figure(figsize =(20, 20))
for image_batch, label_batch in ds_for_plot.take(1):
    for i in range(10):
        ax = plt.subplot(2,5,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
N = 0  # total files
for dirpath, dirnames, filenames in os.walk(train_path):
    N_c = len(filenames)
    N += N_c
    print ("Files in ", dirpath, N_c)
print ("Total Files ",N)

N = 0  # total files
for dirpath, dirnames, filenames in os.walk(valid_path):
    N_c = len(filenames)
    N += N_c
    print ("Files in ", dirpath, N_c)
print ("Total Files ",N)


N = 0  # total files
for dirpath, dirnames, filenames in os.walk(test_path):
    N_c = len(filenames)
    N += N_c
    print ("Files in ", dirpath, N_c)
print ("Total Files ",N)



train_n=[]
valid_n=[]
test_n=[] 
i=0
for  dirpath, dirnames, filenames in os.walk(train_path):
    N = len(filenames)
    train_n.insert(i,N)
    # print(N)
    
for  dirpath, dirnames, filenames in os.walk(valid_path):
    N = len(filenames)
    valid_n.insert(i,N)
    # print(N)
    
for  dirpath, dirnames, filenames in os.walk(test_path):
    N = len(filenames)
    test_n.insert(i,N)
    # print(N)
#################################
train_n.sort()
valid_n.sort()
test_n.sort()
#################################
plotdata = pd.DataFrame({
    
    "Train": train_n[1:],
    "Valid": valid_n[1:],
    "Test": test_n[1:]
    
},
index = class_names)

plotdata.plot(kind='bar', figsize=(15, 8))
plt.title("Data Visualization")
plt.xlabel("Classes")
plt.ylabel("No. of dataset")
################################
print(train_n[1:])
print(valid_n[1:])
print(test_n[1:])
################################
