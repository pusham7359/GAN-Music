# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:14:00 2018

@author: Bhargav
"""

from midiutil import MIDIFile
import numpy as np
import random
from scipy.io import loadmat
import keras
import midi_manipulation
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

    
# HyperParameters
# First, let's take a look at the hyperparameters of our model:

lowest_note = 21  # the index of the lowest note on the piano roll
highest_note = 108 # the index of the highest note on the piano roll
note_range = highest_note - lowest_note  # the note range

num_timesteps = 15  # This is the number of timesteps that we will create at a time
n_visible = 2 * note_range * num_timesteps  # This is the size of the visible layer.
n_hidden = 50  # This is the size of the hidden layer

num_epochs = 200  # The number of training epochs that we are going to run.
# For each epoch we go through the entire data set.
batch_size = 100  # The number of training examples that we are going to send through the RBM at a time.
lr = tf.constant(0.005, tf.float32)  # The learning rate of our model

# Generator
generator = Sequential([
        
        Dense(300),
        LeakyReLU(alpha=0.02),
        
        Dense(300),
        LeakyReLU(alpha=0.02),
        
        Dense(300),
        LeakyReLU(alpha=0.02),
        
        Dense(300, input_shape=(1,)),
        Activation('tanh')
    ])

generator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
generator.fit(train,epochs=5)
