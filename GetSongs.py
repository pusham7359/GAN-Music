# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:22:27 2018

@author: Bhargav
"""
import glob,midi_manipulation
from tqdm import tqdm
import numpy as np
def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e           
    return songs

songs = get_songs('Beethoven') #These songs have already been converted from midi to msgpack
print ("{} songs processed".format(len(songs)))
print(songs[0])
print(len(songs))
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
lr = 0.005  # The learning rate of our model

