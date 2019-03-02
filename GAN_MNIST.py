# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:27:24 2018

@author: Bhargav
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers.core import Flatten,Dropout
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras.utils import np_utils
#from keras import initializations

def dataInit():
	print('Loading the data')
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = np.concatenate((X_train, X_test), axis=0)
	X_train = (X_train.astype(np.float32) - 127.5)/127.5
	print('Training Data: ', X_train.shape)
	npRandom = np.random.RandomState(18)
	X_noise = []
	for i in range(X_train.shape[0]):
		randomNoise = npRandom.uniform(-1,1,100)
		X_noise.append(randomNoise)
	X_noise = np.array(X_noise)
	print('Random Noise Data: ', X_noise.shape)
	return X_train, X_noise

def saveImage(imageData, imageName, epoch):
	f, ax = plt.subplots(16, 8)
	k = 0
	for i in range(16):
		for j in range(8):
			pltImage = imageData[k][0]
			ax[i,j].imshow(pltImage, interpolation='nearest',cmap='gray_r')
			ax[i,j].axis('off')
			k = k+1
	f.set_size_inches(18.5, 10.5)
	f.savefig('images/'+imageName+'_after_'+str(epoch)+'_epoch.png', dpi = 100, bbox_inches='tight', pad_inches = 0)
	plt.close(f)
	return None

def initNormal(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value,name=name)

if __name__ == '__main__':
    batchSize = 128
    nbEpoch = 200
    decayIter = 100
    lr = 0.0002
    num_classes = 10
    X_train, X_noise = dataInit()
    X_train = X_train[:, np.newaxis, :, :]
    numExamples = (X_train.shape)[0]
    numBatches = int(numExamples/float(batchSize))
    
    print('Number of examples: ', numExamples)
    print('Number of Batches: ', numBatches)
    print('Number of epochs: ', nbEpoch)
    
    adam=Adam(lr=lr, beta_1=0.5 )
    
    print('Generator Model')
    
    generator = Sequential([
    # Layer 1
    Dense(784, input_shape=(100,)),
    Reshape(target_shape=(7, 7, 16)),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    # Layer 2
    Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'), 
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
    # Layer 3
    Conv2DTranspose(1, kernel_size=5, strides=2, padding='same'),
    Activation('tanh')
    ])
    generator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    print('Discriminator Model')
    discriminator = Sequential([
    # Conv 1: 14,14,32
    Conv2D(32, kernel_size=5, strides=2, padding='same', input_shape=(28,28,1)),
    LeakyReLU(alpha=0.01),
        
    # Conv 2: 7,7,16
    Conv2D(16, kernel_size=5, strides=2, padding='same'),   
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
        
    # FC 1
    Flatten(),
    Dense(784),
    BatchNormalization(),
    LeakyReLU(alpha=0.01),
       
    # Output
    Dense(1),
    Activation('sigmoid')
    ])
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    discriminator.trainable = False
    
    print('DCGAN model')
    
    dcganInput = Input(shape=(100,))
    x = generator(dcganInput)
    dcganOutput = discriminator(x)
    dcgan = Model(input=dcganInput, output=dcganOutput)
    dcgan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    discriminator.trainable = True
    
    if not os.path.exists('images'):
    	os.makedirs('images')
    if not os.path.exists('models'):
    	os.makedirs('models')
    if not os.path.exists('metrics'):
    	os.makedirs('metrics')
    dLoss = []
    gLoss = []
    
    for epoch in range(1, nbEpoch + 1):
        print('Epoch: ', epoch)
        for i in range(numBatches):
            noisePredictBatch = X_noise[np.random.randint(numExamples, size = batchSize)]
            noiseDataBatch = generator.predict(noisePredictBatch)
            origDataBatch = X_train[np.random.randint(numExamples, size = batchSize)]
            noiseLabelsBatch, origLabelsBatch = np.zeros(batchSize).astype(int), np.ones(batchSize).astype(int)
            trainBatch = np.concatenate((noiseDataBatch, origDataBatch), axis = 0)
            trainLabels = np.concatenate((noiseLabelsBatch, origLabelsBatch))
            trainBatch, trainLabels = shuffle(trainBatch, trainLabels)
            discriminatorLoss = discriminator.train_on_batch(trainBatch, trainLabels)
            dcganLabels = np.ones(batchSize).astype(int)			
            discriminator.trainable = False
            dcganLoss = dcgan.train_on_batch(noisePredictBatch, dcganLabels)
            discriminator.trainable = True
        dLoss.append(discriminatorLoss)
        gLoss.append(dcganLoss)
            
        if (epoch % 5 == 0) or (epoch == 1):
            saveImage(noiseDataBatch, 'generated', epoch)
            print('after epoch: ', epoch)
            print ('dcgan Loss: ', dcganLoss, '\t discriminator loss', discriminatorLoss)
            generator.save('models/generator_'+str(epoch)+'.h5')
            
        if epoch > decayIter :
            lrD = discriminator.optimizer.lr.get_value()
            lrG = generator.optimizer.lr.get_value()
            discriminator.optimizer.lr.set_value((lrD - lr/decayIter).astype(np.float32))
            generator.optimizer.lr.set_value((lrG - lr/decayIter).astype(np.float32))
            print('learning rate linearly decayed')
            
        np.save('metrics/dLoss.npy', np.array(dLoss))
        np.save('metrics/gLoss.npy', np.array(gLoss))
    print('Peace')