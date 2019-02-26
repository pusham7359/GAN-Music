#Deep Convolutional GAN (DCGAN) with MNIST

import numpy as np
from scipy.io import loadmat
import keras
import keras.backend as K
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""## Loading MNIST Dataset"""

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

X_noise = [0]*X_train.shape[0]
npRandom = np.random.RandomState(18)
for i in range(X_train.shape[0]):
	randomNoise = npRandom.uniform(-1,1,100)
	X_noise =randomNoise
print('Random Noise Data: ', X_noise.shape)

plt.figure(figsize=(5, 4))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

"""## Preprocessing and Deprocessing
"""

def preprocess(x):    
    x = x.reshape(-1, 28, 28, 1) # 28,28,1
    x = np.float64(x)
    x = (x / 255 - 0.5) * 2
    x = np.clip(x, -1, 1)
    return x

def deprocess(x):
    x = (x / 2 + 1) * 255
    x = np.clip(x, 0, 255)
    x = np.uint8(x)
    x = x.reshape(28, 28)
    return x

X_train_real = preprocess(X_train)
X_test_real  = preprocess(X_test)

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

def show_results(losses):
    labels = ['Classifier', 'Discriminator', 'Generator']
    losses = np.array(losses)    
    
    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()


def show_images(generated_images):
    n_images = len(generated_images)
    rows = 4
    cols = n_images//rows
    
    plt.figure(figsize=(cols, rows))
    for i in range(n_images):
        img = deprocess(generated_images[i])
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


batchSize = 128
num_epoch = 200
numExamples = (X_train.shape)[0]
numBatches = int(numExamples/float(batchSize))
    
print('Number of examples: ', numExamples)
print('Number of Batches: ', numBatches)
print('Number of epochs: ', num_epoch)

# Generator
generator = Sequential([
        
        # generates images in (28,28,1)
        # FC 1: 7,7,16
        Dense(784, input_shape=(100,)),
        Reshape(target_shape=(7, 7, 16)),
        BatchNormalization(),
        LeakyReLU(alpha=0.02),
        
        # Conv 1: 14,14,32
        Conv2DTranspose(32, kernel_size=5, strides=2, padding='same'), 
        BatchNormalization(),
        LeakyReLU(alpha=0.02),
        
        # Conv 2: 28,28,1
        Conv2DTranspose(1, kernel_size=5, strides=2, padding='same'),
        Activation('tanh')
    ])

generator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
# Discriminator
discriminator = Sequential([  
        # classifies images in (28,28,1)    
        # Conv 1: 14,14,32
        Conv2D(32, kernel_size=5, strides=2, padding='same', input_shape=(28,28,1)),
        LeakyReLU(alpha=0.02),
        
        # Conv 2: 7,7,16
        Conv2D(16, kernel_size=5, strides=2, padding='same'),   
        BatchNormalization(),
        LeakyReLU(alpha=0.02),
        
        # FC 1
        Flatten(),
        Dense(784),
        BatchNormalization(),
        LeakyReLU(alpha=0.02),
        
        # Output
        Dense(1),
        Activation('sigmoid')        
    ])

discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gLoss=[]
dLoss=[]
losses = []
for e in range(num_epoch):
    for i in range(numBatches):
        
        # real MNIST digit images
        X_batch_real = X_train_real[i*batchSize:(i+1)*batchSize]
        
        y_train_real, y_train_fake = np.ones([batchSize, 1]), np.zeros([batchSize, 1])
        y_eval_real,  y_eval_fake  = np.ones([128, 1]), np.zeros([128, 1])
     
        # latent samples and the generated digit images
        X_batch_fake = generator.predict_on_batch(np.random.normal(loc=0, scale=1, size=(batchSize, 100)))
        discriminator.train_on_batch(X_batch_real, y_train_real)
        discriminator.train_on_batch(X_batch_fake, y_train_fake)
        # train the generator via GAN
        #make_trainable(discriminator, False)
        #discriminator.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta_1), loss='binary_crossentropy')
        gan.train_on_batch(np.random.normal(loc=0, scale=1, size=(batchSize, 100)), y_train_real)

    # evaluate
    X_eval_real = X_test_real[np.random.choice(len(X_test_real), 128, replace=False)]

    
    X_eval_fake = generator.predict_on_batch(np.random.normal(loc=0, scale=1, size=(batchSize, 100)))

    d_loss  = discriminator.test_on_batch(X_eval_real, y_eval_real)
    d_loss += discriminator.test_on_batch(X_eval_fake, y_eval_fake)
    g_loss  = gan.test_on_batch(np.random.normal(loc=0, scale=1, size=(batchSize, 100)), y_eval_real) # we want the fake to be realistic!

    losses.append((d_loss, g_loss))

    print("Epoch: ",e+1,"/",num_epoch,"\tDiscriminator Loss: ",np.around(d_loss,4),"\tGenerator Loss: ",np.around(g_loss,4))
