"""
Author: Bhargav Patel and Pushpam Punjabi
Last Edited: 09-09-2018
Title: Basic implementation of GAN
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

#Sample Data
def get_y(x):
	return 10 + 3*x * x

def sample_data(n = 10000, scale = 100):
	data = []
	x = scale * (np.random.random_sample((n,))-0.5)
	for i in range(n):
		y = get_y(x[i])
		data.append([x[i], y])
	return np.array(data)
#Randomizing order
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size = [m, n])

#Defining Generator and Discriminator
def generator(Z, hsize = [16, 16], reuse = False):
	with tf.variable_scope("GAN/Generator", reuse = reuse):
		h1 = tf.layers.dense(Z, hsize[0], activation = tf.nn.leaky_relu)
		h2 = tf.layers.dense(h1, hsize[1], activation = tf.nn.leaky_relu)
		out = tf.layers.dense(h2, 2)
	return out

def discriminator(X, hsize = [16, 16], reuse = False):
	with tf.variable_scope("GAN/Discriminator", reuse = reuse):
		h1 = tf.layers.dense(X, hsize[0], activation = tf.nn.leaky_relu)
		h2 = tf.layers.dense(h1, hsize[1], activation = tf.nn.leaky_relu)
		h3 = tf.layers.dense(h2, 2)
		out = tf.layers.dense(h3, 1)
	return out, h3
#Normalization
X = tf.placeholder(tf.float32, [None, 2])
Z = tf.placeholder(tf.float32, [None, 2])

G_sample = generator(Z)
r_logits, r_rep = discriminator(X)
f_logits, g_rep = discriminator(G_sample, reuse = True)
#Loss funcrtion
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = r_logits, labels = tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits = f_logits, labels = tf.zeros_like(f_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = f_logits, labels = tf.ones_like(f_logits)))
#Plotting graph
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "GAN/Discriminator")
#Optimizing
gen_step = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(gen_loss, var_list = gen_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(disc_loss, var_list = disc_vars)
#Initializing tensorflow session
sess = tf.Session()
tf.global_variables_initializer().run(session = sess)
#Hyper-parameters
batch_size = 256
nd_steps = 10
ng_steps = 10

x_plot = sample_data(n = batch_size)

f = open('loss_logs.csv','w')
f.write('Iteration, Discriminator Loss, Generator Loss\n')
#Training the model
for i in range(50001):
	X_batch = sample_data(n = batch_size)
	Z_batch = sample_Z(batch_size, 2)
	for _ in range(nd_steps):
		_, dloss = sess.run([disc_step, disc_loss], feed_dict = {X: X_batch, Z: Z_batch})
	rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict = {X: X_batch, Z: Z_batch})
	
	for _ in range(ng_steps):
		_, gloss = sess.run([gen_step, gen_loss], feed_dict = {Z: Z_batch})
	rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict = {X: X_batch, Z: Z_batch})

	if i%10 == 0:
		f.write("%d,%f,%f\n"%(i,dloss,gloss))
	
	print("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))
	#Plot the generated data by generator
	if i%1000 == 0:
		plt.figure()
		g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
		xax = plt.scatter(x_plot[:,0], x_plot[:,1])
		gax = plt.scatter(g_plot[:,0],g_plot[:,1])

		plt.legend((xax,gax), ("Real Data","Generated Data"))
		plt.title('Samples at Iteration %d'%i)
		plt.tight_layout()
		plt.savefig('iterations/iteration_%d.png'%i)
		plt.close()
f.close()