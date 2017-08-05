from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from vae import VAE
from vae import FLAGS

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

NUM_SAMPLES = mnist.train.num_examples

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Train the VAE using mini-batches
def train_model(model, sess, n_epochs=100, display_step=5):
    # Training cycle
    for epoch in range(n_epochs):
        avg_cost = 0.
        total_batch = int(NUM_SAMPLES / FLAGS.bs)
    
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(FLAGS.bs)

            # Fit training using batch data
            cost = model.partial_fit(sess, batch_xs)
      
            # Compute average loss
            avg_cost += (cost / NUM_SAMPLES) * FLAGS.bs

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                  
    return model
  
# Define the network architecture
def network_architecture():
    network_architecture = \
	       {'n_input': IMAGE_PIXELS,    # MNIST data input
	        'n_hidden_1': 500,          # Dimensionality of hidden layer 1
	        'n_hidden_2': 500,          # Dimensionality of hidden layer 2
	        'n_z': FLAGS.latent_dim}    # Dimensionality of latent space

    return network_architecture
  
if __name__ == '__main__':
    # Define and instantiate VAE model
    vae = VAE(network_architecture=network_architecture()) 
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Initialise tf variables
        init = tf.global_variables_initializer()
    
        # Launch session
        sess.run(init)
        
        model_path = 'models/digit_model_all'
        vae_trained = train_model(vae, sess, n_epochs=FLAGS.n_epochs)
  
        # Create a saver object that will store all the parameter variables
        saver.save(sess, model_path)
        print("Model saved as: %s" % model_path)
