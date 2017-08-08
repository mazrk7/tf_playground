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
        avg_recon = 0.
        avg_latent = 0.
        total_batch = int(NUM_SAMPLES / FLAGS.bs)
    
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(FLAGS.bs)

            # Fit training using batch data
            cost, recon, latent = model.partial_fit(sess, batch_xs)
            
            # Compute average losses
            avg_cost += (cost / NUM_SAMPLES) * FLAGS.bs
            avg_recon += (recon / NUM_SAMPLES) * FLAGS.bs
            avg_latent += (latent / NUM_SAMPLES) * FLAGS.bs
            
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: %04d / %04d, Cost= %04f, Recon= %04f, Latent= %04f" % \
				(epoch, n_epochs, avg_cost, avg_recon, avg_latent))
                  
    return model
  
# Define the network architecture
def network_architecture():
    network_architecture = \
	       {'n_input': IMAGE_PIXELS,    # MNIST data input
	        'n_hidden_1': 500,          # Dimensionality of hidden layer 1
	        'n_hidden_2': 500,          # Dimensionality of hidden layer 2
	        'n_z': FLAGS.latent_dim}    # Dimensionality of latent space

    return network_architecture
  
def main(name, seed):
    model_path = 'models/' + name
    
    # Define and instantiate VAE model
    vae = VAE(network_architecture=network_architecture()) 
    
    with tf.Session() as sess:
        tf.set_random_seed(seed)
            
        # Initialise tf variables
        init = tf.global_variables_initializer()
    
        # Launch session
        sess.run(init)
        
        vae_trained = train_model(vae, sess, n_epochs=FLAGS.n_epochs)
  
        # Create a saver object that will store all the parameter variables
        saver = tf.train.Saver()
        saver.save(sess, model_path)
        print("Model saved as: %s" % model_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default='digit_model_all', help='Name of model to train')
    parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    main(**arguments)   
