from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import datasets
import tensorflow as tf

from vae import VAE
from vae import FLAGS

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Train the VAE using mini-batches
def train_model(sess, model, num_samples, dataset, n_epochs=100, display_step=5):
    # Training cycle
    for epoch in range(n_epochs):
        avg_cost = 0.
        avg_recon = 0.
        avg_latent = 0.
        total_batch = int(num_samples / FLAGS.bs)
    
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = dataset.next_batch(FLAGS.bs)

            # Fit training using batch data
            cost, recon, latent = model.partial_fit(sess, batch_xs)
            
            # Compute average losses
            avg_cost += (cost / num_samples) * FLAGS.bs
            avg_recon += (recon / num_samples) * FLAGS.bs
            avg_latent += (latent / num_samples) * FLAGS.bs
            
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
    
# Define the convolutional VAE network architecture
def conv_network_architecture():
  network_architecture = \
	       {'n_input': 1,                      # Number of input channels
	       	'n_conv_outer': 2,                 # Convolution kernel sizes for outer layers
	        'n_conv_inner': 3,                 # Convolution kernel sizes for inner layers
	        'n_filters_1': 128,                # Number of output convolution filters at layer 1
	        'n_filters_2': 64,                 # Number of output convolution filters at layer 2
	        'n_filters_3': 64,                 # Number of output convolution filters at layer 3
	        'n_filters_4': 32,                 # Number of output convolution filters at layer 4
	        'n_hidden': 256,                   # Dimensionality of hidden layer
	        'n_z': FLAGS.latent_dim}           # Dimensionality of latent space

  return network_architecture
    
def main(name, seed):
    model_path = 'models/' + name
    train, _, _ = datasets.load_data(FLAGS.dataset)
    
    # Define and instantiate VAE model
    vae = VAE(network_architecture=network_architecture()) 
    
    with tf.Session() as sess:
        tf.set_random_seed(seed)
            
        # Initialise tf variables
        init = tf.global_variables_initializer()
    
        # Launch session
        sess.run(init)
        
        vae_trained = train_model(sess, vae, train.num_examples, train, n_epochs=FLAGS.n_epochs)
  
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
