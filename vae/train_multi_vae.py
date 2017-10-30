from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from dataset import load_data, DataSet
from vae import VAE
from vae import FLAGS

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def segment_dataset(train_data, class_index):
    # Extract indices of relevant class
    class_indices = np.where(np.array(train_data.labels)[:, class_index] == 1.0)[0]
    discr_indices = np.where(np.array(train_data.labels)[:, class_index] != 1.0)[0]

    # Randomly shuffle discriminating classes to use in training
    np.random.shuffle(discr_indices)
    
    # Even split between single and discriminable model training data
    class_images = train_data.images[class_indices]
    class_labels = train_data.labels[class_indices]
    discr_images = train_data.images[discr_indices[0:len(class_indices)]]
    discr_labels = train_data.labels[discr_indices[0:len(class_indices)]]
    
    return np.concatenate((class_images, discr_images), axis=0), np.concatenate((class_labels, discr_labels), axis=0)
        
# Train the VAE using mini-batches
def train_model(sess, model, train_data, class_index, n_epochs=100, display_step=5):
    split_images, split_labels = segment_dataset(train_data, class_index)
    segmented_set = DataSet(split_images, split_labels)
    
    num_samples = segmented_set.num_examples
    
    # Training cycle
    for epoch in range(n_epochs):
        avg_cost = 0.
        avg_recon = 0.
        avg_latent = 0.
        total_batch = int(num_samples / model.bs)
        
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = segmented_set.next_batch(model.bs)
            
            # Converts one hot representation to dense array of labels
            labels = np.where(np.equal(batch_ys, 1.0))[1]

            # Truth values for discrimination in training
            discr = np.equal(class_index, labels)

            # Fit training using batch data
            cost, recon, latent = model.partial_fit(sess, batch_xs, discr)
            
            # Compute average loss
            avg_cost += (cost / num_samples) * model.bs
            avg_recon += (recon / num_samples) * model.bs
            avg_latent += (latent / num_samples) * model.bs

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: %04d / %04d, Cost= %04f, Recon= %04f, Latent= %04f" % \
				(epoch, n_epochs, avg_cost, avg_recon, avg_latent))
                  
    return model

# Define the network architecture
def network_architecture():
    network_architecture = \
	       {'n_input': IMAGE_PIXELS,     # MNIST data input
	        'n_hidden_1': 500,           # Dimensionality of hidden layer 1
	        'n_hidden_2': 500,           # Dimensionality of hidden layer 2
	        'n_z': FLAGS.latent_dim}     # Dimensionality of latent space

    return network_architecture
  
def main(name, index, seed):
    model_path = 'models/' + name + '_' + str(index)
    data = load_data(FLAGS.dataset, one_hot=True, validation_size=10000)
    
    # Define and instantiate VAE model
    vae = VAE(network_architecture=network_architecture(), train_multiple=True)
    
    with tf.Session() as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        # Initialise tf variables
        init = tf.global_variables_initializer()
        
        # Launch session
        sess.run(init)
        
        vae_trained = train_model(sess, vae, data.train, index, n_epochs=FLAGS.n_epochs)
  
        # Create a saver object that will store all the parameter variables
        saver = tf.train.Saver()
        saver.save(sess, model_path)
        print("Model saved as: %s" % model_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default='digit_model', help='Name of model to train')
    parser.add_argument('--index', type=int, default='0', help='Index of model class to learn over')
    parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    main(**arguments)
