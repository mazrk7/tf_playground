from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from vae import VAE
from mnist_single_vae import network_architecture

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

NUM_SAMPLES = mnist.test.num_examples
NUM_CLASSES = 10

# Sample some test inputs and visualize how well the VAE can reconstruct these samples
def test_multiple_models(sess, model):    
    total_batch = int(NUM_SAMPLES / model.bs)
   
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.test.next_batch(model.bs)
        
        # Initialise a cost array for each model's reconstruction of a sample
        cost_array = np.zeros((model.bs, NUM_CLASSES), dtype=np.float32)
        for index in range(NUM_CLASSES):
            model_path = 'models/digit_model_' + str(index)
            
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

            recon_loss, latent_loss = model.get_vae_loss(sess, batch_xs)    
            cost_vec = recon_loss + latent_loss
                
            cost_array[:, index] = cost_vec
        
        min_cost_indices = tf.argmin(cost_array, axis=1)
        
        # Converts one hot representation to dense array of locations
        locations = tf.where(tf.equal(batch_ys, 1.0))
        # Strip first column
        correct_indices = locations[:, 1]
        
        correct_estimation = tf.equal(correct_indices, min_cost_indices)
        accuracy = tf.reduce_mean(tf.cast(correct_estimation, tf.float32))
        print("Batch: %d" % i)
        
    mean_accuracy = tf.reduce_mean(accuracy)
    print("Test accuracy of multiple-selection VAE models: %g" % sess.run(mean_accuracy))
      
def main(seed): 
    vae = VAE(network_architecture=network_architecture())
        
    with tf.Session() as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
                 
        test_multiple_models(sess, vae)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    main(**arguments)      
