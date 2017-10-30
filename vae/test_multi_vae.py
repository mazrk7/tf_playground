from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import load_data, DataSet
from vae import VAE
from vae import FLAGS
from train_multi_vae import network_architecture

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

NUM_CLASSES = 10

# Sample a test input and see how well the modular VAE can reconstruct these samples
def plot_single_model(sess, model, test_data):
    x_sample = test_data.next_batch(model.bs)[0]
    x_reconstruct = model.reconstruct(sess, x_sample)
    vae_loss = model.get_latent_loss(sess, x_sample)

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(IMAGE_SIZE, IMAGE_SIZE), vmin=0, vmax=1, cmap='gray')
        plt.title("Test input")
        plt.colorbar()
    
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(IMAGE_SIZE, IMAGE_SIZE), vmin=0, vmax=1, cmap='gray')
        plt.title("Reconstruction")
        plt.colorbar()
        
        print(vae_loss[i])

    plt.tight_layout()
    plt.show()
    
# Train a VAE with 2d latent space and illustrate how the encoder (the recognition network) 
# encodes some of the labeled inputs (collapsing the Gaussian distribution in latent space to its mean)
def visualise_latent_space(sess, model, test_data, batch_size=5000):
    x_sample, y_sample = test_data.next_batch(batch_size)
    z_mu = model.transform(sess, x_sample)

    plt.figure(figsize=(8, 6)) 
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.grid()
    plt.show()    
    
# Sample some test inputs and visualize how well the VAE can reconstruct these samples
def test_multiple_models(sess, model, test_data):  
    num_samples = test_data.num_examples  
    total_batch = int(num_samples / model.bs)
   
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = test_data.next_batch(model.bs)
        
        # Initialise a cost array for each model's reconstruction of a sample
        cost_array = np.zeros((model.bs, NUM_CLASSES), dtype=np.float32)
        for index in range(NUM_CLASSES):
            model_path = 'models/digit_model_' + str(index)
            
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

            cost_vec = model.get_latent_loss(sess, batch_xs)     
            cost_array[:, index] = cost_vec
        
        min_cost_indices = tf.argmin(cost_array, axis=1)
        
        # Converts one hot representation to dense array of locations
        locations = tf.where(tf.equal(batch_ys, 1.0))
        # Strip first column
        correct_indices = locations[:, 1]
        
        correct_estimation = tf.equal(correct_indices, min_cost_indices)
        accuracy = tf.reduce_mean(tf.cast(correct_estimation, tf.float32))
        print("Batch accuracy: %g" % sess.run(accuracy))
        print("Batch: %d" % i)
        
    mean_accuracy = tf.reduce_mean(accuracy)
    print("Test accuracy of multiple-selection VAE models: %g" % sess.run(mean_accuracy))
      
def main(seed): 
    vae = VAE(network_architecture=network_architecture())
    data = load_data(FLAGS.dataset, one_hot=True, validation_size=10000)
    
    with tf.Session() as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        
        test_multiple_models(sess, vae, data.test)
        
        ##### DEBUGGING ROUTINES ####
        #for index in range(NUM_CLASSES):
            #model_path = 'models/digit_model_' + str(index)
            
            #saver = tf.train.Saver()
            #saver.restore(sess, model_path) 
        
            #plot_single_model(sess, vae, data.test)
            #visualise_latent_space(sess, vae, data.test)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    main(**arguments)      
