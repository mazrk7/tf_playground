from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import datasets

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from vae import VAE
from vae import FLAGS
from train_vae import network_architecture

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Sample a test input and see how well the VAE can reconstruct these samples
def test_model(sess, model, dataset):
    x_sample = dataset.next_batch(model.bs)[0]
    x_reconstruct = model.reconstruct(sess, x_sample)

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

    plt.tight_layout()
    plt.show()
    
# Train a VAE with 2d latent space and illustrate how the encoder (the recognition network) 
# encodes some of the labeled inputs (collapsing the Gaussian distribution in latent space to its mean)
def visualise_latent_space(sess, model, dataset, batch_size=5000):
    x_sample, y_sample = dataset.next_batch(batch_size)
    z_mu = model.transform(sess, x_sample)

    plt.figure(figsize=(8, 6)) 
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.grid()
    plt.show()

# Use the generator network to plot reconstrunctions at the relative positions in the latent space
def plot_reconstructions(sess, model):
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((IMAGE_SIZE*ny, IMAGE_SIZE*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]] * model.bs)
            x_mean = model.generate(sess, z_mu)
      
            canvas[(nx-i-1)*IMAGE_SIZE:(nx-i)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE] = x_mean[0].reshape(IMAGE_SIZE, IMAGE_SIZE)

    plt.figure(figsize=(8, 12))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin='upper', cmap='gray')
    plt.tight_layout()
    plt.show()
      
def main(name, seed): 
    model_path = 'models/' + name
    _, _, test = datasets.load_data(FLAGS.dataset)
    
    # Define and instantiate VAE model
    vae = VAE(network_architecture=network_architecture())
	
    with tf.Session() as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
		
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("Model restored from: %s" % model_path)
            
        test_model(sess, vae, test)
  
        if vae.net_arch['n_z'] == 2:
            visualise_latent_space(sess, vae, test)
            plot_reconstructions(sess, vae)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, default='digit_model_all', help='Name of model to train')
    parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
    
    args = parser.parse_args()
    arguments = args.__dict__
    
    main(**arguments)      
