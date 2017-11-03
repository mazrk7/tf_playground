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
from conv_vae import ConvVAE
from train_multi_vae import network_architecture

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

NUM_CLASSES = 10

# Sample a test input and see how well the modular VAE can reconstruct these samples
def plot_single_model(sess, model, test_data, batch_size):
  x_sample = test_data.next_batch(batch_size)[0]
  x_reconstruct = model.reconstruct(sess, x_sample)
  z_mu = model.transform(sess, x_sample)
  
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
    
    print(np.sqrt(np.sum(z_mu[i]**2)))

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
      
def main(_): 
  data = load_data(FLAGS.dataset, one_hot=True, validation_size=10000)

  # Define and instantiate VAE model
  if FLAGS.vae_type == 'vae':
    vae = VAE(network_architecture=network_architecture(FLAGS.vae_type, FLAGS.latent_dim), batch_size=FLAGS.batch_size, learn_rate=FLAGS.learn_rate, train_multiple=True) 
  elif FLAGS.vae_type == 'conv':
    vae = ConvVAE(network_architecture=network_architecture(FLAGS.vae_type, FLAGS.latent_dim), batch_size=FLAGS.batch_size, learn_rate=FLAGS.learn_rate, train_multiple=True) 
  else:
    raise ValueError("Autoencoder type should be either conv or vae. Received: {}.".format(FLAGS.vae_type))
    
  # Sample some test inputs and visualize how well the VAE can reconstruct these samples
  with tf.Session() as sess:
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
   
    # Loop over `batch_count` batches
    for i in range(FLAGS.batch_count):
      batch_xs, batch_ys = data.test.next_batch(FLAGS.batch_size)
        
      # Initialise a cost array for each model's reconstruction of a sample
      cost_array = np.zeros((FLAGS.batch_size, NUM_CLASSES), dtype=np.float32)
      for index in range(NUM_CLASSES):
        model_path = 'models/' + FLAGS.name + '_' + str(index)
            
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        z_mu = vae.transform(sess, batch_xs)
        cost_array[:, index] = np.sqrt(np.sum(z_mu**2, axis=1))

      min_cost_indices = tf.argmin(cost_array, axis=1)
        
      # Converts one hot representation to dense array of locations
      locations = tf.where(tf.equal(batch_ys, 1.0))
      # Strip first column
      correct_indices = locations[:, 1]
        
      correct_estimation = tf.equal(correct_indices, min_cost_indices)
      accuracy = tf.reduce_mean(tf.cast(correct_estimation, tf.float32))
      print("Batch %d accuracy: %g" % (i, sess.run(accuracy)))
        
    ##### DEBUGGING ROUTINES ####
    for index in range(NUM_CLASSES):
      model_path = 'models/' + FLAGS.name + '_' + str(index)
            
      saver = tf.train.Saver()
      saver.restore(sess, model_path) 
        
      plot_single_model(sess, vae, data.test, FLAGS.batch_size)
      if FLAGS.latent_dim == 2:
        visualise_latent_space(sess, vae, data.test)
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
    
  parser.add_argument('--name', type=str, default='digit_model', help='Name of model to train')
  parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
  parser.add_argument('--batch_count', type=int, default='5', help='Number of batches to test models over')
    
  parser.add_argument('--dataset', type=str, default='mnist', help='Name of dataset to load')
  parser.add_argument('--vae_type', type=str, default='vae', help='Either a standard VAE (vae) or a convolutational VAE (conv)')
  parser.add_argument('--batch_size', type=int, default='100', help='Sets the batch size')
  parser.add_argument('--learn_rate', type=float, default='1e-5', help='Sets the learning rate')
  parser.add_argument('--latent_dim', type=int, default='2', help='Latent dimensionality of the VAE')
      
  FLAGS, unparsed = parser.parse_known_args()
  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)    
