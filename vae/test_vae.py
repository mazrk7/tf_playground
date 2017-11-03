from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import load_data
from vae import VAE
from conv_vae import ConvVAE
from train_vae import network_architecture

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

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

# Use the generator network to plot reconstrunctions at the relative positions in the latent space
def plot_reconstructions(sess, model, batch_size):
  nx = ny = 20
  x_values = np.linspace(-3, 3, nx)
  y_values = np.linspace(-3, 3, ny)

  canvas = np.empty((IMAGE_SIZE*ny, IMAGE_SIZE*nx))
  for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
      z_mu = np.array([[xi, yi]] * batch_size)
      x_mean = model.generate(sess, z_mu)

      canvas[(nx-i-1)*IMAGE_SIZE:(nx-i)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE] = x_mean[0].reshape(IMAGE_SIZE, IMAGE_SIZE)

  plt.figure(figsize=(8, 12))        
  Xi, Yi = np.meshgrid(x_values, y_values)
  plt.imshow(canvas, origin='upper', cmap='gray')
  plt.tight_layout()
  plt.show()
      
def main(_): 
  model_path = 'models/' + FLAGS.name
  data = load_data(FLAGS.dataset, one_hot=True, validation_size=10000)

  # Define and instantiate VAE model
  if FLAGS.vae_type == 'vae':
    vae = VAE(network_architecture=network_architecture(FLAGS.vae_type, FLAGS.latent_dim), batch_size=FLAGS.batch_size, learn_rate=FLAGS.learn_rate) 
  elif FLAGS.vae_type == 'conv':
    vae = ConvVAE(network_architecture=network_architecture(FLAGS.vae_type, FLAGS.latent_dim), batch_size=FLAGS.batch_size, learn_rate=FLAGS.learn_rate) 
  else:
    raise ValueError("Autoencoder type should be either conv or vae. Received: {}.".format(FLAGS.vae_type))

  with tf.Session() as sess:
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
	
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print("Model restored from: %s" % model_path)
        
    # Sample a test input and see how well the VAE can reconstruct these samples
    x_sample = data.test.next_batch(FLAGS.batch_size)[0]
    x_reconstruct = vae.reconstruct(sess, x_sample)

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

    visualise_latent_space(sess, vae, data.test)
    
    if FLAGS.latent_dim == 2:
      plot_reconstructions(sess, vae, FLAGS.batch_size)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--name', type=str, default='digit_model_all', help='Name of model to train')
  parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
  
  parser.add_argument('--dataset', type=str, default='mnist', help='Name of dataset to load')
  parser.add_argument('--vae_type', type=str, default='vae', help='Either a standard VAE (vae) or a convolutational VAE (conv)')
  parser.add_argument('--batch_size', type=int, default='100', help='Sets the batch size')
  parser.add_argument('--learn_rate', type=float, default='1e-5', help='Sets the learning rate')
  parser.add_argument('--latent_dim', type=int, default='2', help='Latent dimensionality of the VAE')
    
  FLAGS, unparsed = parser.parse_known_args()
  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
