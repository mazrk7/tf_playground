from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from dataset import load_data, DataSet
from vae import VAE
from conv_vae import ConvVAE

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

# Define the VAE network architecture
def network_architecture(vae_type, latent_dim):
  if vae_type == 'conv':
    network_architecture = \
       {'n_input': 1,                      # Number of input channels
        'kernel_outer': 5,                 # Convolution kernel sizes for outer layers
        'kernel_inner': 3,                 # Convolution kernel sizes for inner layers
        'n_filters_1': 64,                 # Number of output convolution filters at layer 1
        'n_filters_2': 64,                 # Number of output convolution filters at layer 2
        'n_filters_3': 64,                 # Number of output convolution filters at layer 3
        'n_filters_4': 64,                 # Number of output convolution filters at layer 4
        'n_hidden': 500,                   # Dimensionality of intermediate layer
        'n_z': latent_dim}                 # Dimensionality of latent space
  else:
    network_architecture = \
       {'n_input': IMAGE_PIXELS,           # MNIST data input
        'n_hidden_1': 500,                 # Dimensionality of hidden layer 1
        'n_hidden_2': 500,                 # Dimensionality of hidden layer 2
        'n_z': latent_dim}                 # Dimensionality of latent space
        
  return network_architecture
  
def main(_):
  model_path = 'models/' + FLAGS.name + '_' + str(FLAGS.index)
  data = load_data(FLAGS.dataset, one_hot=True, validation_size=10000)
    
  # Define and instantiate VAE model
  if FLAGS.vae_type == 'vae':
    vae = VAE(network_architecture=network_architecture(FLAGS.vae_type, FLAGS.latent_dim), batch_size=FLAGS.batch_size, learn_rate=FLAGS.learn_rate, train_multiple=True) 
  elif FLAGS.vae_type == 'conv':
    vae = ConvVAE(network_architecture=network_architecture(FLAGS.vae_type, FLAGS.latent_dim), batch_size=FLAGS.batch_size, learn_rate=FLAGS.learn_rate, train_multiple=True) 
  else:
    raise ValueError("Autoencoder type should be either conv or vae. Received: {}.".format(FLAGS.vae_type))
    
  # Wish to allocate approximately gpu_memory_frac% of GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_frac)
  
  with tf.device('/gpu:%d' % FLAGS.gpu_device):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options))
    
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
        
    # Initialise tf variables
    init = tf.global_variables_initializer()
        
    # Launch session
    sess.run(init)
        
    split_images, split_labels = segment_dataset(data.train, FLAGS.index)
    segmented_set = DataSet(split_images, split_labels)
    
    num_samples = segmented_set.num_examples
    
    # Training cycle
    for epoch in range(FLAGS.n_epochs):
      avg_cost = 0.
      avg_recon = 0.
      avg_latent = 0.
      total_batch = int(num_samples / FLAGS.batch_size)
        
      # Loop over all batches
      for i in range(total_batch):
        batch_xs, batch_ys = segmented_set.next_batch(FLAGS.batch_size)
            
        # Converts one hot representation to dense array of labels
        labels = np.where(np.equal(batch_ys, 1.0))[1]

        # Truth values for discrimination in training
        discr = np.equal(FLAGS.index, labels)

        # Fit training using batch data
        if FLAGS.vae_type == 'conv':
          cost, recon, latent = vae.partial_fit(sess, batch_xs, FLAGS.keep_prob, discr)
        else:
          cost, recon, latent = vae.partial_fit(sess, batch_xs, discr)
        
        # Compute average loss
        avg_cost += (cost / num_samples) * FLAGS.batch_size
        avg_recon += (recon / num_samples) * FLAGS.batch_size
        avg_latent += (latent / num_samples) * FLAGS.batch_size

      # Display logs per epoch step
      if epoch % FLAGS.display_step == 0:
        print("Epoch: %04d / %04d, Cost= %04f, Recon= %04f, Latent= %04f" % \
			(epoch, FLAGS.n_epochs, avg_cost, avg_recon, avg_latent))
  
  # Create a saver object that will store all the parameter variables
  saver = tf.train.Saver()
  saver.save(sess, model_path)
  print("Model saved as: %s" % model_path)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
    
  parser.add_argument('--name', type=str, default='digit_model', help='Name of model to train')
  parser.add_argument('--index', type=int, default='0', help='Index of model class to learn over')
  parser.add_argument('--seed', type=int, default='0', help='Sets the random seed for both numpy and tf')
    
  parser.add_argument('--dataset', type=str, default='mnist', help='Name of dataset to load')
  parser.add_argument('--vae_type', type=str, default='vae', help='Either a standard VAE (vae) or a convolutational VAE (conv)')
  parser.add_argument('--batch_size', type=int, default='100', help='Sets the batch size')
  parser.add_argument('--learn_rate', type=float, default='1e-5', help='Sets the learning rate')
  parser.add_argument('--n_epochs', type=int, default='50', help='Number of training epochs')
  parser.add_argument('--latent_dim', type=int, default='2', help='Latent dimensionality of the VAE')
  
  parser.add_argument('--keep_prob', type=float, default='1.0', help='Sets the dropout rate')
  
  parser.add_argument('--gpu_device', type=int, default=0, help='Specifying which GPU device to use')
  parser.add_argument('--log_device_placement', type=bool, default=False, help='Logs the devices that operations and tensors are assigned to')
  parser.add_argument('--gpu_memory_frac', type=float, default=0.8, help='Specifying what fraction of your GPU memory to occupy')
  parser.add_argument('--display_step', type=int, default='5', help='Display step during training')
      
  FLAGS, unparsed = parser.parse_known_args()
  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
