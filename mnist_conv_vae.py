from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(0)
tf.set_random_seed(0)

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.flags

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("n_epochs", 50, "n_epochs")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_float("dropout", 0.5, "dropout")
flags.DEFINE_integer("latent_dim", 2, "latent_dim")

FLAGS = flags.FLAGS

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
NUM_SAMPLES = mnist.train.num_examples

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Downsample images to 14x14 pixel resolution.
DOWNSAMPLED = 14

class VAE(object):
  def __init__(self, network_architecture, transfer_func=tf.nn.relu, learning_rate=0.001, batch_size=100):
    self.__net_arch = network_architecture
    self.__tran_func = transfer_func
    self.lr = learning_rate
    self.bs = batch_size
    
    # tf Graph input
    self.__x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    
    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    # Create a placeholder for the probability that a neuron's output is kept during dropout
    # Turn dropout ON during training and OFF during testing
    self.__kp = tf.placeholder(tf.float32)
    
    # Create VAE network
    self.__create_autoencoder()
    
    # Define loss function for the VAE
    self.__create_loss_optimiser()
    
    # Initialise tf variables
    init = tf.global_variables_initializer()
    
    # Launch session
    self.__sess = tf.InteractiveSession()
    self.__sess.run(init)
	
  def __create_autoencoder(self):
    # Initialize autoencode network weights and biases
    network_weights = self.__init_weights(**self.__net_arch)

    # Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
    self.__z_mean, self.__z_log_sigma_sq = self.__encoder(network_weights['W_e'], network_weights['B_e'])

    # Randomly draw one sample z from the latent normal distribution - assumed to generate the data
    n_z = self.__net_arch['n_zdim']
    # Epsilon is a random normal tensor - represents noise
    eps = tf.random_normal((self.bs, n_z), mean=0., stddev=1.0, dtype=tf.float32)
    # z = mu + sigma*epsilon
    self.__z = tf.add(self.__z_mean, tf.multiply(tf.sqrt(tf.exp(self.__z_log_sigma_sq)), eps))

    # Use generator to determine mean of Bernoulli distribution of reconstructed input
    self.__x_reconstr_mean = self.__decoder(network_weights['W_d'], network_weights['B_d'])
    
  def __init_weights(self, n_input, n_conv_inner, n_conv_outer, n_filters_1, n_filters_2, n_filters_3, n_filters_4, n_hdim, n_zdim):
        all_weights = dict()
        
        all_weights['W_e'] = {
            'hconv1': weight_variable([n_conv_outer, n_conv_outer, n_input, n_filters_1]),
            'hconv2': weight_variable([n_conv_outer, n_conv_outer, n_filters_1, n_filters_2]),
            'hconv3': weight_variable([n_conv_inner, n_conv_inner, n_filters_2, n_filters_3]),
            'hconv4': weight_variable([n_conv_inner, n_conv_inner, n_filters_3, n_filters_4]),
            'hfc': weight_variable([n_filters_4*DOWNSAMPLED*DOWNSAMPLED, n_hdim]),
            'out_mean': weight_variable([n_hdim, n_zdim]),
            'out_log_sigma': weight_variable([n_hdim, n_zdim])}
           
        all_weights['B_e'] = {
            'bconv1': bias_variable([n_filters_1]),
            'bconv2': bias_variable([n_filters_2]),
            'bconv3': bias_variable([n_filters_3]),
            'bconv4': bias_variable([n_filters_4]),
            'bfc': bias_variable([n_hdim]),
            'out_mean': bias_variable([n_zdim]),
            'out_log_sigma': bias_variable([n_zdim])}
            
        all_weights['W_d'] = {
            'hfc1': weight_variable([n_zdim, n_hdim]),
            'hfc2': weight_variable([n_hdim, n_filters_4*DOWNSAMPLED*DOWNSAMPLED]),
            'hdconv1': weight_variable([n_conv_inner, n_conv_inner, n_filters_3,  n_filters_4]),
            'hdconv2': weight_variable([n_conv_inner, n_conv_inner, n_filters_2, n_filters_3]),
            'hdconv3': weight_variable([n_conv_outer, n_conv_outer, n_filters_1, n_filters_2]),
            'hconv': weight_variable([n_conv_outer, n_conv_outer, n_filters_1, n_input])}
            
        all_weights['B_d'] = {
            'bfc1': bias_variable([n_hdim]),
            'bfc2': bias_variable([n_filters_4*DOWNSAMPLED*DOWNSAMPLED]),
            'bdconv1': bias_variable([n_filters_3]),
            'bdconv2': bias_variable([n_filters_2]),
            'bdconv3': bias_variable([n_filters_1]),
            'bconv': bias_variable([n_input])}
            
        return all_weights
            
  # Generate probabilistic encoder (recognition network), which maps inputs onto a normal distribution in latent space                        
  # Encoder network turns the input samples x into two parameters in a latent space: z_mean & z_log_sigma_sq
  def __encoder(self, weights, biases):
    x_image = tf.reshape(self.__x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    
    # The transformation is parametrized and can be learned
    conv1_layer = self.__tran_func(tf.add(conv2d(x_image, weights['hconv1']), biases['bconv1']))
    conv2_layer = self.__tran_func(tf.add(conv2d(conv1_layer, weights['hconv2'], stride=2), biases['bconv2']))
    conv3_layer = self.__tran_func(tf.add(conv2d(conv2_layer, weights['hconv3']), biases['bconv3']))
    conv4_layer = self.__tran_func(tf.add(conv2d(conv3_layer, weights['hconv4']), biases['bconv4']))
  
    # Reshape tensor from last layer into a batch of vectors
    h_shape = conv4_layer.get_shape().as_list()
    dim = np.prod(h_shape[1:])  
    flat = tf.reshape(conv4_layer, [-1, dim])
  
    # Multiply by weight matrix, add bias and apply activation function
    h_fc_layer = self.__tran_func(tf.add(tf.matmul(flat, weights['hfc']), biases['bfc']))
    
    # Function automatically handles scaling neuron outputs and masking them
    # Basically inverted dropout, outputs scaled input by 1/keep_prob, else outputs 0    
    h_fc_drop = tf.nn.dropout(h_fc_layer, self.__kp)
    
    z_mean = tf.add(tf.matmul(h_fc_layer, weights['out_mean']), biases['out_mean'])
    z_log_sigma_sq = tf.add(tf.matmul(h_fc_layer, weights['out_log_sigma']), biases['out_log_sigma'])
    
    return (z_mean, z_log_sigma_sq)

  # Generate probabilistic decoder (generator network), which maps points in latent space onto a Bernoulli distribution in data space
  # Decoder network maps the latent space points back to the original input data
  def __decoder(self, weights, biases):
    h_fc1_layer = self.__tran_func(tf.add(tf.matmul(self.__z, weights['hfc1']), biases['bfc1']))
    h_fc2_layer = self.__tran_func(tf.add(tf.matmul(h_fc1_layer, weights['hfc2']), biases['bfc2']))
    
    h_fc2_layer = tf.reshape(h_fc2_layer, [-1, DOWNSAMPLED, DOWNSAMPLED, self.__net_arch['n_filters_4']])

    # The transformation is parametrized and can be learned
    dconv1_layer = self.__tran_func(tf.add(conv2d_transpose(h_fc2_layer, weights['hdconv1'], self.__net_arch['n_filters_3']), biases['bdconv1']))
    dconv2_layer = self.__tran_func(tf.add(conv2d_transpose(dconv1_layer, weights['hdconv2'], self.__net_arch['n_filters_2']), biases['bdconv2']))
    dconv3_layer = self.__tran_func(tf.add(conv2d_transpose(dconv2_layer, weights['hdconv3'], self.__net_arch['n_filters_1'], stride=2, 
                                                            filter_size=self.__net_arch['n_conv_outer'], pad='VALID'), biases['bdconv3']))
    x_reconstr_mean_image = tf.nn.sigmoid(tf.add(conv2d(dconv3_layer, weights['hconv'], pad='VALID'), biases['bconv'])) 
    x_reconstr_mean = tf.reshape(x_reconstr_mean_image, [-1, IMAGE_PIXELS])
    
    return x_reconstr_mean

  # Define VAE Loss as sum of reconstruction term and KL Divergence regularisation term
  def __create_loss_optimiser(self, epsilon=1e-8):
    # 1.) The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli distribution 
    #     induced by the decoder in the data space). This can be interpreted as the number of "nats" required
    #     to reconstruct the input when the activation in latent space is given. Adding 1e-8 to avoid evaluation of log(0.0).
    reconstr_loss = self.__x * tf.log(epsilon + self.__x_reconstr_mean) + (1 - self.__x) * tf.log(epsilon + 1 - self.__x_reconstr_mean)
    reconstr_loss = -tf.reduce_sum(reconstr_loss, 1)
    
    # 2.) The latent loss, which is defined as the KL divergence between the distribution in latent space induced 
    #     by the encoder on the data and some prior. Acts as a regulariser and can be interpreted as the number of "nats" 
    #     required for transmitting the latent space distribution given the prior.
    latent_loss = 1 + self.__z_log_sigma_sq - tf.square(self.__z_mean) - tf.exp(self.__z_log_sigma_sq)
    latent_loss = -0.5 * tf.reduce_sum(latent_loss, 1)
    
    # Average over batch
    self.__cost = tf.reduce_mean(reconstr_loss + latent_loss)
    
    # Use ADAM optimizer
    self.__optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.__cost)
  
  # Train model on mini-batch of input data
  # Return cost of mini-batch
  def partial_fit(self, X, dropout):
    opt, cost = self.__sess.run((self.__optimizer, self.__cost), feed_dict={self.__x: X, self.__kp: dropout})
    
    return cost
  
  # Transform data by mapping it into the latent space
  # Note: This maps to mean of distribution, alternatively could sample from Gaussian distribution
  def transform(self, X):
    return self.__sess.run(self.__z_mean, feed_dict={self.__x: X})
    
  # Generate data by sampling from latent space
  # If z_mu is not None, data for this point in latent space is generated
  # Otherwise, z_mu is drawn from prior in latent space 
  def generate(self, z_mu=None):
    if z_mu is None:
      z_mu = np.random.normal(size=self.__net_arch['n_z'])
    
    return self.__sess.run(self.__x_reconstr_mean, feed_dict={self.__z: z_mu})
    
  # Use VAE to reconstruct given data
  def reconstruct(self, X):
    return self.__sess.run(self.__x_reconstr_mean, feed_dict={self.__x: X})

################################################# GLOBAL FUNCTIONS #################################################

def conv2d(x, W, stride=1, pad='SAME'):
  """conv2d returns a 2d convolution layer."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=pad)
    
# Stride two transposed convolutions and zero padded so that output is same size as input
def conv2d_transpose(x, W, output_channels, stride=1, filter_size=3, pad='SAME'):
  """conv2d_transpose returns a 2d deconvolution layer with two strides."""
  input_shape = x.get_shape().as_list()
  
  batch_size = input_shape[0]
  input_size_h = input_shape[1]
  input_size_w = input_shape[2]

  if pad == 'SAME':
    output_size_h = input_size_h * stride
    output_size_w = input_size_w * stride
  elif pad == 'VALID':
    output_size_h = (input_size_h*stride) + filter_size - 1
    output_size_w = (input_size_w*stride) + filter_size - 1
  else:
    raise ValueError("unknown padding")
  
  out_shape = [batch_size, output_size_h, output_size_w, output_channels]
  
  return tf.nn.conv2d_transpose(x, W, output_shape=out_shape, strides=[1, stride, stride, 1], padding=pad)
	
def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  
  return tf.Variable(initial)

# ReLU neurons should be initialised with a slightly positive initial bias to avoid "dead neurons"
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  
  return tf.Variable(initial)
    
# Train the VAE using mini-batches
def train_model(model, dropout=0.5, learning_rate=0.001, batch_size=100, n_epochs=50, display_step=5):
  # Training cycle
  for epoch in range(n_epochs):
    avg_cost = 0.
    total_batch = int(NUM_SAMPLES / batch_size)
    
    # Loop over all batches
    for i in range(total_batch):
      batch_xs, _ = mnist.train.next_batch(batch_size)

      # Fit training using batch data
      cost = vae.partial_fit(batch_xs, dropout)
      
      # Compute average loss
      avg_cost += cost / NUM_SAMPLES * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                  
  return vae

# Sample some test inputs and visualize how well the VAE can reconstruct these samples
def test_model(model, batch_size=100):
  x_sample = mnist.test.next_batch(batch_size)[0]
  x_reconstruct = model.reconstruct(x_sample)

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
def visualise_latent_space(model, batch_size=1000):
  x_sample, y_sample = mnist.test.next_batch(batch_size)
  z_mu = model.transform(x_sample)

  plt.figure(figsize=(8, 6)) 
  plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
  plt.colorbar()
  plt.grid()
  plt.show()

# Use the generator network to plot reconstrunctions at the relative positions in the latent space
def plot_reconstructions(model):
  nx = ny = 20
  x_values = np.linspace(-3, 3, nx)
  y_values = np.linspace(-3, 3, ny)

  canvas = np.empty((IMAGE_SIZE*ny, IMAGE_SIZE*nx))
  for i, yi in enumerate(x_values):
    for j, xi in enumerate(y_values):
      z_mu = np.array([[xi, yi]]*model.bs)
      x_mean = model.generate(z_mu)
      
      canvas[(nx-i-1)*IMAGE_SIZE:(nx-i)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE] = x_mean[0].reshape(IMAGE_SIZE, IMAGE_SIZE)

  plt.figure(figsize=(8, 12))        
  Xi, Yi = np.meshgrid(x_values, y_values)
  plt.imshow(canvas, origin='upper', cmap='gray')
  plt.tight_layout()
  plt.show()
  
# Define the network architecture
def network_architecture():
  network_architecture = \
	       {'n_input': 1,                      # Number of input channels
	       	'n_conv_outer': 2,                 # Convolution kernel sizes for outer layers
	        'n_conv_inner': 3,                 # Convolution kernel sizes for inner layers
	        'n_filters_1': 128,                # Number of output convolution filters at layer 1
	        'n_filters_2': 64,                 # Number of output convolution filters at layer 2
	        'n_filters_3': 64,                 # Number of output convolution filters at layer 3
	        'n_filters_4': 32,                 # Number of output convolution filters at layer 4
	        'n_hdim': 256,                     # Dimensionality of hidden layer
	        'n_zdim': FLAGS.latent_dim}        # Dimensionality of latent space

  return network_architecture
  
if __name__ == '__main__':
  # Define and instantiate VAE model
  vae = VAE(network_architecture=network_architecture(), learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size)
  
  vae = train_model(vae, 
                    dropout=FLAGS.learning_rate,
                    learning_rate=FLAGS.learning_rate, 
                    batch_size=FLAGS.batch_size,
                    n_epochs=FLAGS.n_epochs)
  
  test_model(vae, batch_size=FLAGS.batch_size)
  
  if FLAGS.latent_dim == 2:
    visualise_latent_space(vae)
    plot_reconstructions(vae)
