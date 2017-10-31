from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nn import nn_utils

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Downsample images to 14x14 pixel resolution.
DOWNSAMPLED = 14
DOWN_PIXELS = DOWNSAMPLED * DOWNSAMPLED

class ConvVAE(object):

  def __init__(self, network_architecture, batch_size, learn_rate, transfer_func=tf.nn.relu, train_multiple=False):
    self.__net_arch = network_architecture
    self.__lr = learn_rate
    self.__bs = batch_size
    
    self.__tran_func = transfer_func
    # Flag for whether cost function associated with training multiple VAE models
    self.__train_multiple = train_multiple
    
    # tf Graph input
    self.__x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    
    # Boolean tensor to signify whether input should be discriminated against
    self.__discr = tf.placeholder(tf.bool, [None])
        
    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    # Create a placeholder for the probability that a neuron's output is kept during dropout
    # Turn dropout ON during training and OFF during testing
    self.__kp = tf.placeholder(tf.float32)
    
    # Create ConvVAE network
    self.__create_autoencoder()
    
    # Compute loss terms for ConvVAE
    self.__create_vae_loss()

    # Define loss function for the ConvVAE
    self.__create_loss_optimiser()
	
  def __create_autoencoder(self):
    # Initialize autoencode network weights and biases
    network_weights = self.__init_weights(**self.__net_arch)

    # Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
    self.__z_mean, self.__z_log_sigma_sq = self.__encoder(network_weights['W_e'], network_weights['B_e'])

    # Randomly draw one sample z from the latent normal distribution - assumed to generate the data
    n_z = self.__net_arch['n_z']
    # Epsilon is a random normal tensor - represents noise
    eps = tf.random_normal((self.__bs, n_z), mean=0., stddev=1.0, dtype=tf.float32)
    # z = mu + sigma*epsilon
    self.__z = tf.add(self.__z_mean, tf.multiply(tf.sqrt(tf.exp(self.__z_log_sigma_sq)), eps))

    # Use generator to determine mean of Bernoulli distribution of reconstructed input
    self.__x_reconstr_logits, self.__x_reconstr_mean = self.__decoder(network_weights['W_d'], network_weights['B_d'])
    
  def __init_weights(self, n_input, n_conv_outer, n_conv_inner, n_filters_1, n_filters_2, n_filters_3, n_filters_4, n_hidden, n_z):
    all_weights = dict()
    
    all_weights['W_e'] = {
        'hconv1': nn_utils.weight_variable([n_conv_outer, n_conv_outer, n_input, n_filters_1]),
        'hconv2': nn_utils.weight_variable([n_conv_outer, n_conv_outer, n_filters_1, n_filters_2]),
        'hconv3': nn_utils.weight_variable([n_conv_inner, n_conv_inner, n_filters_2, n_filters_3]),
        'hconv4': nn_utils.weight_variable([n_conv_inner, n_conv_inner, n_filters_3, n_filters_4]),
        'hfc': nn_utils.weight_variable([n_filters_4*DOWN_PIXELS, n_hidden]),
        'out_mean': nn_utils.weight_variable([n_hidden, n_z]),
        'out_log_sigma': nn_utils.weight_variable([n_hidden, n_z])}
       
    all_weights['B_e'] = {
        'bconv1': nn_utils.bias_variable([n_filters_1]),
        'bconv2': nn_utils.bias_variable([n_filters_2]),
        'bconv3': nn_utils.bias_variable([n_filters_3]),
        'bconv4': nn_utils.bias_variable([n_filters_4]),
        'bfc': nn_utils.bias_variable([n_hidden]),
        'out_mean': nn_utils.bias_variable([n_z]),
        'out_log_sigma': nn_utils.bias_variable([n_z])}
        
    all_weights['W_d'] = {
        'hfc1': nn_utils.weight_variable([n_z, n_hidden]),
        'hfc2': nn_utils.weight_variable([n_hidden, n_filters_4*DOWN_PIXELS]),
        'hdconv1': nn_utils.weight_variable([n_conv_inner, n_conv_inner, n_filters_3,  n_filters_4]),
        'hdconv2': nn_utils.weight_variable([n_conv_inner, n_conv_inner, n_filters_2, n_filters_3]),
        'hdconv3': nn_utils.weight_variable([n_conv_outer, n_conv_outer, n_filters_1, n_filters_2]),
        'hconv': nn_utils.weight_variable([n_conv_outer, n_conv_outer, n_filters_1, n_input])}
        
    all_weights['B_d'] = {
        'bfc1': nn_utils.bias_variable([n_hidden]),
        'bfc2': nn_utils.bias_variable([n_filters_4*DOWN_PIXELS]),
        'bdconv1': nn_utils.bias_variable([n_filters_3]),
        'bdconv2': nn_utils.bias_variable([n_filters_2]),
        'bdconv3': nn_utils.bias_variable([n_filters_1]),
        'bconv': nn_utils.bias_variable([n_input])}
        
    return all_weights
            
  # Generate probabilistic encoder (recognition network), which maps inputs onto a normal distribution in latent space                        
  # Encoder network turns the input samples x into two parameters in a latent space: z_mean & z_log_sigma_sq
  def __encoder(self, weights, biases):
    x_image = tf.reshape(self.__x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    
    # The transformation is parametrized and can be learned
    conv1_layer = self.__tran_func(tf.add(nn_utils.conv2d(x_image, weights['hconv1']), biases['bconv1']))
    conv2_layer = self.__tran_func(tf.add(nn_utils.conv2d(conv1_layer, weights['hconv2'], stride=2), biases['bconv2']))
    conv3_layer = self.__tran_func(tf.add(nn_utils.conv2d(conv2_layer, weights['hconv3']), biases['bconv3']))
    conv4_layer = self.__tran_func(tf.add(nn_utils.conv2d(conv3_layer, weights['hconv4']), biases['bconv4']))
  
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
    dconv1_layer = self.__tran_func(tf.add(nn_utils.conv2d_transpose(h_fc2_layer, weights['hdconv1'], self.__net_arch['n_filters_3']), biases['bdconv1']))
    dconv2_layer = self.__tran_func(tf.add(nn_utils.conv2d_transpose(dconv1_layer, weights['hdconv2'], self.__net_arch['n_filters_2']), biases['bdconv2']))
    dconv3_layer = self.__tran_func(tf.add(nn_utils.conv2d_transpose(dconv2_layer, weights['hdconv3'], self.__net_arch['n_filters_1'], stride=2, 
                                                            filter_size=self.__net_arch['n_conv_outer'], pad='VALID'), biases['bdconv3']))
    
    x_reconstr_logits = tf.add(nn_utils.conv2d(dconv3_layer, weights['hconv'], pad='VALID'), biases['bconv'])       
    x_reconstr_logits = tf.reshape(x_reconstr_logits, [-1, IMAGE_PIXELS])                                              
    x_reconstr_mean_image = tf.nn.sigmoid(x_reconstr_logits) 
    x_reconstr_mean = tf.reshape(x_reconstr_mean_image, [-1, IMAGE_PIXELS])
    
    return x_reconstr_logits, x_reconstr_mean

  # Define VAE Loss as sum of reconstruction term and KL Divergence regularisation term
  def __create_vae_loss(self):
    # 1.) The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli distribution 
    #     induced by the decoder in the data space). This can be interpreted as the number of "nats" required
    #     to reconstruct the input when the activation in latent space is given. Adding 1e-8 to avoid evaluation of log(0.0).
    reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.__x_reconstr_logits, labels=self.__x)
    self.__reconstr_loss = tf.reduce_sum(reconstr_loss, axis=1)
    self.__m_reconstr_loss = tf.reduce_mean(self.__reconstr_loss)
    
    # 2.) The latent loss, which is defined as the KL divergence between the distribution in latent space induced 
    #     by the encoder on the data and some prior. Acts as a regulariser and can be interpreted as the number of "nats" 
    #     required for transmitting the latent space distribution given the prior.
    latent_loss = 1 + self.__z_log_sigma_sq - tf.square(self.__z_mean) - tf.exp(self.__z_log_sigma_sq)
    self.__latent_loss = -0.5 * tf.reduce_sum(latent_loss, axis=1)     
    self.__m_latent_loss = tf.reduce_mean(self.__latent_loss)
    
    if self.__train_multiple:    
        self.__cost = tf.where(self.__discr, self.__latent_loss, (1./self.__latent_loss))
    else:
        self.__cost = tf.add(self.__reconstr_loss, self.__latent_loss)
    
    # Average over batch
    self.__batch_cost = tf.reduce_mean(self.__cost)
    
  def __create_loss_optimiser(self):
    optimiser = tf.train.AdamOptimizer(learning_rate=self.__lr) 
    
    # Extract trainable variables and gradients   
    grads, tvars = zip(*optimiser.compute_gradients(self.__batch_cost))
    
    # Use gradient clipping to avoid 'exploding' gradients
    # clipped, _ = tf.clip_by_global_norm(grads, 1.)
    self.__train_op = optimiser.apply_gradients(zip(grads, tvars))
    
  # Train model on mini-batch of input data
  def partial_fit(self, sess, X, keep_prob=1.0, discr=None):
    if discr is None:
      discr = [True] * self.__bs
    
    opt, cost, recon_loss, latent_loss = sess.run((self.__train_op, self.__batch_cost, self.__m_reconstr_loss, self.__m_latent_loss), 
                                                   feed_dict={self.__x: X, self.__discr: discr, self.__kp: keep_prob})
    
    return cost, recon_loss, latent_loss
    
  # Return KL Divergence loss across batch sammples
  def get_latent_loss(self, sess, X, keep_prob=1.0):                  
    return sess.run((self.__latent_loss), feed_dict={self.__x: X, self.__kp: keep_prob})
    
  # Transform data by mapping it into the latent space
  # Note: This maps to mean of distribution, alternatively could sample from Gaussian distribution
  def transform(self, sess, X, keep_prob=1.0):
    return sess.run(self.__z_mean, feed_dict={self.__x: X, self.__kp: keep_prob})
    
  # Generate data by sampling from latent space
  # If z_mu is not None, data for this point in latent space is generated
  # Otherwise, z_mu is drawn from prior in latent space 
  def generate(self, sess, z_mu=None, keep_prob=1.0):
    if z_mu is None:
      z_mu = np.random.normal(size=self.__net_arch['n_z'])
    
    return sess.run(self.__x_reconstr_mean, feed_dict={self.__z: z_mu, self.__kp: keep_prob})
    
  # Use VAE to reconstruct given data
  def reconstruct(self, sess, X, keep_prob=1.0):
    return sess.run(self.__x_reconstr_mean, feed_dict={self.__x: X, self.__kp: keep_prob})
