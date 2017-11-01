from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
        
class VAE(object):

  def __init__(self, network_architecture, batch_size, learn_rate, transfer_func=tf.nn.relu, train_multiple=False):
    self.__net_arch = network_architecture
    self.__lr = learn_rate
    self.__bs = batch_size
    
    self.__tran_func = transfer_func
    # Flag for whether cost function associated with training multiple VAE models
    self.__train_multiple = train_multiple
    
    # Graph input
    self.__x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    
    # Boolean tensor to signify whether input should be discriminated against
    self.__discr = tf.placeholder(tf.bool, [None])

    # Create VAE
    self.__create_autoencoder()
    
    # Compute loss terms for VAE
    self.__create_vae_loss()

    # Define loss function for the VAE
    self.__create_loss_optimiser()
		
  def __create_autoencoder(self):
    # Initialize autoencode network weights and biases
    network_weights = self.__init_weights(**self.__net_arch)

    # Use recognition network to determine mean and (log) variance of Gaussian distribution in latent space
    self.__z_mean, self.__z_log_sigma_sq = self.__encoder(network_weights['W_q'], network_weights['B_q'])

    # Randomly draw one sample z from the latent normal distribution - assumed to generate the data
    n_z = self.__net_arch['n_z']
    # Epsilon is a random normal tensor - represents noise
    eps = tf.random_normal((self.__bs, n_z), mean=0., stddev=1.0, dtype=tf.float32)
    # z = mu + sigma*epsilon
    self.__z = tf.add(self.__z_mean, tf.multiply(tf.sqrt(tf.exp(self.__z_log_sigma_sq)), eps))
    
    # Use generator to determine mean of Bernoulli distribution of reconstructed input
    self.__x_reconstr_logits, self.__x_reconstr_mean = self.__decoder(network_weights['W_p'], network_weights['B_p'])
        
  def __init_weights(self, n_input, n_hidden_1, n_hidden_2, n_z):
    all_weights = dict()
    w_init = tf.contrib.layers.xavier_initializer(uniform=False)
    b_init = tf.constant_initializer(0.)
    
    all_weights['W_q'] = {
        'h1': tf.Variable(w_init(shape=[n_input, n_hidden_1])),
        'h2': tf.Variable(w_init(shape=[n_hidden_1, n_hidden_2])),
        'out_mean': tf.Variable(w_init(shape=[n_hidden_2, n_z])),
        'out_log_sigma': tf.Variable(w_init(shape=[n_hidden_2, n_z]))}
       
    all_weights['B_q'] = {
        'b1': tf.Variable(b_init(shape=[n_hidden_1])),
        'b2': tf.Variable(b_init(shape=[n_hidden_2])),
        'out_mean': tf.Variable(b_init(shape=[n_z])),
        'out_log_sigma': tf.Variable(b_init(shape=[n_z]))}
        
    all_weights['W_p'] = {
        'h1': tf.Variable(w_init(shape=[n_z, n_hidden_2])),
        'h2': tf.Variable(w_init(shape=[n_hidden_2, n_hidden_1])),
        'out_mean': tf.Variable(w_init(shape=[n_hidden_1, n_input])),
        'out_log_sigma': tf.Variable(w_init(shape=[n_hidden_1, n_input]))}
        
    all_weights['B_p'] = {
        'b1': tf.Variable(b_init(shape=[n_hidden_2])),
        'b2': tf.Variable(b_init(shape=[n_hidden_1])),
        'out_mean': tf.Variable(b_init(shape=[n_input])),
        'out_log_sigma': tf.Variable(b_init(shape=[n_input]))}
        
    return all_weights
            
  # Generate probabilistic encoder (recognition network), which maps inputs onto a normal distribution in latent space                        
  # Encoder network turns the input samples x into two parameters in a latent space: z_mean & z_log_sigma_sq
  """ Q(z|X) """
  def __encoder(self, weights, biases):
    # The transformation is parametrized and can be learned
    h_layer_1 = self.__tran_func(tf.add(tf.matmul(self.__x, weights['h1']), biases['b1']))
    h_layer_2 = self.__tran_func(tf.add(tf.matmul(h_layer_1, weights['h2']), biases['b2']))
    
    z_mean = tf.add(tf.matmul(h_layer_2, weights['out_mean']), biases['out_mean'])
    z_log_sigma_sq = tf.add(tf.matmul(h_layer_2, weights['out_log_sigma']), biases['out_log_sigma'])

    return (z_mean, z_log_sigma_sq)

  # Generate probabilistic decoder (generator network), which maps points in latent space onto a Bernoulli distribution in data space
  # Decoder network maps the latent space points back to the original input data
  """ P(X|z) """
  def __decoder(self, weights, biases, reuse=False):
    # The transformation is parametrized and can be learned
    h_layer_1 = self.__tran_func(tf.add(tf.matmul(self.__z, weights['h1']), biases['b1']))
    h_layer_2 = self.__tran_func(tf.add(tf.matmul(h_layer_1, weights['h2']), biases['b2'])) 

    x_reconstr_logits = tf.add(tf.matmul(h_layer_2, weights['out_mean']), biases['out_mean'])
    x_reconstr_mean = tf.nn.sigmoid(x_reconstr_logits)

    return x_reconstr_logits, x_reconstr_mean

  # Define VAE Loss as sum of reconstruction term and KL Divergence regularisation term
  def __create_vae_loss(self):
    ''' 1.) The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli distribution 
            induced by the decoder in the data space). This can be interpreted as the number of "nats" required
            to reconstruct the input when the activation in latent space is given. Adding 1e-10 to avoid evaluation of log(0.0).
    '''
    # Prone to numerical instability
    # reconstr_loss = self.__x * tf.log(1e-10 + self.__x_reconstr_mean) + (1 - self.__x) * tf.log(1e-10 + 1 - self.__x_reconstr_mean)
    # reconstr_loss = -tf.reduce_sum(reconstr_loss, 1)    
    reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.__x_reconstr_logits, labels=self.__x)
    reconstr_loss = tf.reduce_sum(reconstr_loss, axis=1)
    self.__m_reconstr_loss = tf.reduce_mean(reconstr_loss)
    
    ''' 2.) The latent loss, which is defined as the KL divergence between the distribution in latent space induced 
            by the encoder on the data and some prior. Acts as a regulariser and can be interpreted as the number of "nats" 
            required for transmitting the latent space distribution given the prior.
            Fitting the variational objective is equivalent to optimising a lower bound on the log marginal likelihood,
            given that we know KL-divergence is non-negative --> Termed "ELBO" or "Evidence Lower Bound"
    '''      
    # Clip values of KL divergence to prevent NANs
    # latent_loss = 1 + tf.clip_by_value(self.__z_log_sigma_sq, -10., 10.) \
                    # - tf.square(tf.clip_by_value(self.__z_mean, -10., 10.)) \
                    # - tf.exp(tf.clip_by_value(self.__z_log_sigma_sq, -10., 10.))                      
    latent_loss = 1 + self.__z_log_sigma_sq - tf.square(self.__z_mean) - tf.exp(self.__z_log_sigma_sq)
    latent_loss = -0.5 * tf.reduce_sum(latent_loss, axis=1)    
    self.__m_latent_loss = tf.reduce_mean(latent_loss)
    
    if self.__train_multiple:    
        self.__cost = tf.where(self.__discr, latent_loss, (1./latent_loss))
    else:
        self.__cost = tf.add(reconstr_loss, latent_loss)
    
    # Average over batch
    self.__batch_cost = tf.reduce_mean(self.__cost)
        
  def __create_loss_optimiser(self):
    optimiser = tf.train.AdamOptimizer(learning_rate=self.__lr) 
    
    # Extract trainable variables and gradients   
    grads, tvars = zip(*optimiser.compute_gradients(self.__batch_cost))
    
    # Use gradient clipping to avoid 'exploding' gradients
    # clipped, _ = tf.clip_by_global_norm(grads, 1.)
    self.__train_op = optimiser.apply_gradients(zip(grads, tvars))

  # Train model on mini-batch of input data & return the cost
  def partial_fit(self, sess, X, discr=None):
    if discr is None:
      discr = [True] * self.__bs
    
    opt, cost, recon_loss, latent_loss = sess.run((self.__train_op, self.__batch_cost, self.__m_reconstr_loss, self.__m_latent_loss), 
                                                   feed_dict={self.__x: X, self.__discr: discr})
    
    return cost, recon_loss, latent_loss
    
  # Transform data by mapping it into the latent space
  # Note: This maps to mean of distribution, alternatively could sample from Gaussian distribution
  def transform(self, sess, X):
    return sess.run(self.__z_mean, feed_dict={self.__x: X})

  # Generate data by sampling from latent space
  # If z_mu is not None, data for this point in latent space is generated
  # Otherwise, z_mu is drawn from prior in latent space
  def generate(self, sess, z_mu=None):
    if z_mu is None:
      z_mu = np.random.normal(size=self.__net_arch['n_z'])

    return sess.run(self.__x_reconstr_mean, feed_dict={self.__z: z_mu})

  # Use VAE to reconstruct given data
  def reconstruct(self, sess, X):
    return sess.run(self.__x_reconstr_mean, feed_dict={self.__x: X})
