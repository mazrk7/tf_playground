# Adapted from https://github.com/tensorflow/models/tree/master/research/fivo
# Chris J. Maddison*, Dieterich Lawson*, George Tucker*, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. 
# "Filtering Variational Objectives." NIPS 2017.

"""
Implementation of objectives for training stochastic latent variable models.

Contains implementations of the Importance Weighted Autoencoder objective (IWAE)
and the Filtering Variational objective (FIVO).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from fivo import utils
from fivo import smc


def iwae(model,
         observations,
         seq_lengths,
         num_samples=1,
         parallel_iterations=30,
         swap_memory=True):
  """Computes the IWAE lower bound on the log marginal probability.

  This method accepts a stochastic latent variable model and some observations
  and computes a stochastic lower bound on the log marginal probability of the
  observations. The IWAE estimator is defined by averaging multiple importance
  weights. For more details see "Importance Weighted Autoencoders" by Burda
  et al. https://arxiv.org/abs/1509.00519.

  When num_samples = 1, this bound becomes the evidence lower bound (ELBO).

  Args:
    model: A subclass of ELBOTrainableSequenceModel that implements one
      timestep of the model. See models/vrnn.py for an example.
    observations: The inputs to the model. A potentially nested list or tuple of
      Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
      have a rank at least two and have matching shapes in the first two
      dimensions, which represent time and the batch respectively. The model
      will be provided with the observations before computing the bound.
    seq_lengths: A [batch_size] Tensor of ints encoding the length of each
      sequence in the batch (sequences can be padded to a common length).
    num_samples: The number of samples to use.
    parallel_iterations: The number of parallel iterations to use for the
      internal while loop.
    swap_memory: Whether GPU-CPU memory swapping should be enabled for the
      internal while loop.

  Returns:
    log_p_hat: A Tensor of shape [batch_size] containing IWAE's estimate of the
      log marginal probability of the observations.
    log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
      containing the log weights at each timestep. Will not be valid for
      timesteps past the end of a sequence.
  """

  log_p_hat, log_weights, _, final_state = fivo(
      model,
      observations,
      seq_lengths,
      num_samples=num_samples,
      resampling_criterion=smc.never_resample_criterion,
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  return log_p_hat, log_weights, final_state


def fivo(model,
         observations,
         seq_lengths,
         num_samples=1,
         resampling_criterion=smc.ess_criterion,
         resampling_type='multinomial',
         relaxed_resampling_temperature=0.5,
         parallel_iterations=30,
         swap_memory=True,
         random_seed=None):
  """Computes the FIVO lower bound on the log marginal probability.

  This method accepts a stochastic latent variable model and some observations
  and computes a stochastic lower bound on the log marginal probability of the
  observations. The lower bound is defined by a particle filter's unbiased
  estimate of the marginal probability of the observations. For more details see
  "Filtering Variational Objectives" by Maddison et al.
  https://arxiv.org/abs/1705.09279.

  When the resampling criterion is "never resample", this bound becomes IWAE.

  Args:
    model: A subclass of ELBOTrainableSequenceModel that implements one
      timestep of the model. See models/vrnn.py for an example.
    observations: The inputs to the model. A potentially nested list or tuple of
      Tensors each of shape [max_seq_len, batch_size, ...]. The Tensors must
      have a rank at least two and have matching shapes in the first two
      dimensions, which represent time and the batch respectively. The model
      will be provided with the observations before computing the bound.
    seq_lengths: A [batch_size] Tensor of ints encoding the length of each
      sequence in the batch (sequences can be padded to a common length).
    num_samples: The number of particles to use in each particle filter.
    resampling_criterion: The resampling criterion to use for this particle
      filter. Must accept the number of samples, the current log weights,
      and the current timestep and return a boolean Tensor of shape [batch_size]
      indicating whether each particle filter should resample. See
      ess_criterion and related functions for examples. When
      resampling_criterion is never_resample_criterion, resampling_fn is ignored
      and never called.
    resampling_type: The type of resampling, one of "multinomial" or "relaxed".
    relaxed_resampling_temperature: A positive temperature only used for relaxed
      resampling.
    parallel_iterations: The number of parallel iterations to use for the
      internal while loop. Note that values greater than 1 can introduce
      non-determinism even when random_seed is provided.
    swap_memory: Whether GPU-CPU memory swapping should be enabled for the
      internal while loop.
    random_seed: The random seed to pass to the resampling operations in
      the particle filter. Mainly useful for testing.

  Returns:
    log_p_hat: A Tensor of shape [batch_size] containing FIVO's estimate of the
      log marginal probability of the observations.
    log_weights: A Tensor of shape [max_seq_len, batch_size, num_samples]
      containing the log weights at each timestep of the particle filter. Note
      that on timesteps when a resampling operation is performed the log weights
      are reset to 0. Will not be valid for timesteps past the end of a
      sequence.
    resampled: A Tensor of shape [max_seq_len, batch_size] indicating when the
      particle filters resampled. Will be 1.0 on timesteps when resampling
      occurred and 0.0 on timesteps when it did not.
  """

  # batch_size is the number of particle filters running in parallel.
  batch_size = tf.shape(seq_lengths)[0]

  # Each sequence in the batch will be the input data for a different
  # particle filter. The batch will be laid out as:
  #   particle 1 of particle filter 1
  #   particle 1 of particle filter 2
  #   ...
  #   particle 1 of particle filter batch_size
  #   particle 2 of particle filter 1
  #   ...
  #   particle num_samples of particle filter batch_size
  observations = utils.tile_tensors(observations, [1, num_samples])
  tiled_seq_lengths = tf.tile(seq_lengths, [num_samples])
  model.set_observations(observations, tiled_seq_lengths)

  if resampling_type == 'multinomial':
    resampling_fn = smc.multinomial_resampling
  elif resampling_type == 'relaxed':
    resampling_fn = functools.partial(
        smc.relaxed_resampling, temperature=relaxed_resampling_temperature)

  resampling_fn = functools.partial(resampling_fn, random_seed=random_seed)

  def transition_fn(prev_state, t):
    if prev_state is None:
      return model.zero_state(batch_size * num_samples, tf.float32)

    return model.propose_and_weight(prev_state, t)

  log_p_hat, log_weights, resampled, final_state, _ = smc.smc(
      transition_fn,
      seq_lengths,
      num_particles=num_samples,
      resampling_criterion=resampling_criterion,
      resampling_fn=resampling_fn,
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  return log_p_hat, log_weights, resampled, final_state