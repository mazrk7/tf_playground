"""Functions for downloading and reading from different datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import config
import collections

import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import stats

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

from tensorflow.examples.tutorials.mnist import input_data

DataSets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

class DataSet(object):

  def __init__(self,
               features,
               labels,
               dtype=dtypes.float32,
               seed=None):
    """Construct a DataSet.
    `dtype` can be either `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`. Seed arg provides for convenient deterministic testing.
    """
    
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)
    
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError("Invalid feature dtype %r, expected uint8 or float32" % dtype)
      
    assert features.shape[0] == labels.shape[0], ("features.shape: %s labels.shape: %s" % (features.shape, labels.shape))
    self._num_examples = features.shape[0]
    
    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def features(self):
    return self._features

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._features = self.features[perm0]
      self._labels = self.labels[perm0]
      
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      features_rest_part = self._features[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._features = self.features[perm]
        self._labels = self.labels[perm]
        
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      features_new_part = self._features[start:end]
      labels_new_part = self._labels[start:end]

      return np.concatenate((features_rest_part, features_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch

      return self._features[start:end], self._labels[start:end]

def load_data(dataset, one_hot=False, validation_size=5000):
  # Load data according to type of dataset
  if dataset == 'mnist':
    reader = read_mnist
    data_dir =  config.MNIST_DIR
  elif dataset == 'workload':
    reader = read_workload
    data_dir = config.WORKLOAD_DIR
  else:
    raise ValueError("Invalid dataset %s" % dataset)
  
  global_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.GLOBAL_DATA_DIR)
  
  return reader(os.path.join(global_dir, data_dir), one_hot=one_hot, validation_size=validation_size)

def read_mnist(data_dir,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               validation_size=5000,
               seed=None):
  """Loads and reads the MNIST training, validation and test sets into a collection of DataSet."""
  
  # Note it is very memory inefficient to use one-hot coding
  # Instead use tf.nn.sparse_softmax_cross_entropy_with_logits() to specify a vector of integers as labels
  # Saves you from building a dense one-hot representation
  options = dict(one_hot=one_hot, dtype=dtype, reshape=reshape, validation_size=validation_size)
    
  return input_data.read_data_sets(data_dir, **options)
  
def read_workload(data_dir,
             one_hot=False,
             dtype=dtypes.float32,
             validation_size=10,
             seed=None):
  """Loads and reads the raw ARTA experiment data for cognitive load into a collection of DataSet."""
  
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  # Import Matlab matrix variables, which consist of data & labels for the eye experiment
  dataset = scipy.io.loadmat(os.path.join(data_dir, config.WORKLOAD_DATA))
  num_sources = dataset['num_sources'].item(0)  
  window = dataset['window'].item(0)  
  
  segments, labels = segment_signal(dataset, window, num_sources)
  
  if one_hot:
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    
  reshaped_segments = segments.reshape(len(segments), 1, window, num_sources)
  
  # Don't randomise the train-test set indices, instead rely on different users to provide testing data
  # train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
  train_test_split = int(.7 * len(reshaped_segments)) + 1
  
  # train_data = reshaped_segments[train_test_split]
  # train_labels = labels[train_test_split]
  
  train_data, test_data = np.split(reshaped_segments, [train_test_split])
  train_labels, test_labels = np.split(labels, [train_test_split])
  
  if not 0 <= validation_size <= len(train_data):
    raise ValueError("Validation size should be between 0 and {}. Received: {}.".format(len(train_data), validation_size))
    
  validation_data = train_data[:validation_size]
  validation_labels = train_labels[:validation_size]
  
  # test_data = reshaped_segments[~train_test_split]
  # test_labels = labels[~train_test_split]

  options = dict(dtype=dtype, seed=seed)
  
  train = DataSet(train_data, train_labels, **options)
  validation = DataSet(validation_data, validation_labels, **options)
  test = DataSet(test_data, test_labels, **options)
    
  return DataSets(train=train, validation=validation, test=test), window, num_sources

def windows(data, win):
    start = 0
    while start < data.size:
        yield int(start), int(start + win)
        start += (win / 2)

def segment_signal(data, window_size=400, num_sources=6):
    segments = np.empty((0, window_size, num_sources))
    labels = np.empty((0))
    
    for (start, end) in windows(data['timestamps'], window_size):     
        if(len(data['timestamps'][start:end]) == window_size):
            segments = np.vstack([segments, [data['raw'][start:end]]])
            labels = np.append(labels, stats.mode(data['labels'][start:end])[0][0])

    return segments, labels
