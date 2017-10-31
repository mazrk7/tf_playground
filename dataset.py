"""Functions for downloading and reading from different datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io
import numpy
import tensorflow as tf

import os
import config

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

from tensorflow.examples.tutorials.mnist import input_data

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
    numpy.random.seed(seed1 if seed is None else seed2)
    
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
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
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
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._features = self.features[perm]
        self._labels = self.labels[perm]
        
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      features_new_part = self._features[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((features_rest_part, features_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
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
  """Loads and reads the MNIST training, validation and test sets into a base.Datasets collection of DataSet."""
  
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
  """Loads and reads the ARTA experiment data for cognitive load into a base.Datasets collection of DataSet."""
  
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  # Import Matlab matrix variables, which consist of data & labels for the eye experiment
  dataset = scipy.io.loadmat(os.path.join(data_dir, config.WORKLOAD_DATA))
  labels = scipy.io.loadmat(os.path.join(data_dir, config.WORKLOAD_LABELS))

  train_data = dataset['train_data']
  test_data = dataset['test_data']

  train_labels = labels['train_labels']
  test_labels = labels['test_labels']
    
  if one_hot:
    train_labels = dense_to_one_hot(train_labels, num_classes=2)
    test_labels = dense_to_one_hot(test_labels, num_classes=2)
    
  if not 0 <= validation_size <= len(train_data):
    raise ValueError("Validation size should be between 0 and {}. Received: {}.".format(len(train_data), validation_size))
    
  validation_data = train_data[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_data = train_data[validation_size:]
  train_labels = train_labels[validation_size:]

  options = dict(dtype=dtype, seed=seed)
  
  train = DataSet(train_data, train_labels, **options)
  validation = DataSet(validation_data, validation_labels, **options)
  test = DataSet(test_data, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)
  
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes

  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

  return labels_one_hot
