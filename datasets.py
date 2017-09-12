"""Library of datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os

import scipy.io
import numpy as np

import tensorflow as tf
import config

from tensorflow.examples.tutorials.mnist import input_data

gfile = tf.gfile
        
def load_data(dataset):
    # Load data
    if dataset == 'mnist':
        reader = read_MNIST
    elif hparams.task == 'omni':
        reader = read_omniglot
    
    train, valid, test = reader()

    return train, valid, test

def read_MNIST():
    """ Reads in MNIST images.
        Returns:
            train: 50k training images and labels
            valid: 10k validation images and labels
            test: 10k test images and labels
    """
    # Note it is very memory inefficient to use one-hot coding
    # Instead use tf.nn.sparse_softmax_cross_entropy_with_logits() to specify a vector of integers as labels
    # Saves you from building a dense one-hot representation
    mnist = input_data.read_data_sets(config.DATA_DIR, validation_size=10000, one_hot=True)

    return mnist.train, mnist.validation, mnist.test

def read_omniglot():
    """ Reads in Omniglot images.
        Returns:
            x_train: training images
            x_valid: validation images
            x_test: test images
    """
    n_validation=1345

    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

    omni_raw = scipy.io.loadmat(os.path.join(config.DATA_DIR, config.OMNIGLOT))

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

    shuffle_seed = 123
    permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
    train_data = train_data[permutation]

    x_train = train_data[:-n_validation]
    x_valid = train_data[-n_validation:]
    x_test = test_data

    return x_train, x_valid, x_test
