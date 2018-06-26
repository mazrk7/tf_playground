# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import tensorflow as tf

from dataset import load_data
from nn import nn_utils

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

NUM_CLASSES = 10

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # -1 is to flatten x or to infer the shape
  # Second and third dimension are image width and height
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  # Compute 32 features for each 5x5 patch
  # First two dimensions of weight tensor are patch size
  # Next dimension is # of input channels
  # Last dimension is # of output channels
  # Bias vector with a component for each output channel
  W_conv1 = nn_utils.weight_variable([5, 5, 1, 32])
  b_conv1 = nn_utils.bias_variable([32])
  
  # Convolve the image with the weight tensor, add the bias and apply ReLU
  h_conv1 = tf.nn.relu(nn_utils.conv2d(x_image, W_conv1) + b_conv1)

  # Max pooling layer - downsamples by 2X i.e. reduces image size to 14x14
  h_pool1 = nn_utils.max_pool(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64 i.e. 64 features for each 5x5 patch
  W_conv2 = nn_utils.weight_variable([5, 5, 32, 64])
  b_conv2 = nn_utils.bias_variable([64])
  
  h_conv2 = tf.nn.relu(nn_utils.conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer - image size reduced to 7x7
  h_pool2 = nn_utils.max_pool(h_conv2)

  # Fully connected layer 1 -- after 2 rounds of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features for processing on entire image
  W_fc1 = nn_utils.weight_variable([7*7*64, 1024])
  b_fc1 = nn_utils.bias_variable([1024])

  # Reshape tensor from pooling layer into a batch of vectors
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  
  # Multiply by weight matrix, add bias and apply ReLU
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of features.
  # Create a placeholder for the probability that a neuron's output is kept during dropout
  # Turn dropout ON during training and OFF during testing
  keep_prob = tf.placeholder(tf.float32)
  # Function automatically handles scaling neuron outputs and masking them
  # Basically inverted dropout, outputs scaled input by 1/keep_prob, else outputs 0
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit - Just like layer of softmax regression
  W_fc2 = nn_utils.weight_variable([1024, NUM_CLASSES])
  b_fc2 = nn_utils.bias_variable([NUM_CLASSES])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  
  return y_conv, keep_prob

def main(_):
  data = load_data(FLAGS.dataset, one_hot=True, validation_size=10000)

  ####################################### DEFINE MODEL #######################################
  
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)
  
  # Placeholder to input the correct answers
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  ######################################### TRAINING #########################################
 
  #### DEFINE LOSS AND OPTIMISER ####
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
  
  # Replaced SGD optimiser for more sophisticated ADAM optimiser
  train_step = tf.train.AdamOptimizer(FLAGS.learn_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Use InteractiveSession when you wish to interleave operations for building a graph
  # with ones that actually run the graph -- useful in interactive contexts e.g. IPython
  # Otherwise build entire computation graph before starting a session and launch graph
  # --> This Session approach better separates process of creating the graph (model specification)
  # and the process of evaluating the graph (model fitting or training); basically cleaner
  # With block serves to automatically destroy session or release memory once block is exited
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #### TRAINING ####
    # Add keep_prob parameter to control the dropout rate
    for i in range(2000):
      # Get a 'batch' of `batch_size random data points from training set each loop iteration
      batch = data.train.next_batch(FLAGS.batch_size)
      
      # Logging every 100th iteration
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: FLAGS.keep_prob})

  ######################################### TESTING #########################################
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset', type=str, default='mnist', help='Name of dataset to load')
  parser.add_argument('--batch_size', type=int, default='128', help='Sets the batch size')
  parser.add_argument('--learn_rate', type=float, default='1e-4', help='Sets the learning rate')
  parser.add_argument('--keep_prob', type=float, default='.5', help='Sets the dropout rate')
  
  FLAGS, unparsed = parser.parse_known_args()
  
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
