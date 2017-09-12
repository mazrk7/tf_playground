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

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import datasets

flags = tf.flags

flags.DEFINE_string("dataset", "mnist", "Name of dataset to load")
flags.DEFINE_integer("bs", 100, "batch size")
flags.DEFINE_float("lr", .5, "learning rate")

FLAGS = flags.FLAGS

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

NUM_CLASSES = 10

def main():
  # Import data
  train, _, test = datasets.load_data(FLAGS.dataset)

  ####################################### DEFINE MODEL #######################################
  
  # x is a value that we'll input when TensorFlow is asked to run a computation
  # None means the dimension can be of any length (any number of MNIST images)
  x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
  
  # Variables are modifiable tensors that live in the graph of interacting operations
  # Typical to use this type for model parameters - initially set to 0 as they'll be learnt
  W = tf.Variable(tf.zeros([IMAGE_PIXELS, NUM_CLASSES]))
  b = tf.Variable(tf.zeros([NUM_CLASSES]))
  
  # First multiply x by W and then add bias before applying the softmax layer of a NN
  # y = tf.nn.softmax(tf.matmul(x, W) + b)
  
  # Remove softmax layer due to later instability of raw cross-entropy formulation
  y = tf.matmul(x, W) + b

  # Placeholder to input the correct answers
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
  
  ######################################### TRAINING #########################################

  #### DEFINE LOSS AND OPTIMISER ####
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  # Internally computes the softmax activation
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
  #### APPLY OPTIMISATION ####
  # In one line, compute gradients, compute parameter update steps and apply update steps to parameters
  train_step = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cross_entropy)

  # Launch model in an interactive session
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  #### TRAINING FOR 1000 STEPS ####
  # Stochastic GD as it's less expensive than using all available data for every training step
  for _ in range(1000):
    # Get a 'batch' of `batch_size` random data points from training set each loop iteration
    batch_xs, batch_ys = train.next_batch(FLAGS.bs)
    # Run train_step, feeding in the batch data to replace placeholders
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  ######################################### TESTING #########################################
  
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: test.images, y_: test.labels}))

if __name__ == '__main__':
  main()
  