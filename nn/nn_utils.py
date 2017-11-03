from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def conv2d(x, W, stride=1, pad='SAME'):
  """conv2d returns a 2d convolution layer."""
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=pad)
    
def conv2d_transpose(x, W, output_channels, stride=1, filter_size=3, pad='SAME'):
  """conv2d_transpose returns a 2d deconvolution layer."""
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
    raise ValueError("Unknown padding")
  
  out_shape = [batch_size, output_size_h, output_size_w, output_channels]
  
  return tf.nn.conv2d_transpose(x, W, output_shape=out_shape, strides=[1, stride, stride, 1], padding=pad)

# NOTE: Discarding pooling layers (used to reduce dimension of representation) is
# increasingly more common, especially in generative models e.g. VAEs 
# --> Pooling likely to be abandoned at some point altogether
def max_pool_2x2(x, stride=2, filter_size=2, pad='SAME'):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding=pad)
  
def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  
  return tf.Variable(initial)

# ReLU neurons should be initialised with a slightly positive initial bias to avoid "dead neurons"
def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  
  return tf.Variable(initial)
