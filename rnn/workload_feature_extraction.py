from __future__ import print_function

import tensorflow as tf
import numpy as np

from dataset import load_data
from nn import nn_utils

# Training Parameters
learning_rate = 0.0001
num_epochs = 50
batch_size = 20

# Read workload dataset, sequence length and input dimensionality from the workload Matlab matrix
data, window, num_input = load_data('workload', one_hot=True, validation_size=10)

# Storing the CNN model containing combined features
model_path = 'models/cnn_workload_features'
  
# CNN Parameters
kernel_size_1 = 40                          # First convolutional layer's filter size
depth_1 = 40                                # First convolutional layer's number of output channels i.e. depth
pool_kernel = 20                            # Kernel size of max pooling layer
kernel_size_2 = 4                           # Second convolutional layer's filter size
depth_2 = 4                                 # Second convolutional layer's number of output channels i.e. depth
num_hidden = 600                            # Number of hidden neurons in the fully connected layer
drop_rate = 0.5                             # Dropout rate

input_height = 1                            # Dealing with 1D signals
input_width = window                        # Window of timesteps to traverse the signals
num_labels = 2                              # Total number of classes --> Divided into binary classifier of beneficial or detrimental cognitive load
num_channels = num_input                    # Number of input signal channels
downsample_dim = 91                         # Downsample dimensions after max pooling

X = tf.placeholder(tf.float32, shape=[None, input_height, input_width, num_channels])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])
keep_prob = tf.placeholder(tf.float32)

# Define weights & biases
weights = {
    'depth_conv1': nn_utils.weight_variable([1, kernel_size_1, num_channels, depth_1], name='conv1_weights'),
    'depth_conv2': nn_utils.weight_variable([1, kernel_size_2, depth_1*num_channels, depth_2], name='conv2_weights'),
    'dense': tf.Variable(tf.random_normal([downsample_dim*depth_1*depth_2*num_channels, num_hidden]), name='dense_weights'),
    'out': tf.Variable(tf.random_normal([num_hidden, num_labels]))
}
biases = {
    'depth_conv1': nn_utils.bias_variable([depth_1*num_channels], name='conv1_biases'),
    'depth_conv2': nn_utils.bias_variable([depth_1*depth_2*num_channels], name='conv2_biases'),
    'dense': tf.Variable(tf.random_normal([num_hidden]), name='dense_biases'),
    'out': tf.Variable(tf.random_normal([num_labels]))
}

def CNN(x, weights, biases, keep_prob):
    # Convolve the input channels independently with the weight tensor, add the bias and apply ReLU
    conv1 = tf.nn.relu(tf.add(nn_utils.depthwise_conv2d(x, weights['depth_conv1']), biases['depth_conv1']))
    
    # Max pooling layer - downsamples the convolutional layer
    # Stride and pad 'VALID' --> ceil[(in_dim - filter_size + 1)/stride] --> (`window` - `pool_kernel` + 1)/2 = 91
    pool1 = nn_utils.max_pool(conv1, stride=[1,2], filter_size=[1,pool_kernel], pad='VALID')
    
    # Second convolutional layer, followed by a flattening of the tensor
    conv2 = tf.nn.relu(tf.add(nn_utils.depthwise_conv2d(pool1, weights['depth_conv2']), biases['depth_conv2']))
    conv2_shape = conv2.get_shape().as_list()
    conv2_flat = tf.reshape(conv2, [-1, conv2_shape[1] * conv2_shape[2] * conv2_shape[3]])
    
    # Fully connected layer to map the feature maps to `num_hidden` features across the input channels
    dense = tf.nn.tanh(tf.add(tf.matmul(conv2_flat, weights['dense']), biases['dense']))
    
    # Dropout - controls the complexity of the model, prevents co-adaptation of features
    drop = tf.nn.dropout(dense, keep_prob)

    # Squase the `num_hidden` features into `num_labels` classes --> Binary workload
    return tf.add(tf.matmul(drop, weights['out']), biases['out'])

y_conv = CNN(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(y_conv)

# Define loss function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_conv))

# Define optimiser
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# Evaluate the model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
# Confusion matrix
conf_mat = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(prediction, 1), num_classes=num_labels)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Wish to allocate approximately 80% of GPU memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.8)

saver = tf.train.Saver({'conv1_weights': weights['depth_conv1'], 'conv1_biases': biases['depth_conv1'],
                        'conv2_weights': weights['depth_conv2'], 'conv2_biases': biases['depth_conv2'],
                        'dense_weights': weights['dense'], 'dense_biases': biases['dense']})

# -----------------------------------------------
# Training the model
# -----------------------------------------------
with tf.device('/gpu:0'):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
    # Run the initializer
    sess.run(init)
    
    num_samples = data.train.num_examples
    
    # Training cycle
    for epoch in range(num_epochs):
        total_batch = int(num_samples / batch_size)
        
        # Loop over all batches
        for i in range(total_batch):
            batch_train_x, batch_train_y = data.train.next_batch(batch_size)
            
            # Run optimisation op (backprop)
            sess.run(train_op, feed_dict={X: batch_train_x, Y: batch_train_y, keep_prob: drop_rate})
        
        # Calculate batch loss and accuracy for both training and test sets
        loss_train, acc_train = sess.run([loss_op, accuracy], feed_dict={X: data.train.features, Y: data.train.labels, keep_prob: 1.0})
        print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss_train) + ", Training Accuracy= " + \
              "{:.3f}".format(acc_train))
              
        loss_test, acc_test, confy = sess.run([loss_op, accuracy, conf_mat], feed_dict={X: data.test.features, Y: data.test.labels, keep_prob: 1.0})
        print("--> Minibatch Loss= " + \
              "{:.4f}".format(loss_test) + ", Testing Accuracy= " + \
              "{:.3f}".format(acc_test))
        print("Conf Mat= ", confy)

    print("Optimisation Finished!")
    
# Create a saver object that will store all the parameter variables
saver.save(sess, model_path)
print("Model saved as: %s" % model_path)
