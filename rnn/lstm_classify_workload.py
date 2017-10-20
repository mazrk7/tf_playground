from __future__ import print_function

import scipy.io
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
from tensorflow.contrib.data import Dataset, Iterator
from sklearn.model_selection import train_test_split

# Training Parameters
learning_rate = 0.001
num_epochs = 100
batch_size = 128
display_step = 5

# Network Parameters
num_input = 1       # Input dimensionality
seq_length = 250    # Sequence length
num_hidden = 128    # Hidden layer number of features
num_classes = 2     # Total number of classes ('NEL', 'L', 'M', 'H') --> Divided into binary classifier of beneficial cognitive load or not

# Import Matlab data matrix, which consists of pupil diameter data for the eye experiment (base & overlap)
dataset = scipy.io.loadmat('data.mat')
labels = scipy.io.loadmat('output.mat')

x_train, x_test, y_train, y_test = train_test_split(dataset['design'], labels['outputs'], test_size=0.33);

# tf Graph input
X = tf.placeholder("float", [None, seq_length, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights &biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# -----------------------------------------------
# Next Batch
# -----------------------------------------------
def next_batch(input_data, out_labels, batch_size, index_in_epoch, num_samples):
    start = index_in_epoch
    
    # Go to the next epoch
    if start + batch_size > num_samples:
        # Get the rest examples in this epoch
        rest_num_examples = num_samples - start
        input_rest_part = input_data[start:num_samples]
        labels_rest_part = out_labels[start:num_samples]
      
        # Shuffle the data
        perm = np.arange(num_samples)
        np.random.shuffle(perm)
        input_data = input_data[perm]
        out_labels = out_labels[perm]
      
        # Start next epoch
        start = 0
        index_in_epoch = batch_size - rest_num_examples
        end = index_in_epoch
        input_new_part = input_data[start:end]
        labels_new_part = out_labels[start:end]
      
        return np.concatenate((input_rest_part, input_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0), index_in_epoch
    else:
        index_in_epoch += batch_size
        end = index_in_epoch
        
        return input_data[start:end], out_labels[start:end], index_in_epoch
        
# -----------------------------------------------
# LSTM-RNN from TensorFlow examples
# -----------------------------------------------
def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, seq_length, num_input)
    # Required shape: 'seq_length' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'seq_length' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, seq_length, num_input)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.DropoutWrapper(rnn.LSTMCell(num_hidden, forget_bias=1.0))

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# -----------------------------------------------
# Create graph for training
# -----------------------------------------------    
logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Confusion matrix
conf_mat = tf.contrib.metrics.confusion_matrix(tf.argmax(Y, 1), tf.argmax(prediction, 1))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# -----------------------------------------------
# Training the model
# -----------------------------------------------
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    
    num_samples = len(x_train)
    index_in_epoch = 0
    
    # Training cycle
    for epoch in range(num_epochs):
        total_batch = int(num_samples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y, index_in_epoch = next_batch(x_train, y_train, batch_size, index_in_epoch, num_samples)
            
            # Reshape data to get sequence of 'num_input' elements
            batch_x = batch_x.reshape((batch_size, seq_length, 1))
            
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            
        if epoch % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 1028 test samples
    test_len = 1028
    test_data = x_test.reshape((-1, seq_length, num_input))
    test_label = y_test
    print("No. of non-workload cases:", np.size(np.where(test_label[:,0] == 1)))
    print("Test length:", len(x_test))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    print("Confusion Matrix:", sess.run(conf_mat, feed_dict={X: test_data, Y: test_label}))
