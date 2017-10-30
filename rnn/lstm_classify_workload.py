from __future__ import print_function

import scipy.io
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

from dataset import load_data

# Training Parameters
learning_rate = 0.001
num_epochs = 250
batch_size = 80

# Network Parameters
num_input = 4       # Input dimensionality
seq_length = 400    # Sequence length
num_hidden = 100    # Hidden layer number of features
num_classes = 2     # Total number of classes --> Divided into binary classifier of beneficial or detrimental cognitive load

data = load_data('workload', one_hot=True, validation_size=10)

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
# LSTM-RNN from TensorFlow examples
# -----------------------------------------------
def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, seq_length, num_input)
    # Required shape: 'seq_length' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'seq_length' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, seq_length, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden)#rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden), rnn.BasicLSTMCell(num_hidden)]))

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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
    
    num_samples = data.train.num_examples
    
    # Training cycle
    for epoch in range(num_epochs):
        total_batch = int(num_samples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = data.train.next_batch(batch_size)

            # Reshape data to get sequence of 'num_input' elements
            batch_x = batch_x.reshape((batch_size, seq_length, num_input)) 

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
        print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for all test samples
    test_data = data.test.features.reshape((-1, seq_length, num_input))
    test_label = data.test.labels
    print("No. of non-workload cases:", np.size(np.where(test_label[:,0] == 1)))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    print("Confusion Matrix:", sess.run(conf_mat, feed_dict={X: test_data, Y: test_label}))
