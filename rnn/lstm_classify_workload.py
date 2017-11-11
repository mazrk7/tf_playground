from __future__ import print_function

import scipy.io
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

from dataset import load_data

# Training Parameters
learning_rate = 0.01
num_epochs = 200
batch_size = 100

# Read workload dataset, sequence length and input dimensionality from the workload Matlab matrix
data, seq_length, num_input = load_data('workload', one_hot=True, validation_size=10)

# Network Parameters
seq_length = seq_length.item(0)         # RNN sequence length (# of timesteps)
num_input = num_input.item(0)           # Input dimensionality
num_hidden = 300                        # RNN number of hidden units
num_classes = 2                         # Total number of classes --> Divided into binary classifier of beneficial or detrimental cognitive load
num_layers = 3                          # Number of stacked LSTM cells

# tf Graph input
X = tf.placeholder(tf.float32, [None, seq_length, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Define weights &biases
weights = {
    'dense': tf.Variable(tf.random_normal([num_input, num_hidden])),
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'dense': tf.Variable(tf.random_normal([num_hidden])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):
    # Construct the RNN cells
    # Reshape input into (seq_length*batch_size, num_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, num_input])
    
    dense = tf.nn.relu(tf.add(tf.matmul(x, weights['dense']), biases['dense']))
    # Splits the tensor into a list of sequence length
    dense = tf.split(dense, seq_length, 0)
    
    cell = rnn.DropoutWrapper(rnn.GRUCell(num_hidden, activation=tf.nn.relu), input_keep_prob=keep_prob)
    multi_layer_cell = rnn.MultiRNNCell([cell] * num_layers)              
    
    # Get RNN cell output
    outputs, final_state = rnn.static_rnn(multi_layer_cell, dense, dtype=tf.float32)
    
    # Linear activation, using output for the last time step of RNN
    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])
    
# -----------------------------------------------
# Create graph for training
# -----------------------------------------------    
logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# Define optimiser and perform gradient clipping to avoid exploding gradients
tvars = tf.trainable_variables()
gradients = tf.gradients(loss_op, tvars)
clipped_grads, _ = tf.clip_by_global_norm(gradients, 5)

optimiser = tf.train.AdamOptimizer(learning_rate)
train_op = optimiser.apply_gradients(zip(gradients, tvars))
            
# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Confusion matrix
conf_mat = tf.confusion_matrix(labels=tf.argmax(Y, 1), predictions=tf.argmax(prediction, 1), num_classes=num_classes)

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
            batch_train_x, batch_train_y = data.train.next_batch(batch_size)
            
            # Reshape data to get sequence of 'num_input' elements
            batch_train_x = batch_train_x.reshape((batch_size, seq_length, num_input)) 
             
            # Run optimisation op (backprop)
            sess.run(train_op, feed_dict={X: batch_train_x, Y: batch_train_y, keep_prob: 0.8})
            
        # Calculate batch loss and accuracy for both training and test sets
        loss_train, acc_train = sess.run([loss_op, accuracy], 
                                feed_dict={X: data.train.features.reshape((-1, seq_length, num_input)), Y: data.train.labels, keep_prob: 1.0})
        print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss_train) + ", Training Accuracy= " + \
              "{:.3f}".format(acc_train))
              
        loss_test, acc_test, confy = sess.run([loss_op, accuracy, conf_mat], 
                                feed_dict={X: data.test.features.reshape((-1, seq_length, num_input)), Y: data.test.labels, keep_prob: 1.0})
        print("--> Minibatch Loss= " + \
              "{:.4f}".format(loss_test) + ", Testing Accuracy= " + \
              "{:.3f}".format(acc_test))
        print("Conf Mat= ", confy)

    print("Optimisation Finished!")
