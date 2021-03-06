#!/usr/bin/env python

import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 1001)
# [-1,0, 1, 2, ...100]
# trX is a numpy.ndarray of shape (101,)
# trX.shape=(101,), shape : tuple of ints

# xxx print(typeof trX)
# f(arg1,arg2, *argx, **karg)
print(type(trX))

trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise
print(trX.shape)
print(trY.shape)

#exit()

X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")


def model(X, w):
    return tf.mul(X, w) # lr is just X*w so this model line is pretty simple


w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

cost = tf.square(Y - y_model) # use square error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.initialize_all_variables().run()

    for i in range(2):
        for (x, y) in zip(trX, trY):
            #print(x,y)
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))  # It should be something around 2
