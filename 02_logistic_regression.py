#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

print(teY.shape)
print(np.argmax(teY, axis=1).shape)
print(np.mean([True,True,True, False, True]))
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            # print( sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end]}) )
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            # print(sess.run(w)[1][1]) xxxx print(w[1][1])
        w_val=sess.run(w)    
        print("weighting adapting: ", w_val[0], w_val[100])    
        
        print(i, np.mean(np.argmax(teY, axis=1) ==
                 sess.run(predict_op, feed_dict={X: teX, Y: teY})))
