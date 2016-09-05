#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, showimg=False):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    
    pyx = tf.matmul(l4, w_o)
    if(showimg==True):
        #====================================
import numpy
#import pylab
from PIL import Image
from matplotlib import pyplot 

# open random image of dimensions 550x580, 516x639
img = Image.open(open('3wolfmoon.jpg')) # brainLR
w,h=img.size
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, h, w)
filtered_img = f(img_)

# plot original image and first and second components of output
fig = pyplot.figure()
ax = fig.add_subplot(131)
# pylab.subplot(1, 3, 1); pylab.axis('off'); 
ax.axis('off');
pyplot.imshow(img, cmap='gray')
#pylab.gray();

# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
ax=fig.add_subplot(132); ax.axis('off'); pyplot.imshow(filtered_img[0, 0, :, :], cmap='gray')
ax=fig.add_subplot(133); ax.axis('off'); pyplot.imshow(filtered_img[0, 1, :, :], cmap='gray')
#pylab.show()
pyplot.savefig('lenet_ex2.jpg')
     #====================================
    return pyx

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

py_x = model(X, w, w2, w3, w4, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end] })

        # test_indices = np.arange(len(teX)) # Get A Test Batch
        # np.random.shuffle(test_indices)
        # test_indices = test_indices[0:test_size]

        # print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                    # sess.run(predict_op, feed_dict={X: teX[test_indices],  Y: teY[test_indices]})))

