#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

from PIL import Image
from matplotlib import pyplot 
from tensorflow.python.ops import control_flow_ops


batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, showimg=False, i_tensor=0):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    #l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
    #                    strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.max_pool(l1a, ksize=[1, 4, 4, 1],              # l1 shape=(?, 7, 7, 32)
                        strides=[1, 4, 4, 1], padding='SAME')

    #l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
    #                    strides=[1, 1, 1, 1], padding='SAME'))
    #l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
    #                    strides=[1, 2, 2, 1], padding='SAME')
    
    #l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
    #                    strides=[1, 1, 1, 1], padding='SAME'))
    #l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
    #                    strides=[1, 2, 2, 1], padding='SAME')
    #l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l1 = tf.reshape(l1, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 7*7*32)
    
    l4 = tf.nn.relu(tf.matmul(l1, w4))
    
    pyx = tf.matmul(l4, w_o)
    # if showimg == True :
    def if_true():    
        #====================================
        print 'showimg is True, save the conved imges' 
        # plot original image and first and second components of output
        #fig = pyplot.figure()
        #ax = fig.add_subplot(131)
        #ax.axis('off');
        #pyplot.imshow(X.eval(session=sess)[0,:, :, 1]) #, cmap='gray')
        
        # recall that the convOp output (filtered image) is actually a "minibatch",
        # of size 1 here, so we take index 0 in the first dimension:
        #ax=fig.add_subplot(132); ax.axis('off'); pyplot.imshow(l1a[0, :, :,0])#, cmap='gray')
        #ax=fig.add_subplot(133); ax.axis('off'); pyplot.imshow(l1a[0, :, :, 30])#, cmap='gray')
        #pyplot.savefig('l1a_{0}.jpg'.format(i_tensor) )
        #====================================
        return 1
    
    def if_false():
       return 0
       
    #============
    #control_flow_ops.cond(tf.Variable(showimg), if_true, if_false)
    #============   
    return pyx, X[0,:, :, 1], [l1a[0, :, :,0] , l1a[0, :, :, 30] ]

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])
showimg=tf.Variable(False) #xxx tf.placeholder('bool', False)
i_tensor=tf.Variable(0)
T=tf.constant(True)

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
#w = init_weights([3, 3, 1, 16])       # 3x3x1 conv, 16 outputs
#w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
#w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
# w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w4 = init_weights([32 * 7 * 7, 64]) # FC 32 * 7 * 7 inputs, 64 outputs
#w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)
w_o = init_weights([64, 10])         # FC 128 inputs, 10 outputs (labels)
w2=[]
w3=[]

py_x, X_val, L1a_val = model(X, w, w2, w3, w4, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)


#=========
def myplot(X_val, L1a_val,i):    
        #====================================
        print 'showimg is True, save the conved imges' 
        # plot original image and first and second components of output
        fig = pyplot.figure()
        ax = fig.add_subplot(131)
        ax.axis('off');
        pyplot.imshow(X_val) #, cmap='gray')
        
        # recall that the convOp output (filtered image) is actually a "minibatch",
        # of size 1 here, so we take index 0 in the first dimension:
        ax=fig.add_subplot(132); ax.axis('off'); pyplot.imshow(L1a_val[0])#, cmap='gray')
        ax=fig.add_subplot(133); ax.axis('off'); pyplot.imshow(L1a_val[1])#, cmap='gray')
        pyplot.savefig('l1a_{0}.jpg'.format(i) )
        fig.close()
        return
#====================================

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end] })

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        pyx=    sess.run(predict_op, feed_dict={X: teX[test_indices],  Y: teY[test_indices]})
        # will L1a_val reflect the change of weightings ?
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == pyx) )
                          # sess.run(predict_op, feed_dict={sess: sess, showimg:True, i_tensor: i, X: teX[test_indices],  Y: teY[test_indices]})))
        print type(X_val)
        print X_val.shape()
        print type(L1a_val)
        print L1a_val.shape()
        myplot(X_val, L1a_val, i)        

