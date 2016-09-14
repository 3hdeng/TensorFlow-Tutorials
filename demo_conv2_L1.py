#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

from PIL import Image
from matplotlib import pyplot 
from tensorflow.python.ops import control_flow_ops


#=========
def myplot(X_val, L1a_val, L_val, m, i):    
        #====================================
        print 'myplot to save the images' 
        # plot original image and first and second components of output
        # >9 plot, fig.add_subplot(4,4,11) , index start with 1
        fig = pyplot.figure()
        ax = fig.add_subplot(3,8,1)
        ax.axis('off');
        pyplot.imshow(X_val[m,:,:,0]) #, cmap='gray')
        
        # recall that the convOp output (filtered image) is actually a "minibatch",
        # of size 1 here, so we take index 0 in the first dimension:
        for k in range(0, 16,2):
           ax=fig.add_subplot(3,8,k/2 +9); ax.axis('off'); pyplot.imshow(L1a_val[m,:,:,k])#, cmap='gray')
           ax=fig.add_subplot(3,8,k/2 +17); ax.axis('off'); pyplot.imshow(L1_val[m,:,:,k])#, cmap='gray')
        pyplot.savefig('t{0}L1_{1}.jpg'.format(m,i) )
        # fig.close()
        
#====================================
# input numpy ndarray
def calc_shape(l1a):
    shape0=l1a.shape
    #print 'l1a.shape = {0}'.format(shape0)
    shape1=shape0[1:3] # (?,28,28,16) --> (28,28)
    #print 'l1a.shape trucated to  {0}'.format(shape1)
    shape1=np.append([2], shape1)# --> [2,28,28]
    #print 'l1a.shape trucated and reshaped to  {0}'.format(shape1)
    return shape1
#=====================================
def Ln_sampling(l1a):
    l1a_0=l1a[0,:,:,0] # the 0-th output feature for the 0th input, [28,28]
    #print 'l1a sample0 shape = {0}'.format(l1a_0.shape)
    l1a_15=l1a[0,:,:,15] # the 15-th output feature for the 0th input
    #l1a_sample=tf.concat(0, [l1a_0, l1a_15]) # concate at dim 0
    return np.append([l1a_0], [l1a_15], 0)
#==============================    

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 16)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 16)
                        strides=[1, 2, 2, 1], padding='SAME')
    #l1 = tf.nn.max_pool(l1a, ksize=[1, 4, 4, 1],              # l1 shape=(?, 7, 7, 16)
    #                    strides=[1, 4, 4, 1], padding='SAME')

    
    l1_reshape = tf.reshape(l1, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 7*7*32)
    
    l4 = tf.nn.relu(tf.matmul(l1_reshape, w4))
    pyx = tf.matmul(l4, w_o)
    return pyx, l1a, l1

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])


w = init_weights([3, 3, 1, 16])       # 3x3x1 conv, 16 outputs
w4 = init_weights([16 * 14 * 14, 32]) # FC 16 * 14 * 14 inputs, 64 outputs
#w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)
w_o = init_weights([32, 10])         # FC 128 inputs, 10 outputs (labels)
w2=[]
w3=[]

py_x, L1a, L1 = model(X, w, w2, w3, w4, w_o)

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

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        result=    sess.run([predict_op, L1a, L1], feed_dict={X: teX[test_indices],  Y: teY[test_indices]})
        argMax=result[0]
        """
        print '=== argMax ============='
        print type(argMax)
        print len(argMax)
        """
        X_val=teX[test_indices]
        L1a_val=result[1]
        L1_val=result[2]
        # will L1a_val reflect the change of weightings ?
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == argMax) )
                          # sess.run(predict_op, feed_dict={sess: sess, showimg:True, i_tensor: i, X: teX[test_indices],  Y: teY[test_indices]})))
        
        print type(X_val)
        print X_val.shape
        print type(L1_val)
        print L1_val.shape
        print '============================='
        
        myplot(X_val, L1a_val, L1_val, 0, i)        
        myplot(X_val, L1a_val, L1_val, 31, i)  
