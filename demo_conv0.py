#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

from PIL import Image
from matplotlib import pyplot 
from tensorflow.python.ops import control_flow_ops


batch_size = 128
test_size = 256


#sess = tf.InteractiveSession()
sess = tf.Session()  # with sess.as_default()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 16)
                        strides=[1, 1, 1, 1], padding='SAME'))

    l1 = tf.nn.max_pool(l1a, ksize=[1, 4, 4, 1],              # l1 shape=(?, 7, 7, 16)
                        strides=[1, 4, 4, 1], padding='SAME')

    l1 = tf.reshape(l1, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 7*7*16)
    
    l4 = tf.nn.relu(tf.matmul(l1, w4))
    
    pyx = tf.matmul(l4, w_o)
    #saver = tf.train.Saver({"my_Xval": X[0,:, :, 1], "my_L1a":l1a[0, :, :, 15]})
    #save_path = saver.save(sess, "tmp/myVars.ckpt") 
    # print("Model saved in file: %s" % save_path)
    shape0=tf.shape(l1a)
    shape1=tf.slice(tf.shape(l1a),[1],[2]) # [?,28,28,16] --> [28,28]
    shape1=tf.concat(0,[[1], shape1]) # --> [1,28,28]
    l1a_0=tf.reshape(tf.slice(l1a,[0,0,0,0], [1,-1, -1, 1]), shape1) # tf.slice(arr, begin, size)
    l1a_15=tf.reshape(tf.slice(l1a,[0,0,0,15], [1,-1, -1, 1]),shape1)
    l1a_sample=tf.concat(0, [l1a_0, l1a_15])
    return pyx,  l1a_sample, shape0 # xxxx [l1a[0, :, :,0] , l1a[0, :, :, 15] ]

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])


#w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w = init_weights([3, 3, 1, 16])       # 3x3x1 conv, 16 outputs

w4 = init_weights([16 * 7 * 7, 32]) # FC 16 * 7 * 7 inputs, 32 outputs
#w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)
w_o = init_weights([32, 10])         # FC 128 inputs, 10 outputs (labels)
w2=[]
w3=[]

py_x,  L1a_sample, shape0 = model(X, w, w2, w3, w4, w_o)

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
        # fig.close()
        return
#====================================

# Launch the graph in a session
# with tf.Session() as sess:

# you need to initialize all variables
tf.initialize_all_variables().run(session=sess)

X_val=[]
L1a_val=[]

for i in range(10):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        t=0
        for start, end in training_batch:
            result=sess.run([train_op, py_x, L1a_sample, shape0], feed_dict={X: trX[start:end], Y: trY[start:end] })
            X_val=trX[start][:, :, 0]
            L1a_val=result[2] #[0,:,:,0:1]
            L1a_shape=result[3]
            #print L1a_shape
            #train_val=sess.run(train_op)
            #print type(L1a_val)
            #print L1a_val.shape
            print t
            t += 1
            
        myplot(X_val, L1a_val, i) 
         
          
        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        result=    sess.run([predict_op, L1a_sample], feed_dict={X: teX[test_indices],  Y: teY[test_indices]})
        pyx=result[0]
        X_val=teX[test_indices[0]][:, :, 0]
        L1a_val=result[1]
        # will L1a_val reflect the change of weightings ?
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == pyx) )
        myplot(X_val, L1a_val, "test"+ str(i))
        

sess.close()